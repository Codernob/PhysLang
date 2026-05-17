"""
Data loading utilities for the physics dataset (FIXED VERSION)
Use this to load and process the generated trajectories for model training

FIXES APPLIED:
- Loads NPZ format (much faster than JSON)
- State vectors exclude color (prevents rule leakage)
- Includes mass and friction for verification
- Supports ablation control conditions
- Deterministic loading with seeds
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class PhysicsTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for loading physics trajectories
    FIXED: Now loads NPZ format and excludes color from state
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 split: str = "train",
                 sequence_length: int = 20,
                 prediction_horizon: int = 5,
                 include_ablation: bool = False,
                 seed: int = 42):
        """
        Args:
            dataset_dir: Path to the dataset directory
            split: "train", "test_compositional", "test_contradiction", or "control_ablation"
            sequence_length: Number of past states to use as context
            prediction_horizon: Number of future states to predict
            include_ablation: Whether to include ablation control data
            seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load metadata
        with open(self.dataset_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Get episode IDs for this split
        self.episodes = metadata[split]
        self.rule_texts = metadata["rules"]
        
        # Determine subdirectory
        if split == "control_ablation":
            self.subdir = "control_ablation"
        else:
            self.subdir = "trajectories"
        
        print(f"Loaded {len(self.episodes)} {split} episodes")
        
    def __len__(self):
        return len(self.episodes)
    
    def load_episode(self, episode_id: int) -> Dict:
        """Load a single episode from disk (NPZ format)"""
        episode_file = self.dataset_dir / self.subdir / f"episode_{episode_id:04d}.npz"
        
        # Load NPZ file
        data = np.load(episode_file, allow_pickle=True)
        
        # Convert to dictionary
        episode = {
            "episode_id": int(data["episode_id"]),
            "rule_text": str(data["rule_text"]),
            "rule_key": str(data["rule_key"]),
            "gravity_text": str(data["gravity_text"]),
            "gravity_key": str(data["gravity_key"]),
            "gravity_vector": data["gravity_vector"],
            "ablation_mode": str(data["ablation_mode"]) if "ablation_mode" in data else "none",
            # Trajectory data
            "positions": data["positions"],  # [timesteps, num_blocks, 3]
            "velocities": data["velocities"],  # [timesteps, num_blocks, 3]
            "masses": data["masses"],  # [timesteps, num_blocks]
            "frictions": data["frictions"],  # [timesteps, num_blocks]
            "scales": data["scales"],  # [timesteps, num_blocks]
            "times": data["times"],  # [timesteps]
        }
        
        return episode
    
    def extract_state_vector(self, positions: np.ndarray, velocities: np.ndarray,
                            masses: np.ndarray, frictions: np.ndarray,
                            scales: np.ndarray) -> np.ndarray:
        """
        FIXED: Convert state to flat vector WITHOUT color (prevents rule leakage)
        
        State vector per block: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, friction, scale]
        Total dimension: num_blocks * 9
        
        Args:
            positions: [num_blocks, 3]
            velocities: [num_blocks, 3]
            masses: [num_blocks]
            frictions: [num_blocks]
            scales: [num_blocks]
            
        Returns:
            state_vector of shape [num_blocks * 9]
        """
        num_blocks = positions.shape[0]
        
        state_vectors = []
        for i in range(num_blocks):
            block_vec = np.concatenate([
                positions[i],      # [3] position
                velocities[i],     # [3] velocity
                [masses[i]],       # [1] mass
                [frictions[i]],    # [1] friction
                [scales[i]],       # [1] scale
            ])
            state_vectors.append(block_vec)
        
        # Flatten: [num_blocks * 9]
        state_vector = np.concatenate(state_vectors)
        return state_vector
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, Dict]:
        """
        Get a single training sample
        
        Returns:
            context_states: [sequence_length, state_dim]
            rule_text: String describing the rule
            future_states: [prediction_horizon, state_dim]
            metadata: Dict with additional info (gravity, masses, etc.)
        """
        episode_meta = self.episodes[idx]
        episode = self.load_episode(episode_meta["episode_id"])
        
        # Extract dimensions
        num_timesteps = episode["positions"].shape[0]
        num_blocks = episode["positions"].shape[1]
        
        # Convert to state vectors for each timestep
        states = []
        for t in range(num_timesteps):
            state_vec = self.extract_state_vector(
                episode["positions"][t],
                episode["velocities"][t],
                episode["masses"][t],
                episode["frictions"][t],
                episode["scales"][t]
            )
            states.append(state_vec)
        
        states = np.array(states)  # [num_timesteps, state_dim]
        
        # Sample a random starting point for the sequence
        max_start = len(states) - self.sequence_length - self.prediction_horizon
        if max_start > 0:
            start_idx = np.random.randint(0, max_start)
        else:
            start_idx = 0
        
        # Extract context and future
        context_end = start_idx + self.sequence_length
        future_end = context_end + self.prediction_horizon
        
        context_states = states[start_idx:context_end]
        future_states = states[context_end:future_end]
        
        # Pad if necessary
        if len(context_states) < self.sequence_length:
            pad_length = self.sequence_length - len(context_states)
            context_states = np.pad(context_states, 
                                   ((pad_length, 0), (0, 0)), 
                                   mode='edge')
        
        if len(future_states) < self.prediction_horizon:
            pad_length = self.prediction_horizon - len(future_states)
            future_states = np.pad(future_states,
                                  ((0, pad_length), (0, 0)),
                                  mode='edge')
        
        # Metadata for verification/analysis
        metadata = {
            "episode_id": episode["episode_id"],
            "rule_key": episode["rule_key"],
            "gravity_key": episode["gravity_key"],
            "gravity_vector": episode["gravity_vector"],
            "ablation_mode": episode["ablation_mode"],
            "num_blocks": num_blocks,
            "masses": episode["masses"][0],  # Initial masses
            "frictions": episode["frictions"][0],  # Initial frictions
        }
        
        return (
            torch.FloatTensor(context_states),
            episode["rule_text"],
            torch.FloatTensor(future_states),
            metadata
        )


def create_dataloaders(dataset_dir: str,
                      batch_size: int = 32,
                      sequence_length: int = 20,
                      prediction_horizon: int = 5,
                      num_workers: int = 4,
                      seed: int = 42):
    """
    Create train and test dataloaders with proper splits
    
    Args:
        dataset_dir: Path to the dataset
        batch_size: Batch size
        sequence_length: Number of past states
        prediction_horizon: Number of future states to predict
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of dataloaders: {
            "train": train_loader,
            "test_compositional": compositional_loader,
            "test_contradiction": contradiction_loader,
            "ablation": ablation_loader (optional)
        }
    """
    
    # Create datasets for each split
    train_dataset = PhysicsTrajectoryDataset(
        dataset_dir=dataset_dir,
        split="train",
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        seed=seed
    )
    
    compositional_dataset = PhysicsTrajectoryDataset(
        dataset_dir=dataset_dir,
        split="test_compositional",
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        seed=seed
    )
    
    contradiction_dataset = PhysicsTrajectoryDataset(
        dataset_dir=dataset_dir,
        split="test_contradiction",
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        seed=seed
    )
    
    ablation_dataset = PhysicsTrajectoryDataset(
        dataset_dir=dataset_dir,
        split="control_ablation",
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        seed=seed
    )
    
    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test_compositional": DataLoader(
            compositional_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test_contradiction": DataLoader(
            contradiction_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "ablation": DataLoader(
            ablation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
    }
    
    return dataloaders


def analyze_dataset_statistics(dataset_dir: str):
    """
    Analyze and print dataset statistics
    Useful for verifying rule compliance and data quality
    """
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    metadata_path = Path(dataset_dir) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nSeed: {metadata['seed']}")
    print(f"\nSplits:")
    print(f"  Train: {len(metadata['train'])} episodes")
    print(f"  Compositional Test: {len(metadata['test_compositional'])} episodes")
    print(f"  Contradiction Test: {len(metadata['test_contradiction'])} episodes")
    print(f"  Ablation Control: {len(metadata['control_ablation'])} episodes")
    
    # Analyze rule distribution
    print(f"\nRule Distribution:")
    for split_name in ["train", "test_compositional", "test_contradiction"]:
        rules = [ep["rule"] for ep in metadata[split_name]]
        unique_rules = set(rules)
        print(f"\n  {split_name}:")
        for rule in sorted(unique_rules):
            count = rules.count(rule)
            print(f"    {rule}: {count} episodes ({metadata['rules'][rule]})")
    
    # Load and analyze a sample episode
    print(f"\nSample Episode Analysis:")
    dataset = PhysicsTrajectoryDataset(dataset_dir, split="train", sequence_length=10)
    context, rule_text, future, meta = dataset[0]
    
    print(f"  Context shape: {context.shape}")
    print(f"  Future shape: {future.shape}")
    print(f"  Rule text: {rule_text}")
    print(f"  State dimension: {context.shape[1]} (= num_blocks * 9)")
    print(f"  Number of blocks: {meta['num_blocks']}")
    print(f"  Masses: {meta['masses']}")
    print(f"  Frictions: {meta['frictions']}")
    print(f"  Gravity: {meta['gravity_vector']}")
    
    print("\n" + "="*60)


def test_dataloader():
    """Test the dataloader with sample data"""
    print("Testing dataloader...")
    
    # Create sample loaders
    loaders = create_dataloaders(
        dataset_dir="./physics_dataset",
        batch_size=4,
        sequence_length=10,
        prediction_horizon=5
    )
    
    # Test each split
    for split_name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"Testing {split_name} split")
        print(f"{'='*60}")
        
        # Get a batch
        context, rules, future, metadata = next(iter(loader))
        
        print(f"\nBatch shapes:")
        print(f"  Context states: {context.shape}")  # [batch, seq_len, state_dim]
        print(f"  Future states: {future.shape}")    # [batch, pred_horizon, state_dim]
        print(f"  Rules: {len(rules)} text strings")
        
        print(f"\nSample rules:")
        for i, rule in enumerate(rules[:min(3, len(rules))]):
            print(f"  {i+1}. {rule}")
        
        print(f"\nMetadata keys: {list(metadata.keys())}")
        print(f"  Episode IDs: {metadata['episode_id'][:3].tolist()}")
        print(f"  Gravity keys: {[metadata['gravity_key'][i] for i in range(min(3, len(metadata['gravity_key'])))]}")
    
    print("\n✓ All dataloaders tested successfully!")
    
    # Analyze dataset
    analyze_dataset_statistics("./physics_dataset")


if __name__ == "__main__":
    test_dataloader()
