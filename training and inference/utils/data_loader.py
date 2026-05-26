"""
Data loading utilities for physics trajectory dataset
Supports train/test splits and language conditioning
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

STATE_MEAN = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0, 0.5, 0.2])
STATE_STD = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 0.6, 0.1])

def normalize_states(states):
    """
    Robust normalization WITHOUT log transforms
    Simple clip + scale approach that won't cause NaN/Inf
    """
    normalized = states.copy()
    
    # Positions (indices 0,1,2): -2.5 to 2.5 range
    normalized[:, :, 0:3] = states[:, :, 0:3] / 5.0
    
    # Velocities (indices 3,4,5): can be -10 to 10
    normalized[:, :, 3:6] = np.clip(states[:, :, 3:6], -10, 10) / 10.0
    
    # Mass (index 6): 0.05 to 15.0 - just linear scale
    normalized[:, :, 6:7] = np.clip(states[:, :, 6:7], 0.05, 15.0) / 15.0
    
    # Friction (index 7): 0.001 to 2.0
    normalized[:, :, 7:8] = np.clip(states[:, :, 7:8], 0.001, 2.0) / 2.0
    
    # Scale (index 8): 0.08 to 0.35
    normalized[:, :, 8:9] = np.clip(states[:, :, 8:9], 0.08, 0.35) / 0.35
    
    assert not np.isnan(normalized).any(), "NaN in normalized states"
    assert not np.isinf(normalized).any(), "Inf in normalized states"
    
    return normalized


class PhysicsTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for loading physics trajectories with language conditioning
    """
    
    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        sequence_length: int = 20,
        prediction_horizon: int = 10,
        max_blocks: int = 6,
        state_dim_per_block: int = 9,
        language_encoder: Optional[SentenceTransformer] = None,
        seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_blocks = max_blocks
        self.state_dim_per_block = state_dim_per_block
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        with open(self.dataset_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        if split in ["control_shuffled", "control_wrong_rule"]:
            self.episodes = self.metadata[split]
            self.subdir = "control_ablation"
        else:
            self.episodes = self.metadata[split]
            self.subdir = "trajectories"
        
        self.rule_texts = self.metadata["rules"]
        self.gravity_texts = self.metadata["gravity_rules"]
        
        # Auto-detect max_blocks from dataset
        self.max_blocks = max_blocks
        detected_max = 0
        for ep_info in self.episodes[:min(10, len(self.episodes))]:
            ep_id = ep_info["episode_id"]
            episode_file = self.dataset_dir / self.subdir / f"episode_{ep_id:04d}.npz"
            data = np.load(episode_file)
            detected_max = max(detected_max, int(data["num_blocks"]))
        
        if detected_max > self.max_blocks:
            print(f"Warning: Dataset has {detected_max} blocks, updating max_blocks from {self.max_blocks}")
            self.max_blocks = detected_max
        
        self.cache = {}
        
        # PRE-COMPUTE all language embeddings to avoid CUDA in workers
        self.language_embeddings = {}
        if language_encoder is not None:
            print(f"Pre-computing language embeddings for {len(self.episodes)} episodes...")
            language_encoder.to('cpu')
            for ep_info in self.episodes:
                ep_id = ep_info["episode_id"]
                episode = self.load_episode(ep_id)
                rule_text = episode["rule_text"]
                emb = language_encoder.encode(
                    rule_text, 
                    convert_to_tensor=True, 
                    show_progress_bar=False,
                    device='cpu'
                )
                self.language_embeddings[ep_id] = emb
        
        print(f"Loaded {len(self.episodes)} episodes from {split} split (max_blocks={self.max_blocks})")
    
    def __len__(self):
        return len(self.episodes)
    
    def load_episode(self, episode_id: int) -> Dict:
        """Load a single episode from disk"""
        if episode_id in self.cache:
            return self.cache[episode_id]
        
        episode_file = self.dataset_dir / self.subdir / f"episode_{episode_id:04d}.npz"
        data = np.load(episode_file)
        
        positions = data["positions"]  # [T, num_blocks, 3]
        velocities = data["velocities"]  # [T, num_blocks, 3]
        masses = data["masses"]  # [T, num_blocks]
        frictions = data["frictions"]  # [T, num_blocks]
        scales = data["scales"]  # [T, num_blocks]
        
        T, num_blocks = positions.shape[0], positions.shape[1]
        states = np.zeros((T, num_blocks, 9))
        states[:, :, 0:3] = positions
        states[:, :, 3:6] = velocities
        states[:, :, 6] = masses
        states[:, :, 7] = frictions
        states[:, :, 8] = scales
        
        episode = {
            "states": states,
            "rule": str(data["rule_key"]),
            "gravity": str(data["gravity_key"]),
            "rule_text": str(data["rule_text"]),
            "num_blocks": int(data["num_blocks"]),
        }
        
        self.cache[episode_id] = episode
        return episode
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        episode_info = self.episodes[idx]
        episode_id = episode_info["episode_id"]
        
        episode = self.load_episode(episode_id)
        states = episode["states"]  # [T, num_blocks, 9]
        num_blocks = episode["num_blocks"]
        
        T = len(states)
        context_length = self.sequence_length
        total_length = context_length + self.prediction_horizon
        
        if T < total_length:
            start_idx = 0
            states_padded = np.zeros((total_length, self.max_blocks, self.state_dim_per_block))
            states_padded[:T, :num_blocks, :] = states
        else:
            start_idx = np.random.randint(0, T - total_length + 1)
            states_window = states[start_idx : start_idx + total_length]
            
            states_padded = np.zeros((total_length, self.max_blocks, self.state_dim_per_block))
            states_padded[:, :num_blocks, :] = states_window
        
        states_padded = normalize_states(states_padded)
        
        context = states_padded[:context_length]
        target = states_padded[context_length:]
        
        context_flat = context.reshape(context_length, -1)
        target_flat = target.reshape(self.prediction_horizon, -1)
        
        mask = np.zeros(self.max_blocks)
        mask[:num_blocks] = 1.0
        
        if episode_id in self.language_embeddings:
            language_emb = self.language_embeddings[episode_id]
        else:
            language_emb = torch.zeros(384)
        
        if np.random.random() < 0.01:
            print(f"Context stats - min: {context_flat.min():.2f}, max: {context_flat.max():.2f}, "
                  f"mean: {context_flat.mean():.2f}, std: {context_flat.std():.2f}")
        
        return {
            "context": torch.from_numpy(context_flat).float(),
            "target": torch.from_numpy(target_flat).float(),
            "language": language_emb.cpu() if isinstance(language_emb, torch.Tensor) else language_emb,
            "mask": torch.from_numpy(mask).float(),
            "rule": episode["rule"],
            "gravity": episode["gravity"],
            "episode_id": episode_id,
        }


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    sequence_length: int = 20,
    prediction_horizon: int = 10,
    num_workers: int = 4,
    language_model: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits
    """
    print(f"Loading language encoder: {language_model}")
    language_encoder = SentenceTransformer(language_model)
    language_encoder.eval()
    
    dataloaders = {}
    
    splits = ["train", "test_compositional", "test_contradiction", 
              "control_shuffled", "control_wrong_rule"]
    
    for split in splits:
        dataset = PhysicsTrajectoryDataset(
            dataset_dir=dataset_dir,
            split=split,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            language_encoder=language_encoder,
            seed=seed,
        )
        
        shuffle = (split == "train")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders, language_encoder


if __name__ == "__main__":
    dataset_dir = "/home/uap-lab-pc/world_model/dataset/physics_dataset"
    
    dataloaders, encoder = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=4,
        sequence_length=20,
        prediction_horizon=10,
    )
    
    print("\nTesting data loading...")
    for split, loader in dataloaders.items():
        batch = next(iter(loader))
        print(f"\n{split}:")
        print(f"  Context shape: {batch['context'].shape}")
        print(f"  Target shape: {batch['target'].shape}")
        print(f"  Language shape: {batch['language'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
