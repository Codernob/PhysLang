"""
Evaluation script for trained models
Comprehensive diagnostic analysis across all splits
"""

import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch

sys.path.append(str(Path(__file__).parent))

from configs.train_config import (
    DATASET_CONFIG, MODEL_CONFIGS, LANGUAGE_CONFIG, STATE_DIM, HARDWARE_CONFIG
)
from utils.data_loader import create_dataloaders
from utils.metrics import (
    compute_trajectory_metrics, compute_language_necessity_score,
    compute_split_comparison, compute_ablation_analysis
)
from models import create_model


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint"""
    model = create_model(
        model_config=MODEL_CONFIGS[model_name],
        state_dim=STATE_DIM,
        language_dim=LANGUAGE_CONFIG["embedding_dim"],
        sequence_length=DATASET_CONFIG["sequence_length"],
        prediction_horizon=DATASET_CONFIG["prediction_horizon"],
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloaders: dict,
    device: torch.device,
    splits: list,
) -> dict:
    """Evaluate model on multiple splits"""
    results = {}
    criterion = torch.nn.MSELoss()
    
    for split in splits:
        if split not in dataloaders:
            continue
        
        print(f"\nEvaluating on {split}...")
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloaders[split]:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            language = batch["language"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)
            
            predictions = model(context, language, mask)
            loss = criterion(predictions, target)
            
            all_predictions.append(predictions)
            all_targets.append(target)
            total_loss += loss.item()
            num_batches += 1
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_trajectory_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / num_batches
        
        results[split] = metrics
        
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Position MSE: {metrics['position_mse']:.6f}")
    
    return results


def plot_results(all_results: dict, output_dir: Path):
    """Create visualization of results"""
    sns.set_style("whitegrid")
    
    models = list(all_results.keys())
    splits = ["train", "test_compositional", "test_contradiction"]
    
    # MSE comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_to_plot = ["mse", "position_mse", "velocity_mse"]
    titles = ["Overall MSE", "Position MSE", "Velocity MSE"]
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        data = []
        for split in splits:
            split_data = []
            for model in models:
                if split in all_results[model]:
                    split_data.append(all_results[model][split][metric])
                else:
                    split_data.append(np.nan)
            data.append(split_data)
        
        data = np.array(data)
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, split in enumerate(splits):
            ax.bar(x + i * width, data[i], width, label=split)
        
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_dir / 'comparison.png'}")
    
    # Generalization gap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    comp_gaps = []
    contra_gaps = []
    
    for model in models:
        train_mse = all_results[model]["train"]["mse"]
        comp_mse = all_results[model].get("test_compositional", {}).get("mse", train_mse)
        contra_mse = all_results[model].get("test_contradiction", {}).get("mse", train_mse)
        
        comp_gaps.append((comp_mse - train_mse) / train_mse * 100)
        contra_gaps.append((contra_mse - train_mse) / train_mse * 100)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, comp_gaps, width, label="Compositional")
    ax.bar(x + width/2, contra_gaps, width, label="Contradiction")
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Generalization Gap (%)")
    ax.set_title("Generalization Gap: Test MSE vs Train MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "generalization_gap.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {output_dir / 'generalization_gap.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/uap-lab-pc/world_model/training_code/outputs",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp_baseline", "gru", "transformer", "mamba"],
        help="Models to evaluate",
    )
    args = parser.parse_args()
    
    device = torch.device(HARDWARE_CONFIG["device"])
    output_dir = Path(args.output_dir)
    
    # Load data
    print("Loading dataset...")
    dataloaders, _ = create_dataloaders(
        dataset_dir=DATASET_CONFIG["dataset_dir"],
        batch_size=DATASET_CONFIG["batch_size"],
        sequence_length=DATASET_CONFIG["sequence_length"],
        prediction_horizon=DATASET_CONFIG["prediction_horizon"],
        num_workers=DATASET_CONFIG["num_workers"],
        language_model=LANGUAGE_CONFIG["model_name"],
        seed=DATASET_CONFIG["seed"],
    )
    
    # Evaluate all models
    all_results = {}
    splits = ["train", "test_compositional", "test_contradiction", 
              "control_shuffled", "control_wrong_rule"]
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        checkpoint_path = output_dir / model_name / f"{model_name}_best.pt"
        
        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            continue
        
        model = load_model(model_name, checkpoint_path, device)
        
        results = evaluate_model(model, dataloaders, device, splits)
        all_results[model_name] = results
        
        split_comp = compute_split_comparison(results)
        ablation = compute_ablation_analysis(results)
        
        print(f"\nDiagnostics:")
        print(f"  Compositional gap: {split_comp['compositional_gap']:.6f}")
        print(f"  Contradiction gap: {split_comp['contradiction_gap']:.6f}")
        print(f"  Ablation sensitivity: {ablation['ablation_sensitivity']:.6f}")
        
        results["diagnostics"] = {
            "split_comparison": split_comp,
            "ablation": ablation,
        }
        
        results_file = output_dir / model_name / f"{model_name}_eval.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {results_file}")
    
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plots...")
        print(f"{'='*60}")
        plot_results(all_results, output_dir)
    
    combined_file = output_dir / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved combined results to {combined_file}")


if __name__ == "__main__":
    main()
