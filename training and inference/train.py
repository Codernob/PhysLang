"""
Main training script for language-grounded world models
Supports multiple architectures and diagnostic evaluation
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# Add project to path
sys.path.append(str(Path(__file__).parent))

from configs.train_config import (
    DATASET_CONFIG, TRAIN_CONFIG, MODEL_CONFIGS, LANGUAGE_CONFIG,
    STATE_DIM, EVAL_CONFIG, HARDWARE_CONFIG
)
from utils.data_loader import create_dataloaders
from utils.metrics import (
    compute_trajectory_metrics, compute_rule_grounding_score,
    compute_language_necessity_score, compute_split_comparison,
    compute_ablation_analysis
)
from models import create_model, count_parameters


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0


def reset_gpu_memory():
    """Reset GPU memory tracker"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def diagnose_language_usage(model, dataloader, device):
    """Check if model actually uses language information"""
    model.eval()
    
    differences = []
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            language = batch["language"].to(device)
            
            # Normal prediction
            pred_normal = model(context, language, None)
            
            # Prediction with zeroed language
            pred_zero_lang = model(context, torch.zeros_like(language), None)
            
            # Prediction with random language
            pred_random_lang = model(context, torch.randn_like(language), None)
            
            diff_zero = (pred_normal - pred_zero_lang).abs().mean().item()
            diff_random = (pred_normal - pred_random_lang).abs().mean().item()
            
            differences.append({
                'zero_lang_diff': diff_zero,
                'random_lang_diff': diff_random
            })
    
    avg_zero_diff = np.mean([d['zero_lang_diff'] for d in differences])
    avg_random_diff = np.mean([d['random_lang_diff'] for d in differences])
    
    print(f"\n{'='*60}")
    print(f"Language Usage Diagnostic")
    print(f"{'='*60}")
    print(f"Avg output diff with zero language: {avg_zero_diff:.6f}")
    print(f"Avg output diff with random language: {avg_random_diff:.6f}")
    
    if avg_zero_diff < 0.01:
        print("⚠️  WARNING: Language has almost NO effect on predictions!")
    elif avg_zero_diff < 0.1:
        print("⚠️  WARNING: Language has very SMALL effect on predictions")
    else:
        print("✓ Language appears to influence predictions")
    
    return avg_zero_diff, avg_random_diff


def _init_weights(self):
    """Initialize weights with small values for stability"""
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if 'film_generator' in name:
                # Initialize FiLM to have stronger effect
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            else:
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class Trainer:
    """Trainer for world models"""
    
    def __init__(
        self,
        model_name: str,
        config: dict,
        device: torch.device,
        output_dir: Path,
        seed: int,
    ):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing and VRAM tracking
        self.start_time = time.time()
        reset_gpu_memory()
        
        # Load data
        print(f"\n{'='*60}")
        print(f"Loading dataset...")
        print(f"{'='*60}")
        self.dataloaders, self.language_encoder = create_dataloaders(
            dataset_dir=DATASET_CONFIG["dataset_dir"],
            batch_size=DATASET_CONFIG["batch_size"],
            sequence_length=DATASET_CONFIG["sequence_length"],
            prediction_horizon=DATASET_CONFIG["prediction_horizon"],
            num_workers=DATASET_CONFIG["num_workers"],
            language_model=LANGUAGE_CONFIG["model_name"],
            seed=seed,
        )
        
        # Create model
        print(f"\n{'='*60}")
        print(f"Creating model: {model_name}")
        print(f"{'='*60}")
        self.model = create_model(
            model_config=config,
            state_dim=STATE_DIM,
            language_dim=LANGUAGE_CONFIG["embedding_dim"],
            sequence_length=DATASET_CONFIG["sequence_length"],
            prediction_horizon=DATASET_CONFIG["prediction_horizon"],
        ).to(device)
        
        num_params = count_parameters(self.model)
        print(f"Model parameters: {num_params:,}")
        
        # Track model VRAM
        self.model_vram = get_gpu_memory()
        print(f"Model VRAM: {self.model_vram:.2f} MB")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=TRAIN_CONFIG["learning_rate"],
            epochs=TRAIN_CONFIG["num_epochs"],
            steps_per_epoch=len(self.dataloaders["train"]),
            pct_start=0.1,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision
        if HARDWARE_CONFIG["mixed_precision"]:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Tracking
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []
        self.epoch_times = []
        self.peak_vram = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}",
        )
        
        for batch in pbar:
            # Move to device
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            language = batch["language"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    with autocast():
                        predictions = self.model(context, language, mask)
                        loss = self.criterion(predictions, target)
                
                if torch.isnan(loss):
                    print(f"\n⚠️  Warning: NaN loss detected at batch {num_batches}")
                    print(f"   Skipping this batch...")
                    self.optimizer.zero_grad()
                    continue
                
                self.scaler.scale(loss).backward()
                
                if TRAIN_CONFIG["gradient_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    total_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        TRAIN_CONFIG["gradient_clip"]
                    )
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        print(f"\n⚠️  Warning: NaN/Inf gradients detected (norm: {total_norm})")
                        print(f"   Skipping this batch...")
                        self.optimizer.zero_grad()
                        self.scaler.update()
                        continue
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(context, language, mask)
                loss = self.criterion(predictions, target)
                
                if torch.isnan(loss):
                    print(f"\n⚠️  Warning: NaN loss detected at batch {num_batches}")
                    print(f"   Skipping this batch...")
                    continue
                
                loss.backward()
                
                if TRAIN_CONFIG["gradient_clip"] > 0:
                    total_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        TRAIN_CONFIG["gradient_clip"]
                    )
                    if torch.isnan(total_norm):
                        print(f"\n⚠️  Warning: NaN gradients detected")
                        print(f"   Skipping this batch...")
                        self.optimizer.zero_grad()
                        continue
                
                self.optimizer.step()
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Track peak VRAM
            current_vram = get_gpu_memory()
            self.peak_vram = max(self.peak_vram, current_vram)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "vram": f"{current_vram:.0f}MB"
            })
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)
        
        return {
            "loss": avg_loss,
            "time": epoch_time,
            "vram": self.peak_vram
        }
    
    @torch.no_grad()
    def evaluate(self, split: str = "test_compositional") -> dict:
        """Evaluate on a split"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.dataloaders[split]:
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            language = batch["language"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            
            predictions = self.model(context, language, mask)
            
            if torch.isnan(predictions).any():
                print(f"  ⚠️ NaN predictions in {split} evaluation, skipping batch")
                continue
            
            loss = self.criterion(predictions, target)
            
            all_predictions.append(predictions)
            all_targets.append(target)
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches == 0:
            print(f"  ⚠️ All batches had NaN predictions in {split}")
            return {"loss": float('inf'), "mse": float('inf'), "mae": float('inf'),
                    "position_mse": float('inf'), "velocity_mse": float('inf')}
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_trajectory_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / num_batches
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        
        for epoch in range(TRAIN_CONFIG["num_epochs"]):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Evaluate
            if (epoch + 1) % TRAIN_CONFIG["eval_every"] == 0:
                print(f"\nEvaluating...")
                val_metrics = self.evaluate("test_compositional")
                self.val_history.append(val_metrics)
                
                print(f"Train loss: {train_metrics['loss']:.6f} | "
                      f"Time: {train_metrics['time']:.1f}s | "
                      f"VRAM: {train_metrics['vram']:.0f}MB")
                print(f"Val loss: {val_metrics['loss']:.6f} | "
                      f"Val MSE: {val_metrics['mse']:.6f}")
                
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ Saved best model (loss: {self.best_val_loss:.6f})")
            
            if (epoch + 1) % TRAIN_CONFIG["save_every"] == 0:
                self.save_checkpoint(epoch)
        
        # Training summary
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        
        print(f"\n{'='*60}")
        print(f"Training Complete - {self.model_name}")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Avg epoch time: {avg_epoch_time:.1f}s")
        print(f"Peak VRAM: {self.peak_vram:.0f} MB")
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}")
        
        stats = {
            "total_time_minutes": total_time / 60,
            "avg_epoch_time_seconds": float(avg_epoch_time),
            "peak_vram_mb": float(self.peak_vram),
            "model_vram_mb": float(self.model_vram),
            "best_val_loss": float(self.best_val_loss),
            "num_epochs": TRAIN_CONFIG["num_epochs"],
        }
        
        with open(self.output_dir / f"{self.model_name}_training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Final evaluation on all splits")
        print(f"{'='*60}")
        self.evaluate_all_splits()
    
    def evaluate_all_splits(self):
        """Comprehensive evaluation on all splits"""
        results = {}
        
        for split in EVAL_CONFIG["test_splits"]:
            if split not in self.dataloaders:
                continue
            
            print(f"\nEvaluating {split}...")
            metrics = self.evaluate(split)
            results[split] = metrics
            
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Position MSE: {metrics['position_mse']:.6f}")
            print(f"  Velocity MSE: {metrics['velocity_mse']:.6f}")
        
        if EVAL_CONFIG["run_diagnostics"]:
            print(f"\n{'='*60}")
            print(f"Diagnostic Analysis")
            print(f"{'='*60}")
            
            split_comp = compute_split_comparison(results)
            print(f"\nGeneralization gaps:")
            print(f"  Compositional gap: {split_comp['compositional_gap']:.6f}")
            print(f"  Contradiction gap: {split_comp['contradiction_gap']:.6f}")
            
            ablation = compute_ablation_analysis(results)
            print(f"\nAblation sensitivity:")
            print(f"  Shuffled degradation: {ablation['shuffled_degradation']:.6f}")
            print(f"  Wrong rule degradation: {ablation['wrong_rule_degradation']:.6f}")
            
            results["diagnostics"] = {
                "split_comparison": split_comp,
                "ablation": ablation,
            }
        
        results_file = self.output_dir / f"{self.model_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": self.config,
        }
        
        if is_best:
            path = self.output_dir / f"{self.model_name}_best.pt"
        else:
            path = self.output_dir / f"{self.model_name}_epoch{epoch+1}.pt"
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mamba2",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1011],
        help="Random seeds for multiple runs",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Specific run ID (uses seeds[run_id])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and results",
    )

    args = parser.parse_args()
    
    # Setup
    device = torch.device(HARDWARE_CONFIG["device"])
    output_dir = Path(args.output_dir) / args.model
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DATASET_CONFIG["seed"])
    
    if HARDWARE_CONFIG["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create trainer
    trainer = Trainer(
        model_name=args.model,
        config=MODEL_CONFIGS[args.model],
        device=device,
        output_dir=output_dir,
        seed=args.seed,
    )
    
    # Train
    trainer.train()
    
    diagnose_language_usage(trainer.model, trainer.dataloaders["train"], trainer.device)


if __name__ == "__main__":
    main()
