"""
Comprehensive evaluation for paper submission
Generates all required tables and figures
FIXED: Handles the actual directory structure: outputs/{model}/seed_{seed}/{model}/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
import torch
from tqdm import tqdm

# Your imports
from configs.train_config import MODEL_CONFIGS, DATASET_CONFIG, LANGUAGE_CONFIG, STATE_DIM
from utils.data_loader import create_dataloaders
from utils.statistical_metrics import (
    compute_bootstrap_ci, compute_effect_size, 
    compute_prediction_intervals, format_result_with_ci
)
from models import create_model


class ComprehensiveEvaluator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.results = {}
        self.seeds = [42, 123, 456, 789, 1011, 1213]
        self.models = list(MODEL_CONFIGS.keys())
    
    def load_all_results(self):
        """Load results from all seeds and models"""
        print(f"Looking for results in: {self.output_dir}")
        
        for model in self.models:
            self.results[model] = defaultdict(list)
            found_count = 0
            
            for seed in self.seeds:
                # FIXED: Handle the actual directory structure with extra model folder
                # Try multiple possible paths
                possible_paths = [
                    # Actual structure: outputs/{model}/seed_{seed}/{model}/{model}_results.json
                    self.output_dir / model / f"seed_{seed}" / model / f"{model}_results.json",
                    # Expected structure: outputs/{model}/seed_{seed}/{model}_results.json
                    self.output_dir / model / f"seed_{seed}" / f"{model}_results.json",
                    # Flat structure: outputs/{model}/{model}_results.json
                    self.output_dir / model / f"{model}_results.json",
                ]
                
                results_file = None
                for path in possible_paths:
                    if path.exists():
                        results_file = path
                        break
                
                if results_file:
                    found_count += 1
                    with open(results_file) as f:
                        data = json.load(f)
                    
                    for split, metrics in data.items():
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    self.results[model][f"{split}/{metric}"].append(value)
            
            if found_count > 0:
                print(f"  ✓ {model}: Found {found_count} seed results")
            else:
                print(f"  ✗ {model}: No results found")
    
    def generate_main_results_table(self) -> pd.DataFrame:
        """
        Generate Table 1: Main results
        Format: Model | Train | Compositional | Contradiction | Δ_comp | Δ_contra
        """
        rows = []
        
        for model in self.models:
            if model not in self.results:
                continue
            
            train_mse = self.results[model].get("train/mse", [])
            comp_mse = self.results[model].get("test_compositional/mse", [])
            contra_mse = self.results[model].get("test_contradiction/mse", [])
            
            if not train_mse:
                continue
            
            n = len(train_mse)
            
            # Calculate deltas safely
            train_mean = np.mean(train_mse)
            comp_mean = np.mean(comp_mse) if comp_mse else train_mean
            contra_mean = np.mean(contra_mse) if contra_mse else train_mean
            
            delta_comp = (comp_mean - train_mean) / train_mean * 100 if train_mean > 0 else 0
            delta_contra = (contra_mean - train_mean) / train_mean * 100 if train_mean > 0 else 0
            
            row = {
                "Model": model,
                "Train MSE": format_result_with_ci(np.mean(train_mse), np.std(train_mse, ddof=1) if n > 1 else 0, n),
                "Compositional MSE": format_result_with_ci(np.mean(comp_mse), np.std(comp_mse, ddof=1) if n > 1 and comp_mse else 0, n) if comp_mse else "N/A",
                "Contradiction MSE": format_result_with_ci(np.mean(contra_mse), np.std(contra_mse, ddof=1) if n > 1 and contra_mse else 0, n) if contra_mse else "N/A",
                "Δ_comp (%)": f"{delta_comp:.1f}",
                "Δ_contra (%)": f"{delta_contra:.1f}",
                "n_runs": n,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            # Save as LaTeX
            latex = df.to_latex(index=False, escape=False)
            with open(self.output_dir / "table1_main_results.tex", "w") as f:
                f.write(latex)
        
        return df
    
    def generate_ablation_table(self) -> pd.DataFrame:
        """
        Generate Table 2: Language ablation results
        Format: Model | Normal | Shuffled | Wrong Rule | Language Sensitivity
        """
        rows = []
        
        for model in self.models:
            if model not in self.results or "mlp" in model:
                continue
            
            train = self.results[model].get("train/mse", [])
            shuffled = self.results[model].get("control_shuffled/mse", [])
            wrong = self.results[model].get("control_wrong_rule/mse", [])
            
            if not train:
                continue
            
            n = len(train)
            
            # Language sensitivity = (shuffled - train) / train
            if shuffled and len(shuffled) == len(train):
                sensitivity = [(s - t) / t if t > 0 else 0 for s, t in zip(shuffled, train)]
                sensitivity_str = f"{np.mean(sensitivity)*100:.1f}% ± {np.std(sensitivity, ddof=1)*100:.1f}%" if len(sensitivity) > 1 else f"{np.mean(sensitivity)*100:.1f}%"
            else:
                sensitivity_str = "N/A"
            
            row = {
                "Model": model,
                "Normal": format_result_with_ci(np.mean(train), np.std(train, ddof=1) if n > 1 else 0, n),
                "Shuffled Lang.": format_result_with_ci(np.mean(shuffled), np.std(shuffled, ddof=1) if len(shuffled) > 1 else 0, len(shuffled)) if shuffled else "N/A",
                "Wrong Rule": format_result_with_ci(np.mean(wrong), np.std(wrong, ddof=1) if len(wrong) > 1 else 0, len(wrong)) if wrong else "N/A",
                "Lang. Sensitivity": sensitivity_str,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_significance_table(self) -> pd.DataFrame:
        """
        Generate Table 3: Statistical significance (pairwise comparisons)
        """
        baseline = "mlp_baseline"
        metric_key = "test_compositional/mse"
        
        if baseline not in self.results:
            print(f"  Warning: Baseline {baseline} not found")
            return pd.DataFrame()
        
        baseline_values = np.array(self.results[baseline].get(metric_key, []))
        
        if len(baseline_values) < 2:
            print(f"  Warning: Not enough baseline results for significance testing")
            return pd.DataFrame()
        
        rows = []
        for model in self.models:
            if model == baseline or model not in self.results:
                continue
            
            model_values = np.array(self.results[model].get(metric_key, []))
            
            if len(model_values) < 2:
                continue
            
            # Ensure same number of samples for paired test
            min_len = min(len(baseline_values), len(model_values))
            baseline_subset = baseline_values[:min_len]
            model_subset = model_values[:min_len]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_subset, model_subset)
            
            # Effect size
            effect = compute_effect_size(baseline_subset, model_subset, paired=True)
            
            # Improvement
            improvement = (np.mean(baseline_subset) - np.mean(model_subset)) / np.mean(baseline_subset) * 100
            
            rows.append({
                "Model": model,
                "vs Baseline Δ": f"{improvement:+.1f}%",
                "Cohen's d": f"{effect['cohens_d']:.2f} ({effect['interpretation']})",
                "p-value": f"{p_value:.4f}" if p_value >= 0.001 else "<0.001",
                "Significant": "✓" if p_value < 0.05 else "✗",
            })
        
        return pd.DataFrame(rows)
    
    def generate_computational_table(self) -> pd.DataFrame:
        """
        Generate Table 4: Computational costs
        """
        rows = []
        
        for model in self.models:
            # FIXED: Handle the actual directory structure
            possible_paths = [
                self.output_dir / model / f"seed_42" / model / f"{model}_training_stats.json",
                self.output_dir / model / f"seed_42" / f"{model}_training_stats.json",
                self.output_dir / model / f"{model}_training_stats.json",
            ]
            
            stats_file = None
            for path in possible_paths:
                if path.exists():
                    stats_file = path
                    break
            
            if not stats_file:
                continue
            
            with open(stats_file) as f:
                stats_data = json.load(f)
            
            # Load model to count params
            try:
                model_instance = create_model(
                    MODEL_CONFIGS[model],
                    state_dim=STATE_DIM,
                    language_dim=LANGUAGE_CONFIG["embedding_dim"],
                    sequence_length=DATASET_CONFIG["sequence_length"],
                    prediction_horizon=DATASET_CONFIG["prediction_horizon"],
                )
                n_params = sum(p.numel() for p in model_instance.parameters())
            except Exception as e:
                print(f"  Warning: Could not create {model}: {e}")
                n_params = "N/A"
            
            rows.append({
                "Model": model,
                "Parameters": f"{n_params:,}" if isinstance(n_params, int) else n_params,
                "Train Time (min)": f"{stats_data.get('total_time_minutes', 0):.1f}",
                "Epoch Time (s)": f"{stats_data.get('avg_epoch_time_seconds', 0):.1f}",
                "Peak VRAM (MB)": f"{stats_data.get('peak_vram_mb', 0):.0f}",
            })
        
        return pd.DataFrame(rows)
    
    def plot_learning_curves(self):
        """
        Figure 1: Learning curves with confidence bands
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for idx, model in enumerate(self.models):
            # Load training histories from all seeds
            histories = []
            for seed in self.seeds:
                # FIXED: Handle the actual directory structure
                possible_paths = [
                    self.output_dir / model / f"seed_{seed}" / model / f"{model}_best.pt",
                    self.output_dir / model / f"seed_{seed}" / f"{model}_best.pt",
                ]
                
                checkpoint = None
                for path in possible_paths:
                    if path.exists():
                        checkpoint = path
                        break
                
                if checkpoint:
                    try:
                        data = torch.load(checkpoint, map_location="cpu")
                        if "val_history" in data and data["val_history"]:
                            histories.append([h["loss"] for h in data["val_history"]])
                    except Exception as e:
                        print(f"  Warning: Could not load {checkpoint}: {e}")
            
            if not histories:
                continue
            
            # Align lengths
            min_len = min(len(h) for h in histories)
            if min_len == 0:
                continue
            histories = [h[:min_len] for h in histories]
            histories = np.array(histories)
            
            epochs = np.arange(1, min_len + 1)
            mean = histories.mean(axis=0)
            std = histories.std(axis=0)
            
            # Plot
            axes[0].plot(epochs, mean, label=model, color=colors[idx])
            axes[0].fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colors[idx])
        
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation Loss")
        axes[0].set_title("Learning Curves (mean ± std)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_yscale("log")
        
        # Plot 2: Final performance comparison
        models_with_data = []
        means = []
        stds = []
        
        for model in self.models:
            values = self.results.get(model, {}).get("test_compositional/mse", [])
            if values:
                models_with_data.append(model)
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        
        if models_with_data:
            x = np.arange(len(models_with_data))
            axes[1].bar(x, means, yerr=stds, capsize=5, color=colors[:len(models_with_data)])
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(models_with_data, rotation=45, ha="right")
            axes[1].set_ylabel("Compositional Test MSE")
            axes[1].set_title("Final Performance (mean ± std)")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure1_learning_curves.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(self.output_dir / "figure1_learning_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_generalization_heatmap(self):
        """
        Figure 2: Generalization gap heatmap
        """
        data = []
        
        for model in self.models:
            if model not in self.results:
                continue
            
            train_vals = self.results[model].get("train/mse", [])
            if not train_vals:
                continue
            
            train = np.mean(train_vals)
            comp = np.mean(self.results[model].get("test_compositional/mse", [train]))
            contra = np.mean(self.results[model].get("test_contradiction/mse", [train]))
            shuffled = np.mean(self.results[model].get("control_shuffled/mse", [train]))
            wrong = np.mean(self.results[model].get("control_wrong_rule/mse", [train]))
            
            if train > 0:
                data.append({
                    "Model": model,
                    "Train": train,
                    "Compositional": (comp - train) / train * 100,
                    "Contradiction": (contra - train) / train * 100,
                    "Shuffled": (shuffled - train) / train * 100 if "mlp" not in model else 0,
                    "Wrong Rule": (wrong - train) / train * 100 if "mlp" not in model else 0,
                })
        
        if not data:
            print("  Warning: No data for heatmap")
            return
        
        df = pd.DataFrame(data).set_index("Model")
        
        # Drop Train column for heatmap (it would be all zeros in gap calculation)
        df_gaps = df.drop(columns=["Train"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_gaps, 
            annot=True, 
            fmt=".1f",
            cmap="RdYlGn_r",  # Red = bad (higher gap), Green = good
            center=0,
            cbar_kws={"label": "Performance Gap (%)"}
        )
        plt.title("Generalization Gap (% increase from train)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure2_generalization_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_per_timestep_error(self, device="cuda"):
        """
        Figure 3: Per-timestep prediction error
        """
        # Load data
        dataloaders, _ = create_dataloaders(
            dataset_dir=DATASET_CONFIG["dataset_dir"],
            batch_size=32,
            sequence_length=DATASET_CONFIG["sequence_length"],
            prediction_horizon=DATASET_CONFIG["prediction_horizon"],
            num_workers=0,
            language_model=LANGUAGE_CONFIG["model_name"],
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for idx, model_name in enumerate(self.models):
            # FIXED: Handle the actual directory structure
            possible_paths = [
                self.output_dir / model_name / "seed_42" / model_name / f"{model_name}_best.pt",
                self.output_dir / model_name / "seed_42" / f"{model_name}_best.pt",
            ]
            
            checkpoint = None
            for path in possible_paths:
                if path.exists():
                    checkpoint = path
                    break
            
            if not checkpoint:
                continue
            
            # Load model
            try:
                model = create_model(
                    MODEL_CONFIGS[model_name],
                    state_dim=STATE_DIM,
                    language_dim=LANGUAGE_CONFIG["embedding_dim"],
                    sequence_length=DATASET_CONFIG["sequence_length"],
                    prediction_horizon=DATASET_CONFIG["prediction_horizon"],
                ).to(device)
                
                data = torch.load(checkpoint, map_location=device)
                model.load_state_dict(data["model_state_dict"])
                model.eval()
            except Exception as e:
                print(f"  Warning: Could not load {model_name}: {e}")
                continue
            
            # Compute per-timestep error
            all_errors = []
            with torch.no_grad():
                for batch in dataloaders["test_compositional"]:
                    context = batch["context"].to(device)
                    target = batch["target"].to(device)
                    language = batch["language"].to(device)
                    
                    pred = model(context, language, None)
                    errors = (pred - target).pow(2).mean(dim=-1)  # [batch, horizon]
                    all_errors.append(errors.cpu().numpy())
            
            if not all_errors:
                continue
            
            all_errors = np.concatenate(all_errors, axis=0)
            mean_errors = all_errors.mean(axis=0)
            std_errors = all_errors.std(axis=0)
            
            timesteps = np.arange(1, len(mean_errors) + 1)
            ax.plot(timesteps, mean_errors, label=model_name, color=colors[idx])
            ax.fill_between(timesteps, mean_errors - std_errors, mean_errors + std_errors,
                           alpha=0.2, color=colors[idx])
        
        ax.set_xlabel("Prediction Timestep")
        ax.set_ylabel("MSE")
        ax.set_title("Prediction Error vs. Horizon")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_yscale("log")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure3_timestep_error.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    
    def generate_all(self):
        """Generate all tables and figures"""
        print("Loading results...")
        self.load_all_results()
        
        print("\nGenerating tables...")
        
        table1 = self.generate_main_results_table()
        print("\nTable 1: Main Results")
        if not table1.empty:
            print(table1.to_string(index=False))
        else:
            print("  No data available")
        
        table2 = self.generate_ablation_table()
        print("\nTable 2: Language Ablation")
        if not table2.empty:
            print(table2.to_string(index=False))
        else:
            print("  No data available")
        
        table3 = self.generate_significance_table()
        print("\nTable 3: Statistical Significance")
        if not table3.empty:
            print(table3.to_string(index=False))
        else:
            print("  No data available")
        
        table4 = self.generate_computational_table()
        print("\nTable 4: Computational Costs")
        if not table4.empty:
            print(table4.to_string(index=False))
        else:
            print("  No data available")
        
        print("\nGenerating figures...")
        self.plot_learning_curves()
        print("  ✓ Figure 1: Learning curves")
        
        self.plot_generalization_heatmap()
        print("  ✓ Figure 2: Generalization heatmap")
        
        try:
            self.plot_per_timestep_error()
            print("  ✓ Figure 3: Per-timestep error")
        except Exception as e:
            print(f"  ✗ Figure 3 failed: {e}")
        
        # Save all tables
        if not table1.empty:
            table1.to_csv(self.output_dir / "table1_main_results.csv", index=False)
        if not table2.empty:
            table2.to_csv(self.output_dir / "table2_ablation.csv", index=False)
        if not table3.empty:
            table3.to_csv(self.output_dir / "table3_significance.csv", index=False)
        if not table4.empty:
            table4.to_csv(self.output_dir / "table4_computational.csv", index=False)
        
        print(f"\n✓ All outputs saved to {self.output_dir}")


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    evaluator.generate_all()
