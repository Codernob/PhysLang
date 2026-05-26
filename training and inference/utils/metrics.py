"""
Evaluation metrics for diagnostic world models
Focus on language grounding, not just prediction accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_trajectory_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute basic trajectory prediction metrics
    
    Args:
        predictions: [batch, horizon, state_dim]
        targets: [batch, horizon, state_dim]
        mask: [batch, max_blocks] - mask for valid blocks
    
    Returns:
        Dictionary of metrics
    """
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
    target_flat = target_np.reshape(-1, target_np.shape[-1])
    
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    mse_per_step = np.mean((pred_np - target_np) ** 2, axis=(0, 2))
    
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mse_per_step": mse_per_step.tolist(),
    }
    
    # Position vs velocity vs properties
    state_dim_per_block = 9
    num_blocks = pred_np.shape[-1] // state_dim_per_block
    
    pos_errors = []
    vel_errors = []
    prop_errors = []
    
    for b in range(num_blocks):
        start_idx = b * state_dim_per_block
        
        pos_pred = pred_flat[:, start_idx:start_idx+3]
        pos_target = target_flat[:, start_idx:start_idx+3]
        pos_errors.append(mean_squared_error(pos_target, pos_pred))
        
        vel_pred = pred_flat[:, start_idx+3:start_idx+6]
        vel_target = target_flat[:, start_idx+3:start_idx+6]
        vel_errors.append(mean_squared_error(vel_target, vel_pred))
        
        prop_pred = pred_flat[:, start_idx+6:start_idx+9]
        prop_target = target_flat[:, start_idx+6:start_idx+9]
        prop_errors.append(mean_squared_error(prop_target, prop_pred))
    
    metrics["position_mse"] = float(np.mean(pos_errors))
    metrics["velocity_mse"] = float(np.mean(vel_errors))
    metrics["properties_mse"] = float(np.mean(prop_errors))
    
    return metrics


def compute_rule_grounding_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rule_activation_step: int = 50,
    prediction_horizon: int = 10,
) -> Dict[str, float]:
    """
    Measure if model correctly responds to rule interventions
    """
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    mid_point = prediction_horizon // 2
    
    error_first_half = np.mean((pred_np[:, :mid_point] - target_np[:, :mid_point]) ** 2)
    error_second_half = np.mean((pred_np[:, mid_point:] - target_np[:, mid_point:]) ** 2)
    
    error_ratio = error_second_half / (error_first_half + 1e-8)
    
    return {
        "error_first_half": float(error_first_half),
        "error_second_half": float(error_second_half),
        "error_ratio": float(error_ratio),
    }


def compute_language_necessity_score(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Test if language actually helps.
    Compare predictions with vs without language.
    """
    model.eval()
    
    context = batch["context"].to(device)
    target = batch["target"].to(device)
    language = batch["language"].to(device)
    mask = batch.get("mask", None)
    if mask is not None:
        mask = mask.to(device)
    
    with torch.no_grad():
        pred_with_lang = model(context, language, mask)
        error_with_lang = torch.mean((pred_with_lang - target) ** 2).item()
        
        zero_lang = torch.zeros_like(language)
        pred_without_lang = model(context, zero_lang, mask)
        error_without_lang = torch.mean((pred_without_lang - target) ** 2).item()
    
    improvement = (error_without_lang - error_with_lang) / (error_without_lang + 1e-8)
    
    return {
        "error_with_language": float(error_with_lang),
        "error_without_language": float(error_without_lang),
        "language_improvement": float(improvement),
        "language_helps": improvement > 0.0,
    }


def compute_split_comparison(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compare performance across splits to test generalization.
    """
    train_mse = results["train"]["mse"]
    comp_mse = results.get("test_compositional", {}).get("mse", train_mse)
    contra_mse = results.get("test_contradiction", {}).get("mse", train_mse)
    
    comp_gap = comp_mse - train_mse
    contra_gap = contra_mse - train_mse
    
    comp_gap_rel = comp_gap / (train_mse + 1e-8)
    contra_gap_rel = contra_gap / (train_mse + 1e-8)
    
    return {
        "compositional_gap": float(comp_gap),
        "contradiction_gap": float(contra_gap),
        "compositional_gap_relative": float(comp_gap_rel),
        "contradiction_gap_relative": float(contra_gap_rel),
    }


def compute_ablation_analysis(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Analyze ablation controls.
    """
    train_mse = results["train"]["mse"]
    shuffled_mse = results.get("control_shuffled", {}).get("mse", train_mse)
    wrong_rule_mse = results.get("control_wrong_rule", {}).get("mse", train_mse)
    
    shuffled_degradation = shuffled_mse - train_mse
    wrong_rule_degradation = wrong_rule_mse - train_mse
    
    return {
        "shuffled_degradation": float(shuffled_degradation),
        "wrong_rule_degradation": float(wrong_rule_degradation),
        "ablation_sensitivity": float(
            (shuffled_degradation + wrong_rule_degradation) / 2
        ),
    }


if __name__ == "__main__":
    batch_size = 16
    horizon = 10
    state_dim = 36
    
    predictions = torch.randn(batch_size, horizon, state_dim)
    targets = torch.randn(batch_size, horizon, state_dim)
    
    metrics = compute_trajectory_metrics(predictions, targets)
    print("Trajectory metrics:")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"  {k}: [list of {len(v)} values]")
        else:
            print(f"  {k}: {v:.6f}")
