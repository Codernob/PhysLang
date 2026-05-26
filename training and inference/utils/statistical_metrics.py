"""
Statistical metrics for rigorous evaluation
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import torch


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(42)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    point_estimate = np.mean(values) if statistic == "mean" else np.median(values)
    
    return point_estimate, ci_lower, ci_upper


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    paired: bool = True,
) -> Dict[str, float]:
    """
    Compute multiple effect size measures
    """
    if paired:
        diff = group1 - group2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        pooled_std = np.sqrt(
            ((len(group1)-1)*np.var(group1, ddof=1) + 
             (len(group2)-1)*np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        "cohens_d": float(cohens_d),
        "interpretation": interpretation,
    }


def compute_prediction_intervals(
    predictions: torch.Tensor,  # [batch, horizon, state_dim]
    targets: torch.Tensor,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Compute prediction intervals per timestep
    """
    errors = (predictions - targets).detach().cpu().numpy()
    
    horizon = errors.shape[1]
    intervals = {
        "mean_error": np.zeros(horizon),
        "ci_lower": np.zeros(horizon),
        "ci_upper": np.zeros(horizon),
    }
    
    alpha = 1 - confidence
    for t in range(horizon):
        timestep_errors = errors[:, t, :].flatten()
        intervals["mean_error"][t] = np.mean(np.abs(timestep_errors))
        intervals["ci_lower"][t] = np.percentile(np.abs(timestep_errors), alpha/2 * 100)
        intervals["ci_upper"][t] = np.percentile(np.abs(timestep_errors), (1-alpha/2) * 100)
    
    return intervals


def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pred_std: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute calibration metrics (if model provides uncertainty)
    """
    if pred_std is None:
        return {"calibration": "N/A - no uncertainty estimates"}
    
    errors = (predictions - targets).abs()
    z_scores = errors / (pred_std + 1e-8)
    
    pct_within_1std = (z_scores < 1).float().mean().item()
    pct_within_2std = (z_scores < 2).float().mean().item()
    
    return {
        "pct_within_1std": pct_within_1std,
        "pct_within_2std": pct_within_2std,
        "calibration_error_1std": abs(pct_within_1std - 0.6827),
        "calibration_error_2std": abs(pct_within_2std - 0.9545),
    }


def format_result_with_ci(
    mean: float,
    std: float,
    n: int,
    ci_level: float = 0.95,
) -> str:
    """
    Format result as: mean ± CI (e.g., "0.0234 ± 0.0012")
    """
    sem = std / np.sqrt(n)
    t_critical = stats.t.ppf((1 + ci_level) / 2, n - 1)
    ci_half = t_critical * sem
    
    if mean == 0:
        decimals = 4
    else:
        decimals = max(4, -int(np.floor(np.log10(abs(mean)))) + 2)
    
    return f"{mean:.{decimals}f} ± {ci_half:.{decimals}f}"
