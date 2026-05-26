"""
Training configuration for language-grounded world models
Designed for ICML workshop submission - diagnostic benchmark focus
"""

import torch

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    "dataset_dir": "/home/uap-lab-pc/world_model/dataset/physics_dataset",
    "sequence_length": 20,  # Number of past timesteps to condition on
    "prediction_horizon": 10,  # Number of future timesteps to predict
    "batch_size": 8,  # Adjust based on GPU memory
    "num_workers": 0,  # Set to 0 to avoid CUDA multiprocessing issues
    "seed": 42,
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# State dimension per block: [pos(3), vel(3), mass(1), friction(1), scale(1)] = 9
STATE_DIM_PER_BLOCK = 9
MAX_BLOCKS = 6  # Maximum number of blocks in any episode
STATE_DIM = STATE_DIM_PER_BLOCK * MAX_BLOCKS  # Total state dimension

# Language embedding configuration
LANGUAGE_CONFIG = {
    "embedding_type": "sentence_transformer",  # or "simple_encoder"
    "model_name": "all-MiniLM-L6-v2",  # Lightweight sentence transformer
    "embedding_dim": 384,  # Output dimension
    "freeze_encoder": True,  # Freeze pretrained weights
}

MODEL_CONFIGS = {
    # Baseline: No language
    "mlp_baseline": {
        "type": "mlp",
        "hidden_dims": [256, 512, 256],
        "activation": "relu",
        "dropout": 0.2,
        "use_language": False,
    },
    
    # Transformer - SIMPLIFIED
    "transformer": {
        "type": "transformer",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "use_language": True,
        "language_fusion": "concat",
    },
    
    # Mamba - SIMPLIFIED
    "mamba": {
        "type": "mamba",
        "d_model": 128,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        "dropout": 0.2,
        "use_language": True,
    },

    "mamba2": {
        "type": "mamba2",
        "d_model": 128,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "use_language": True,
    },
    
    # GRU
    "gru": {
        "type": "gru",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "use_language": True,
        "language_fusion": "concat",
    },

    "lstm": {
        "type": "lstm",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "language_fusion": "concat",
        "use_language": True,
    },
}

# TRAINING CONFIG
TRAIN_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "warmup_epochs": 0,
    
    "loss_type": "mse",
    "loss_weights": {
        "position": 1.0,
        "velocity": 0.5,
        "properties": 0.1,
    },
    
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    
    "log_every": 10,
    "eval_every": 1,
    "save_every": 10,
    "output_dir": "/home/uap-lab-pc/world_model/training_code/outputs",
    "checkpoint_dir": "/home/uap-lab-pc/world_model/training_code/checkpoints",
    "use_wandb": False,
    "wandb_project": "language-grounded-world-models",
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    "test_splits": [
        "train",
        "test_compositional",
        "test_contradiction",
        "control_shuffled",
        "control_wrong_rule",
    ],
    
    "metrics": [
        "mse",
        "mae",
        "rule_timing_error",
        "intervention_accuracy",
    ],
    
    "run_diagnostics": True,
    "diagnostic_tests": [
        "language_necessity",
        "rule_grounding",
        "timing_sensitivity",
    ],
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
HARDWARE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": True,
    "deterministic": True,
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_NAME = "diagnostic_benchmark_v1"
EXPERIMENT_TAGS = [
    "language-grounding",
    "world-models",
    "diagnostic",
    "icml-workshop",
]
