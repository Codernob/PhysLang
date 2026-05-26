"""
Model factory for creating different world model architectures
"""

import torch.nn as nn
from typing import Dict, Any


from .mlp_baseline import MLPBaseline
from .transformer_model import TransformerWorldModel
from .mamba_model import MambaWorldModel
from .mamba2_model import MambaWorldModel2
from .gru_model import GRUWorldModel
from .lstm_model import LSTMWorldModel


__all__ = ['create_model', 'count_parameters', 'MLPBaseline', 
           'TransformerWorldModel', 'MambaWorldModel', 'MambaWorldModel2', 
           'GRUWorldModel', 'LSTMWorldModel']


def create_model(
    model_config: Dict[str, Any],
    state_dim: int,
    language_dim: int,
    sequence_length: int,
    prediction_horizon: int,
) -> nn.Module:
    """Factory function to create world models"""
    model_type = model_config["type"]
    
    if model_type == "mlp":
        model = MLPBaseline(
            state_dim=state_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            hidden_dims=model_config["hidden_dims"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
        )
    
    elif model_type == "transformer":
        model = TransformerWorldModel(
            state_dim=state_dim,
            language_dim=language_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            language_fusion=model_config["language_fusion"],
        )
    
    elif model_type == "mamba":
        model = MambaWorldModel(
            state_dim=state_dim,
            language_dim=language_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            d_model=model_config["d_model"],
            d_state=model_config["d_state"],
            d_conv=model_config["d_conv"],
            expand=model_config["expand"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )

    elif model_type == "mamba2":
        model = MambaWorldModel2(
            state_dim=state_dim,
            language_dim=language_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            d_model=model_config["d_model"],
            d_state=model_config["d_state"],
            d_conv=model_config["d_conv"],
            expand=model_config["expand"],
            headdim=model_config.get("headdim", 64),
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
    
    elif model_type == "gru":
        model = GRUWorldModel(
            state_dim=state_dim,
            language_dim=language_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            language_fusion=model_config["language_fusion"],
        )
    
    elif model_type == "lstm":
        model = LSTMWorldModel(
            state_dim=state_dim,
            language_dim=language_dim,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            bidirectional=model_config.get("bidirectional", True),
            language_fusion=model_config.get("language_fusion", "concat"),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from configs.train_config import MODEL_CONFIGS
    
    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}:")
        try:
            model = create_model(
                model_config=config,
                state_dim=36,
                language_dim=384,
                sequence_length=20,
                prediction_horizon=10,
            )
            print(f"  ✓ Created successfully")
            print(f"  Parameters: {count_parameters(model):,}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
