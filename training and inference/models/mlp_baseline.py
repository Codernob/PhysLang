"""
Baseline MLP model without language conditioning
Used to test if language actually helps
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline for trajectory prediction
    No language conditioning - pure physics-based prediction
    """
    
    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        hidden_dims: List[int] = [256, 512, 512, 256],
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        input_dim = sequence_length * state_dim
        output_dim = prediction_horizon * state_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        context: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq_len, state_dim]
            language: [batch, lang_dim] (ignored in baseline)
            mask: [batch, max_blocks] (ignored in baseline)
        
        Returns:
            predictions: [batch, pred_horizon, state_dim]
        """
        batch_size = context.shape[0]
        
        x = context.reshape(batch_size, -1)
        
        output = self.mlp(x)
        
        predictions = output.reshape(
            batch_size, self.prediction_horizon, self.state_dim
        )
        
        return predictions


if __name__ == "__main__":
    model = MLPBaseline(
        state_dim=36,
        sequence_length=20,
        prediction_horizon=10,
    )
    
    batch_size = 8
    context = torch.randn(batch_size, 20, 36)
    
    output = model(context)
    print(f"Input shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
