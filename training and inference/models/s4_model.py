"""
S4-based world model with language conditioning
Uses the official S4 implementation from state-spaces/s4
"""

import torch
import torch.nn as nn
from typing import Optional

# Install: pip install state-spaces
try:
    from s4 import S4Block
except ImportError:
    # Alternative: use the standalone S4 layer
    from models.s4_standalone import S4Block  # You may need to copy this


class S4WorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        d_model: int = 256,
        d_state: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.d_model = d_model
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.blocks = nn.ModuleList([
            S4Block(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                transposed=False,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.lang_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, prediction_horizon * state_dim),
        )
        
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        context: torch.Tensor,
        language: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        
        x = self.state_encoder(context)
        lang_emb = self.language_encoder(language)
        lang_emb_seq = lang_emb.unsqueeze(1)
        
        for block, norm in zip(self.blocks, self.layer_norms):
            residual = x
            x, _ = block(x)  # S4 returns (output, state)
            x = norm(x + residual)
        
        x_attended, _ = self.lang_cross_attn(
            query=x,
            key=lang_emb_seq.expand(-1, x.shape[1], -1),
            value=lang_emb_seq.expand(-1, x.shape[1], -1),
        )
        x = x + x_attended
        
        x = self.output_norm(x)
        last_hidden = x[:, -1, :]
        
        combined = torch.cat([last_hidden, lang_emb], dim=-1)
        predictions = self.predictor(combined)
        predictions = predictions.reshape(batch_size, self.prediction_horizon, self.state_dim)
        
        return predictions
