"""
GRU-based world model with language conditioning
Simple but effective baseline using recurrent networks
"""

import torch
import torch.nn as nn
from typing import Optional


class GRUWorldModel(nn.Module):
    """
    GRU-based world model with language conditioning
    """
    
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        language_fusion: str = "concat",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.language_fusion = language_fusion
        
        if language_fusion == "concat":
            self.input_proj = nn.Linear(state_dim + language_dim, hidden_dim)
        else:
            self.state_encoder = nn.Linear(state_dim, hidden_dim)
            self.language_encoder = nn.Linear(language_dim, hidden_dim)
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_horizon * state_dim),
        )
    
    def forward(
        self,
        context: torch.Tensor,
        language: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq_len, state_dim]
            language: [batch, lang_dim]
            mask: [batch, max_blocks] (optional)
        
        Returns:
            predictions: [batch, pred_horizon, state_dim]
        """
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        
        if self.language_fusion == "concat":
            lang_expanded = language.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([context, lang_expanded], dim=-1)
            x = self.input_proj(x)
        else:
            x = self.state_encoder(context)
            lang_emb = self.language_encoder(language).unsqueeze(1)
            x = x + lang_emb
        
        output, hidden = self.gru(x)
        
        last_hidden = output[:, -1, :]  # [batch, hidden]
        
        predictions = self.decoder(last_hidden)
        predictions = predictions.reshape(
            batch_size, self.prediction_horizon, self.state_dim
        )
        
        return predictions


if __name__ == "__main__":
    model = GRUWorldModel(
        state_dim=36,
        language_dim=384,
        sequence_length=20,
        prediction_horizon=10,
        hidden_dim=256,
        num_layers=2,
    )
    
    batch_size = 8
    context = torch.randn(batch_size, 20, 36)
    language = torch.randn(batch_size, 384)
    
    output = model(context, language)
    print(f"Input shape: {context.shape}")
    print(f"Language shape: {language.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
