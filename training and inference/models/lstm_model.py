"""
LSTM-based world model with language conditioning
Classic baseline using bidirectional LSTM
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        language_fusion: str = "concat",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.language_fusion = language_fusion
        
        self.effective_hidden = hidden_dim * (2 if bidirectional else 1)
        
        if language_fusion == "concat":
            self.input_proj = nn.Linear(state_dim + language_dim, hidden_dim)
        else:
            self.input_proj = nn.Linear(state_dim, hidden_dim)
            self.language_encoder = nn.Sequential(
                nn.Linear(language_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.effective_hidden, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.effective_hidden + language_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prediction_horizon * state_dim),
        )
    
    def forward(
        self,
        context: torch.Tensor,
        language: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        
        if self.language_fusion == "concat":
            lang_expanded = language.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([context, lang_expanded], dim=-1)
            x = self.input_proj(x)
        else:
            x = self.input_proj(context)
            lang_emb = self.language_encoder(language).unsqueeze(1)
            x = x + lang_emb
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = (attn_weights * lstm_out).sum(dim=1)
        
        combined = torch.cat([context_vector, language], dim=-1)
        
        predictions = self.decoder(combined)
        predictions = predictions.reshape(batch_size, self.prediction_horizon, self.state_dim)
        
        return predictions
