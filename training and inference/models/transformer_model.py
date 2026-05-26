"""
Transformer-based world model with language conditioning
Uses cross-attention to integrate language rules with physics dynamics
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch, d_model]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerWorldModel(nn.Module):
    """
    Transformer-based world model with language grounding
    """
    
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        language_fusion: str = "cross_attention",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.d_model = d_model
        self.language_fusion = language_fusion
        
        # Input projection
        self.state_encoder = nn.Linear(state_dim, d_model)
        self.language_encoder = nn.Linear(language_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Language conditioning
        if language_fusion == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=False,
            )
        elif language_fusion == "film":
            self.film_gamma = nn.Linear(d_model, d_model)
            self.film_beta = nn.Linear(d_model, d_model)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, state_dim)
        
        # Learnable queries for future prediction
        self.future_queries = nn.Parameter(
            torch.randn(prediction_horizon, 1, d_model)
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
        
        x = self.state_encoder(context)  # [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        
        memory = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        
        lang_emb = self.language_encoder(language)  # [batch, d_model]
        lang_emb = lang_emb.unsqueeze(0)  # [1, batch, d_model]
        
        if self.language_fusion == "cross_attention":
            memory_lang, _ = self.cross_attention(
                query=memory,
                key=lang_emb,
                value=lang_emb,
            )
            memory = memory + memory_lang
        
        elif self.language_fusion == "film":
            gamma = self.film_gamma(lang_emb)
            beta = self.film_beta(lang_emb)
            memory = gamma * memory + beta
        
        elif self.language_fusion == "concat":
            memory = torch.cat([lang_emb, memory], dim=0)
        
        queries = self.future_queries.expand(-1, batch_size, -1)
        
        output = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )  # [pred_horizon, batch, d_model]
        
        predictions = self.output_proj(output)  # [pred_horizon, batch, state_dim]
        predictions = predictions.transpose(0, 1)  # [batch, pred_horizon, state_dim]
        
        return predictions


if __name__ == "__main__":
    model = TransformerWorldModel(
        state_dim=36,
        language_dim=384,
        sequence_length=20,
        prediction_horizon=10,
        d_model=256,
        nhead=8,
        num_layers=4,
    )
    
    batch_size = 8
    context = torch.randn(batch_size, 20, 36)
    language = torch.randn(batch_size, 384)
    
    output = model(context, language)
    print(f"Input shape: {context.shape}")
    print(f"Language shape: {language.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
