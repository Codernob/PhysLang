"""
Mamba2-based world model with language conditioning
FIXED: Ensures language actually affects predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for mamba-ssm")

from mamba_ssm import Mamba2 as MambaBlock
print("  ✓ Using official Mamba-SSM implementation (GPU-only)")


class MambaWorldModel2(nn.Module):
    """
    Mamba2/SSM-based world model with WORKING language conditioning
    """
    
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        sequence_length: int,
        prediction_horizon: int,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.d_model = d_model
        self.num_layers = num_layers
        self.headdim = headdim
        
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
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=64,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.lang_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.lang_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])
        
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        context: torch.Tensor,
        language: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with GUARANTEED language influence
        """
        if not context.is_cuda or not language.is_cuda:
            raise RuntimeError("Inputs must be on CUDA")
        
        batch_size = context.shape[0]
        
        context = torch.clamp(context, -50, 50)
        language = torch.clamp(language, -50, 50)
        
        x = self.state_encoder(context)
        
        lang_emb = self.language_encoder(language)
        lang_emb_seq = lang_emb.unsqueeze(1)
        
        for i, (block, norm, cross_attn, gate) in enumerate(
            zip(self.blocks, self.layer_norms, self.lang_cross_attn, self.lang_gates)
        ):
            residual = x
            x = block(x)
            
            lang_attended, _ = cross_attn(
                query=x,
                key=lang_emb_seq.expand(-1, x.shape[1], -1),
                value=lang_emb_seq.expand(-1, x.shape[1], -1),
            )
            
            gate_input = torch.cat([x, lang_attended], dim=-1)
            g = gate(gate_input)
            
            x = (1 - g) * x + g * lang_attended
            
            x = norm(x + residual)
        
        x = self.output_norm(x)
        
        last_hidden = x[:, -1, :]
        
        combined = torch.cat([last_hidden, lang_emb], dim=-1)
        
        predictions = self.predictor(combined)
        predictions = predictions.reshape(
            batch_size, self.prediction_horizon, self.state_dim
        )
        
        predictions = torch.clamp(predictions, -50, 50)
        
        return predictions


if __name__ == "__main__":
    device = torch.device("cuda")
    
    model = MambaWorldModel2(
        state_dim=54,
        language_dim=384,
        sequence_length=20,
        prediction_horizon=10,
        d_model=256,
        d_state=64,
        headdim=64,
        num_layers=4,
    ).to(device)
    
    batch_size = 8
    context = torch.randn(batch_size, 20, 54).to(device)
    language = torch.randn(batch_size, 384).to(device)
    
    output = model(context, language)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        out1 = model(context, language)
        out2 = model(context, torch.zeros_like(language))
        out3 = model(context, torch.randn_like(language))
        
        diff_zero = (out1 - out2).abs().mean().item()
        diff_rand = (out1 - out3).abs().mean().item()
        
        print(f"\nLanguage influence test:")
        print(f"  Diff with zero language: {diff_zero:.6f}")
        print(f"  Diff with random language: {diff_rand:.6f}")
        
        if diff_zero > 0.01:
            print("  ✓ Language IS affecting predictions")
        else:
            print("  ✗ Language is NOT affecting predictions!")
