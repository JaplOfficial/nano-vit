import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from SwiGLUFFN import SwiGLUFFN

class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int, nheads: int
                 , mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            E_q = embed_dim,
            E_k = embed_dim,
            E_v = embed_dim,
            E_total = embed_dim,
            nheads = nheads,
            dropout = dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SwiGLUFFN(embed_dim, embed_dim*mlp_ratio, 2)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

