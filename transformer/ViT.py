import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from SwiGLUFFN import SwiGLUFFN
from ViTBlock import ViTBlock
from PatchEmbedding import PatchEmbedding

class ViT(nn.Module):
    def __init__(
        self,
        img_size=224, 
        patch_size=16,
        in_chans = 3,
        embed_dim=728,
        depth=12,
        nheads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        num_classes=1000
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, nheads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])



if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        nheads=12,
        num_classes=1000
    )
    logits = model(x)
    print(logits)






