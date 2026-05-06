"""Privileged encoder: BEV semantic masks -> embedding."""
import torch
import torch.nn as nn


class PrivilegedEncoder(nn.Module):
    """
    Encodes BEV semantic segmentation masks.
    Input:  (B, C, H, W) binary masks, C=43
    Output: (B, embed_dim) feature vector
    """
    def __init__(self, in_channels: int = 43, embed_dim: int = 512,
                 bev_h: int = 200, bev_w: int = 200):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),   # 100x100
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),            # 50x50
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),           # 25x25
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),           # 13x13
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),           # 7x7
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, bev_masks):
        B = bev_masks.shape[0]
        feat = self.cnn(bev_masks).view(B, -1)
        return self.proj(feat)
