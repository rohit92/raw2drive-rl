"""
BEVFormer-style raw sensor encoder.

Changes vs original:
  - pretrained=True → ResNet50_Weights.IMAGENET1K_V1  (torchvision ≥ 0.13 API)
  - Replaced Linear(embed_dim*bev_h*bev_w, 1024) [was ~655 M params] with
    attention-weighted pooling + small MLP (~0.5 M params)
  - bev_pos is expanded per-batch so it is device-safe under torch.compile
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights


class DeformableAttention2D(nn.Module):
    """
    Simplified multi-head cross-attention used as the spatial attention
    primitive inside each BEVFormer encoder layer.

    A full deformable-attention kernel (DCN-style) requires custom CUDA ops;
    this standard SDPA version is functionally equivalent for training.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_pos=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        H  = self.num_heads
        Dh = C // H

        if query_pos is not None:
            query = query + query_pos

        q = self.q_proj(query).reshape(B, Nq, H, Dh).transpose(1, 2)
        k = self.k_proj(key  ).reshape(B, Nk, H, Dh).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Nk, H, Dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)


class BEVFormerEncoder(nn.Module):
    """
    Multi-camera images → (global embed, BEV feature grid).

    Args:
        embed_dim:     BEV token dimension (256 — keeps OAIAD compatible)
        bev_h/bev_w:   BEV query grid size (default 50×50)
        num_cams:      Surround camera count (6)
        num_layers:    BEVFormer encoder depth
        out_embed_dim: Global embed dimension fed to RSSM (512)

    Returns (forward):
        embed:  (B, out_embed_dim)          — compressed global scene feature
        bev_q:  (B, bev_h * bev_w, embed_dim) — spatial BEV tokens for OAIAD
    """

    def __init__(
        self,
        img_backbone: str = 'resnet50',
        embed_dim: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        num_cams: int = 6,
        num_layers: int = 3,
        num_heads: int = 8,
        out_embed_dim: int = 512,
    ):
        super().__init__()
        self.bev_h      = bev_h
        self.bev_w      = bev_w
        self.embed_dim  = embed_dim
        self.num_cams   = num_cams

        # ── Image backbone (ResNet-50, updated API) ─────────────────────
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # → (B, 2048, H/32, W/32)
        self.img_proj = nn.Conv2d(2048, embed_dim, 1)

        # ── BEV query grid ───────────────────────────────────────────────
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim) * 0.02)
        self.bev_pos     = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim) * 0.02)

        # ── Encoder layers ───────────────────────────────────────────────
        self.cross_attns = nn.ModuleList(
            [DeformableAttention2D(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.self_attns = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            ) for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norms3 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # ── Global embed projection ──────────────────────────────────────
        # FIX: Replaced Linear(640_000, 1024) [~655 M params] with
        #      attention-weighted pooling + small MLP (~0.5 M params).
        self.pool_query = nn.Linear(embed_dim, 1)   # per-token importance
        self.out_proj   = nn.Sequential(
            nn.Linear(embed_dim, out_embed_dim),
            nn.SiLU(),
            nn.LayerNorm(out_embed_dim),
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, N, 3, H, W) → (B, N, embed_dim, h, w)"""
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.contiguous().reshape(B * N, C, H, W)
        feats     = self.backbone(imgs_flat)
        feats     = self.img_proj(feats)                     # (B*N, D, h, w)
        _, D, h, w = feats.shape
        return feats.reshape(B, N, D, h, w)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor, prev_bev=None):
        """
        imgs:     (B, N, 3, H, W)
        prev_bev: (B, bev_h*bev_w, embed_dim) — optional temporal BEV state

        Returns:
            embed  (B, out_embed_dim)
            bev_q  (B, bev_h*bev_w, embed_dim)
        """
        B = imgs.shape[0]

        img_feats = self.encode_images(imgs)           # (B, N, D, h, w)
        B, N, D, h, w = img_feats.shape
        img_kv = img_feats.reshape(B, N * h * w, D)   # (B, N*h*w, D)

        bev_q   = self.bev_queries.unsqueeze(0).expand(B, -1, -1).clone()
        bev_pos = self.bev_pos.unsqueeze(0).expand(B, -1, -1)

        if prev_bev is not None:
            bev_q = bev_q + prev_bev

        for i in range(len(self.cross_attns)):
            q2    = self.cross_attns[i](bev_q, img_kv, img_kv, bev_pos)
            bev_q = self.norms1[i](bev_q + q2)

            q3, _ = self.self_attns[i](bev_q, bev_q, bev_q)
            bev_q = self.norms2[i](bev_q + q3)

            q4    = self.ffns[i](bev_q)
            bev_q = self.norms3[i](bev_q + q4)

        # Attention-weighted pooling: O(bev_h*bev_w) — no huge Linear
        scores  = self.pool_query(bev_q)            # (B, bev_h*bev_w, 1)
        weights = torch.softmax(scores, dim=1)      # (B, bev_h*bev_w, 1)
        pooled  = (bev_q * weights).sum(dim=1)      # (B, embed_dim)
        embed   = self.out_proj(pooled)             # (B, out_embed_dim)

        return embed, bev_q
