"""
Occlusion-aware scene encoder.

Changes vs original:
  - OcclusionRegionDetector: added padding/clipping so the code never crashes
    when the number of occlusion regions in occ_vecs (Nc) differs from
    num_occ_regions (e.g. fewer observed occlusions than the query count).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionRegionDetector(nn.Module):
    """
    Detects and encodes occluded regions from a BEV feature map.

    Args:
        embed_dim:            Feature dimension.
        num_occ_regions:      Number of occlusion query slots (fixed).
        num_points_per_region: Points per region in occ_vecs (default 8).
    """

    def __init__(
        self,
        embed_dim:             int = 256,
        num_occ_regions:       int = 20,
        num_points_per_region: int = 8,
    ):
        super().__init__()
        self.num_occ_regions = num_occ_regions
        self.num_points      = num_points_per_region
        self.embed_dim       = embed_dim

        self.occ_query  = nn.Parameter(
            torch.randn(num_occ_regions, embed_dim) * 0.02
        )
        self.pos_enc    = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

    def forward(
        self,
        bev_features: torch.Tensor,
        occ_vecs:     torch.Tensor = None,
    ) -> torch.Tensor:
        """
        bev_features: (B, N_tokens, embed_dim)
        occ_vecs:     (B, Nc, num_points, 6) or None

        Returns: (B, num_occ_regions, embed_dim)
        """
        B = bev_features.shape[0]
        occ_q = self.occ_query.unsqueeze(0).expand(B, -1, -1).clone()  # (B, num_occ_regions, D)

        if occ_vecs is not None:
            _, Nc, Np, _ = occ_vecs.shape

            # Encode geometry
            geo_feat = self.pos_enc(
                occ_vecs.reshape(B * Nc * Np, 6)
            ).reshape(B, Nc, Np, self.embed_dim).mean(dim=2)   # (B, Nc, D)

            # FIX: pad or trim geo_feat to match num_occ_regions query slots
            if Nc < self.num_occ_regions:
                pad = torch.zeros(
                    B, self.num_occ_regions - Nc, self.embed_dim,
                    device=occ_vecs.device, dtype=occ_vecs.dtype
                )
                geo_feat = torch.cat([geo_feat, pad], dim=1)   # (B, num_occ_regions, D)
            elif Nc > self.num_occ_regions:
                geo_feat = geo_feat[:, :self.num_occ_regions]  # trim excess

            occ_q = occ_q + geo_feat   # (B, num_occ_regions, D)

        occ_feat, _ = self.cross_attn(occ_q, bev_features, bev_features)
        return occ_feat   # (B, num_occ_regions, embed_dim)


class OcclusionAwareSceneEncoder(nn.Module):
    """
    Fuses agent detections, map features, occlusion regions and ego state
    into a unified instance-centric scene representation.

    Returns: (B, num_agents+1, embed_dim)  — ego + N agent slots
    """

    def __init__(self, embed_dim: int = 256, num_agents: int = 20):
        super().__init__()
        self.embed_dim = embed_dim

        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.map_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.occ_module = OcclusionRegionDetector(embed_dim)

        self.ego_proj = nn.Linear(8, embed_dim)
        self.nav_proj = nn.Linear(3, embed_dim)

        self.fusion_self_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.fusion_norm      = nn.LayerNorm(embed_dim)

        self.map_cross_attn   = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.map_norm         = nn.LayerNorm(embed_dim)

        self.occ_cross_attn   = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.occ_norm         = nn.LayerNorm(embed_dim)

    def forward(
        self,
        bev_features: torch.Tensor,
        agent_feats:  torch.Tensor,
        map_feats:    torch.Tensor,
        ego_state:    torch.Tensor,
        nav_cmd:      torch.Tensor,
        occ_vecs:     torch.Tensor = None,
    ) -> torch.Tensor:
        A = self.detection_head(agent_feats)          # (B, Na, D)
        M = self.map_head(map_feats)                  # (B, Nm, D)
        C = self.occ_module(bev_features, occ_vecs)   # (B, num_occ_regions, D)

        ego_feat = (self.ego_proj(ego_state) + self.nav_proj(nav_cmd)).unsqueeze(1)

        # Instance-centric: ego + agents
        I0 = torch.cat([ego_feat, A], dim=1)          # (B, Na+1, D)

        I1, _ = self.fusion_self_attn(I0, I0, I0)
        I1     = self.fusion_norm(I0 + I1)

        I2, _ = self.map_cross_attn(I1, M, M)
        I2     = self.map_norm(I1 + I2)

        I, _  = self.occ_cross_attn(I2, C, C)
        I      = self.occ_norm(I2 + I)

        return I   # (B, Na+1, D)
