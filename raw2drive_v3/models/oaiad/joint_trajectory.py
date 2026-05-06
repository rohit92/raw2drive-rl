import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace


class TrajectoryDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.mode_scene_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mode_scene_norm = nn.LayerNorm(embed_dim)
        self.mode_time_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mode_time_norm  = nn.LayerNorm(embed_dim)
        self.agent_attn      = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.agent_norm      = nn.LayerNorm(embed_dim)
        self.mode_attn       = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mode_norm       = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, Z, scene_feat, temporal_feat):
        B, K, A, D = Z.shape

        Z_flat   = Z.reshape(B * K, A, D)
        scene_exp = scene_feat.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, D)
        z2, _    = self.mode_scene_attn(Z_flat, scene_exp, scene_exp)
        Z_flat   = self.mode_scene_norm(Z_flat + z2)
        Z        = Z_flat.reshape(B, K, A, D)

        Z_flat   = Z.reshape(B * K, A, D)
        time_exp = temporal_feat.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, D)
        z3, _    = self.mode_time_attn(Z_flat, time_exp, time_exp)
        Z_flat   = self.mode_time_norm(Z_flat + z3)
        Z        = Z_flat.reshape(B, K, A, D)

        Z_flat = Z.reshape(B * K, A, D)
        z4, _  = self.agent_attn(Z_flat, Z_flat, Z_flat)
        Z_flat = self.agent_norm(Z_flat + z4)
        Z      = Z_flat.reshape(B, K, A, D)

        Z_t    = Z.permute(0, 2, 1, 3).reshape(B * A, K, D)
        z5, _  = self.mode_attn(Z_t, Z_t, Z_t)
        Z_t    = self.mode_norm(Z_t + z5)
        Z      = Z_t.reshape(B, A, K, D).permute(0, 2, 1, 3)

        Z_flat = Z.reshape(B * K * A, D)
        Z_flat = self.ffn_norm(Z_flat + self.ffn(Z_flat))
        Z      = Z_flat.reshape(B, K, A, D)

        return Z


class JointTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_modes: int = 6,
        num_agents: int = 20,
        future_steps: int = 12,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.K = num_modes
        self.A = num_agents + 1
        self.T = future_steps
        self.D = embed_dim

        self.mode_embeddings = nn.Parameter(torch.randn(num_modes, embed_dim) * 0.02)

        self.layers = nn.ModuleList([
            TrajectoryDecoderLayer(embed_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        self.traj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, future_steps * 4)
        )

        self.mode_score_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.traj_embed  = nn.Linear(2, embed_dim)
        self.refine_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.refine_norm = nn.LayerNorm(embed_dim)
        self.refine_head = nn.Linear(embed_dim, future_steps * 2)

    def forward(self, scene_repr, temporal_feat):
        B, A, D = scene_repr.shape
        assert A == self.A, f"Expected A={self.A}, got {A}"

        mode_e  = self.mode_embeddings.unsqueeze(0).unsqueeze(2).expand(B, -1, A, -1)
        scene_e = scene_repr.unsqueeze(1).expand(-1, self.K, -1, -1)
        Z       = mode_e + scene_e

        for layer in self.layers:
            Z = layer(Z, scene_repr, temporal_feat)

        Z_flat   = Z.reshape(B * self.K * A, D)
        traj_out = self.traj_head(Z_flat).reshape(B, self.K, A, self.T, 4)
        mu       = traj_out[..., :2]
        log_b    = traj_out[..., 2:]
        b        = F.softplus(log_b) + 1e-4

        Z_mean     = Z.mean(dim=2)
        scores     = self.mode_score_head(Z_mean).squeeze(-1)
        mode_probs = F.softmax(scores, dim=-1)

        proposals  = mu
        traj_pts   = proposals.reshape(B * self.K * A, self.T, 2)
        traj_q     = self.traj_embed(traj_pts)
        traj_q2, _ = self.refine_attn(traj_q, traj_q, traj_q)
        traj_q2    = self.refine_norm(traj_q + traj_q2)
        traj_pooled = traj_q2.mean(dim=1)
        offsets    = self.refine_head(traj_pooled).reshape(B, self.K, A, self.T, 2)
        refined    = proposals + offsets

        return {
            'mu':         mu,
            'b':          b,
            'mode_probs': mode_probs,
            'refined':    refined,
        }

    def nll_loss(self, predictions, gt_trajs, gt_modes=None):
        """gt_trajs: (B, A, T, 2)"""
        mu         = predictions['mu']         # (B, K, A, T, 2)
        b          = predictions['b']
        mode_probs = predictions['mode_probs'] # (B, K)

        gt_exp = gt_trajs.unsqueeze(1).expand(-1, self.K, -1, -1, -1)
        dist   = Laplace(mu, b)
        nll    = -dist.log_prob(gt_exp).sum(dim=[-1, -2, -3])  # (B, K)

        if gt_modes is None:
            ade    = (mu - gt_exp).norm(dim=-1).mean(dim=[-1, -2])
            best_k = ade.argmin(dim=-1)
        else:
            best_k = gt_modes

        # FIX: ensure best_k is long (int64) — gather requires LongTensor
        best_k  = best_k.long()
        wta_nll = nll.gather(1, best_k.unsqueeze(-1)).squeeze(-1).mean()
        mode_loss = F.cross_entropy(mode_probs, best_k)
        return {'nll': wta_nll, 'mode': mode_loss}