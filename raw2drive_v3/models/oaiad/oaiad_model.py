"""
Complete OAIAD Model integrating all components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .occlusion_module import OcclusionAwareSceneEncoder
from .joint_trajectory import JointTrajectoryDecoder


class OAIADModel(nn.Module):
    """
    Occlusion-Aware Interactive End-to-End Autonomous Driving.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_agents: int = 20,
        num_modes: int = 6,
        future_steps: int = 12,
        bev_h: int = 50,
        bev_w: int = 50,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_agents = num_agents

        self.scene_encoder = OcclusionAwareSceneEncoder(embed_dim, num_agents)

        self.temporal_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True
        )

        self.traj_decoder = JointTrajectoryDecoder(
            embed_dim=embed_dim,
            num_modes=num_modes,
            num_agents=num_agents,
            future_steps=future_steps,
        )

        self.map_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 50 * 2)
        )

        self.agent_query    = nn.Parameter(torch.randn(num_agents, embed_dim) * 0.02)
        self.map_query      = nn.Parameter(torch.randn(100, embed_dim) * 0.02)
        self.det_cross_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.map_cross_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

    def forward(
        self,
        bev_features,
        ego_state,
        nav_cmd,
        bev_history=None,
        occ_vecs=None,
        gt_trajs=None,
    ):
        B = bev_features.shape[0]

        agent_q     = self.agent_query.unsqueeze(0).expand(B, -1, -1)
        agent_feats, _ = self.det_cross_attn(agent_q, bev_features, bev_features)

        map_q       = self.map_query.unsqueeze(0).expand(B, -1, -1)
        map_feats, _ = self.map_cross_attn(map_q, bev_features, bev_features)

        if bev_history is not None:
            B, T_h, HW, D = bev_history.shape
            bev_mean = bev_history.mean(dim=2)
            temporal_feat, _ = self.temporal_encoder(bev_mean)
        else:
            temporal_feat = bev_features[:, :1, :].expand(-1, 4, -1)

        scene_repr = self.scene_encoder(
            bev_features, agent_feats, map_feats,
            ego_state, nav_cmd, occ_vecs
        )

        traj_out = self.traj_decoder(scene_repr, temporal_feat)

        # FIX: always compute losses when gt_trajs provided, regardless of self.training
        # train_oaiad_step passes gt_trajs — losses must never be empty dict on that path
        losses = {}
        if gt_trajs is not None:
            traj_losses = self.traj_decoder.nll_loss(traj_out, gt_trajs)
            losses.update(traj_losses)

        return {
            'trajectories': traj_out,
            'agent_feats':  agent_feats,
            'scene_repr':   scene_repr,
            'losses':       losses,
        }

    def get_ego_trajectory(self, forward_out, mode='best'):
        traj       = forward_out['trajectories']
        mode_probs = traj['mode_probs']
        refined    = traj['refined']

        if mode == 'best':
            best_k   = mode_probs.argmax(dim=-1)
            B        = refined.shape[0]
            ego_traj = refined[torch.arange(B), best_k, 0]
        elif mode == 'mean':
            probs    = mode_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ego_traj = (refined[:, :, 0] * probs).sum(dim=1)

        return ego_traj