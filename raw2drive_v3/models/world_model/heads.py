"""
World Model Heads: Decoder, Reward, Continue.

Changes vs original:
  - Fixed operator-precedence bug in BEVDecoder `mid` calculation.
  - Removed torch.sigmoid from ContinueHead.forward — caller must use
    F.binary_cross_entropy_with_logits (numerically stable).
  - WorldModelHeads.forward returns raw logit for 'continue'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDecoder(nn.Module):
    """
    Decodes latent state → BEV semantic mask logits.
    Used by BOTH privileged WM and raw sensor WM.

    Input:  latent = cat(h, s),  shape (B, state_dim)
    Output: BEV mask logits,     shape (B, bev_channels, bev_size, bev_size)
    """

    def __init__(
        self,
        state_dim:    int,
        bev_channels: int = 43,
        bev_h:        int = 200,
        bev_w:        int = 200,
    ):
        super().__init__()
        self.bev_h        = bev_h
        self.bev_w        = bev_w
        self.bev_channels = bev_channels

        # Spatial size at the start of the convolutional decoder
        h0 = bev_h // 4   # 50 when bev_h=200
        w0 = bev_w // 4   # 50 when bev_w=200

        # FIX: added parentheses so precedence is unambiguous
        # mid was: bev_h // 4 * bev_w // 4  → wrong for non-square BEV
        mid = (bev_h // 4) * (bev_w // 4)   # noqa: F841 — kept for clarity

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, bev_channels * h0 * w0),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bev_channels, 128, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.Conv2d(64, bev_channels, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B  = latent.shape[0]
        h0 = self.bev_h // 4
        w0 = self.bev_w // 4
        x  = self.mlp(latent)
        x  = x.view(B, self.bev_channels, h0, w0)
        return self.decoder(x)   # (B, bev_channels, bev_h, bev_w)


class RewardHead(nn.Module):
    """Predicts scalar reward from latent state."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)   # (B,)


class ContinueHead(nn.Module):
    """
    Predicts binary continue flag (episode not done).

    FIX: Returns raw logit — apply sigmoid externally only for inference.
    Use F.binary_cross_entropy_with_logits during training for numerical
    stability (avoids log(0) when predictions are exactly 0 or 1).
    """

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (B,). Apply torch.sigmoid for probabilities."""
        return self.net(latent).squeeze(-1)   # (B,)  ← raw logit, no sigmoid


class WorldModelHeads(nn.Module):
    """Container for all world-model prediction heads."""

    def __init__(
        self,
        state_dim:    int,
        bev_channels: int = 43,
        bev_h:        int = 200,
        bev_w:        int = 200,
        use_reward:   bool = True,
        use_continue: bool = True,
    ):
        super().__init__()
        self.decoder      = BEVDecoder(state_dim, bev_channels, bev_h, bev_w)
        self.reward_head  = RewardHead(state_dim)  if use_reward  else None
        self.continue_head = ContinueHead(state_dim) if use_continue else None

    def forward(self, latent: torch.Tensor) -> dict:
        out = {'bev': self.decoder(latent)}
        if self.reward_head is not None:
            out['reward'] = self.reward_head(latent)
        if self.continue_head is not None:
            # Raw logit — caller decides whether to sigmoid (inference) or BCE-logits (train)
            out['continue'] = self.continue_head(latent)
        return out
