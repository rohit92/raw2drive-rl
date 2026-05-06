"""
Complete World Model: Encoder + RSSM + Heads.

Changes vs original:
  - F.binary_cross_entropy → F.binary_cross_entropy_with_logits for the
    continue head (ContinueHead now returns a raw logit, not a sigmoid prob).
  - GT BEV reshape uses bev_size consistently.
  - _decode_chunked uses configurable chunk size.
  - imagine_ahead: rewards/continues stacked only once (no double allocation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .rssm import RSSM
from .heads import WorldModelHeads


class WorldModel(nn.Module):
    """
    Base World Model used by both privileged and raw sensor streams.

    Args:
        encoder:           feature extractor (PrivilegedEncoder or BEVFormerEncoder)
        rssm:              recurrent state space model
        heads:             prediction heads (decoder, reward, continue)
        is_raw:            if True, raw-sensor stream (no reward/continue heads)
        decoder_chunk_size: split B*T latents into chunks to stay within VRAM
    """

    def __init__(
        self,
        encoder: nn.Module,
        rssm: RSSM,
        heads: WorldModelHeads,
        is_raw: bool = False,
        decoder_chunk_size: int = 16,
    ):
        super().__init__()
        self.encoder            = encoder
        self.rssm               = rssm
        self.heads              = heads
        self.is_raw             = is_raw
        self.decoder_chunk_size = decoder_chunk_size

    @property
    def state_dim(self) -> int:
        return self.rssm.state_dim

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode(self, obs: torch.Tensor):
        """
        Encode a single-timestep observation.

        Returns:
            embed    (B, embed_dim)
            bev_grid (B, bev_h*bev_w, D) or None
        """
        if self.is_raw:
            embed, bev_grid = self.encoder(obs)
            return embed, bev_grid
        else:
            embed = self.encoder(obs)
            return embed, None

    # ── Utilities ────────────────────────────────────────────────────────

    def get_latent(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and stochastic states → (B, state_dim)."""
        return torch.cat([h, s], dim=-1)

    def _decode_chunked(self, latents_flat: torch.Tensor) -> torch.Tensor:
        """
        Run BEV decoder in chunks to avoid OOM on large B*T tensors.

        latents_flat: (B*T, state_dim)
        returns:      (B*T, bev_channels, bev_h, bev_w)
        """
        chunks = []
        for i in range(0, latents_flat.shape[0], self.decoder_chunk_size):
            chunk = latents_flat[i : i + self.decoder_chunk_size]
            chunks.append(self.heads.decoder(chunk))
        return torch.cat(chunks, dim=0)

    # ── Imagination ──────────────────────────────────────────────────────

    def imagine_ahead(self, h0: torch.Tensor, s0: torch.Tensor,
                      actor, horizon: int = 15) -> dict:
        """
        Imagine trajectories using actor policy (for policy training).

        Returns dict with keys: latents, actions, rewards (or None), continues (or None).
        """
        h, s = h0, s0
        latents, actions, rewards, continues = [], [], [], []

        for _ in range(horizon):
            latent     = self.get_latent(h, s)
            action     = actor(latent)

            if not self.is_raw:
                head_out = self.heads(latent)
                rewards.append(head_out['reward'])
                # continue_head returns raw logit — sigmoid here for value
                continues.append(torch.sigmoid(head_out['continue']))

            latents.append(latent)
            actions.append(action)
            h, s = self.rssm.imagine_step(h, s, action)

        return {
            'latents':   torch.stack(latents),
            'actions':   torch.stack(actions),
            'rewards':   torch.stack(rewards)   if rewards   else None,
            'continues': torch.stack(continues) if continues else None,
        }

    # ── Loss computation ─────────────────────────────────────────────────

    def world_model_loss(
        self,
        obs_seq:          torch.Tensor,
        action_seq:       torch.Tensor,
        reward_seq:       torch.Tensor,
        done_seq:         torch.Tensor,
        guidance_states:  Optional[dict] = None,
    ) -> tuple:
        """
        Compute world model training loss.

        obs_seq:        (B, T, ...) observations
        action_seq:     (B, T, action_dim)
        reward_seq:     (B, T)
        done_seq:       (B, T)
        guidance_states: from privileged WM for alignment (raw stream only)

        Returns: (loss_dict, states)
        """
        B, T = action_seq.shape[:2]

        # Encode all time-steps
        embeds, bev_grids = [], []
        for t in range(T):
            e, g = self.encode(obs_seq[:, t])
            embeds.append(e)
            bev_grids.append(g)
        embeds = torch.stack(embeds, dim=1)    # (B, T, embed_dim)

        # RSSM sequence
        states  = self.rssm.observe_sequence(action_seq, embeds)
        latents = self.get_latent(states['h'], states['s'])   # (B, T, state_dim)

        loss_dict    = {}
        latents_flat = latents.reshape(B * T, -1)

        # ── BEV reconstruction (chunked to avoid OOM) ────────────────────
        loss_recon = 0.0
        total_elements = 0

        for i in range(0, latents_flat.shape[0], self.decoder_chunk_size):
            chunk_latents = latents_flat[i : i + self.decoder_chunk_size]
            bev_pred_chunk = self.heads.decoder(chunk_latents)

            if not self.is_raw:
                # Privileged stream: reconstruct GT BEV masks
                bev_gt_chunk = obs_seq.reshape(B * T, *obs_seq.shape[2:])[i : i + self.decoder_chunk_size]
                loss_recon = loss_recon + F.binary_cross_entropy_with_logits(
                    bev_pred_chunk, bev_gt_chunk.float(), reduction='sum'
                )
                total_elements += bev_gt_chunk.numel()
            else:
                # Raw stream: supervised by privileged BEV predictions
                if guidance_states is not None and 'bev' in guidance_states:
                    bev_guidance_chunk = guidance_states['bev'].reshape(B * T, *bev_pred_chunk.shape[1:])[i : i + self.decoder_chunk_size]
                    loss_recon = loss_recon + F.mse_loss(
                        torch.sigmoid(bev_pred_chunk),
                        torch.sigmoid(bev_guidance_chunk),
                        reduction='sum'
                    )
                    total_elements += bev_guidance_chunk.numel()

        if total_elements > 0:
            loss_dict['recon'] = loss_recon / total_elements

        # ── Reward + continue loss (privileged only) ─────────────────────
        if not self.is_raw and self.heads.reward_head is not None:
            reward_pred = self.heads.reward_head(latents_flat)
            loss_dict['reward'] = F.mse_loss(reward_pred, reward_seq.reshape(-1))

        if not self.is_raw and self.heads.continue_head is not None:
            # FIX: use BCE-with-logits (ContinueHead returns raw logit now)
            cont_logit = self.heads.continue_head(latents_flat)
            loss_dict['continue'] = F.binary_cross_entropy_with_logits(
                cont_logit, (1.0 - done_seq).reshape(-1).float()
            )

        # ── KL loss (free-nats clipping per DreamerV3) ───────────────────
        kl_loss = torch.zeros(1, device=action_seq.device)
        for prior_d, post_d in zip(states['prior'], states['post']):
            kl = torch.distributions.kl_divergence(post_d, prior_d).sum(-1).mean()
            kl_loss = kl_loss + torch.clamp(kl, min=1.0)
        loss_dict['kl'] = kl_loss / T

        # ── Guidance alignment losses (raw stream only) ───────────────────
        if self.is_raw and guidance_states is not None:
            if 'embed' in guidance_states:
                loss_dict['align_embed'] = 10.0 * F.mse_loss(
                    embeds, guidance_states['embed']
                )
            if 'h' in guidance_states:
                loss_dict['align_h'] = 5.0 * F.mse_loss(
                    states['h'], guidance_states['h']
                )
            if 's_dist' in guidance_states:
                kl_align = torch.distributions.kl_divergence(
                    guidance_states['s_dist'],
                    type(guidance_states['s_dist'])(
                        logits=states['s'].view(
                            *states['s'].shape[:-1],
                            self.rssm.stoch_dim,
                            self.rssm.stoch_classes,
                        )
                    ),
                ).sum(-1).mean()
                loss_dict['align_s'] = 10.0 * kl_align

        return loss_dict, states