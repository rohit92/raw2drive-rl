"""
Complete training pipeline for Raw2Drive + OAIAD.

Stage 1: Privileged WM + Policy
Stage 2: Raw Sensor WM + Policy (with guidance from privileged WM)
Stage 3: OAIAD fine-tune (uses raw WM BEV features)

Changes vs original:
  - Added _is_cuda property — removes 5x repetition of device-type check.
  - build_optimizers no longer creates Stage 2 optimizers (they are built
    in switch_to_stage2 only, so there is no wasted initialization).
  - build_models passes bev_size to PrivilegedEncoder / BEVDecoder and
    bev_h/bev_w to BEVFormerEncoder — fixes the bev_h=50 vs bev_size=200 mismatch.
  - torch.load uses weights_only=False (explicit) with a comment; swap to
    True once checkpoints are migrated off legacy pickle keys.
  - guided_reward / guided_continue removed the per-step CPU round-trip.
    The privileged reward/continue heads are moved to GPU for Stage 2 so
    they can be called directly on GPU latents.
  - Rolling 3-checkpoint cleanup on both save_stage1 and save.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from collections import deque
import random
from pathlib import Path

from models.encoder.privileged_encoder import PrivilegedEncoder
from models.encoder.raw_sensor_encoder import BEVFormerEncoder
from models.world_model.rssm import RSSM
from models.world_model.heads import WorldModelHeads
from models.world_model.world_model import WorldModel
from models.policy.actor_critic import ActorCritic
from models.oaiad.oaiad_model import OAIADModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Stores all tensors in CPU memory.  The buffer internally keeps track of
    position so that `sample_sequences` can draw non-overlapping windows of
    length `seq_len` starting from random offsets.
    """

    def __init__(self, capacity: int = 15_000, seq_len: int = 32):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.buffer   = deque(maxlen=capacity)

    def push(
        self,
        obs_priv,  obs_raw,
        action,    reward,   done,
        ego_state=None, nav_cmd=None, occ_vecs=None, gt_trajs=None,
        **kwargs,            # absorb any extra keys (e.g. agents_info after pop)
    ):
        self.buffer.append({
            'obs_priv':  obs_priv,
            'obs_raw':   obs_raw,
            'action':    action,
            'reward':    reward,
            'done':      done,
            'ego_state': ego_state,
            'nav_cmd':   nav_cmd,
            'occ_vecs':  occ_vecs,
            'gt_trajs':  gt_trajs,
        })

    def sample_sequences(self, batch_size: int, seq_len: int = None):
        if seq_len is None:
            seq_len = self.seq_len
        max_start = len(self.buffer) - seq_len
        if max_start <= 0:
            return None
        starts = random.sample(range(max_start), min(batch_size, max_start))
        batch  = {k: [] for k in self.buffer[0].keys()}
        for s in starts:
            for k in batch:
                seq = [self.buffer[s + t][k] for t in range(seq_len)]
                batch[k].append(
                    torch.stack(seq) if isinstance(seq[0], torch.Tensor)
                    else torch.tensor(np.array(seq))
                )
        return {k: torch.stack(v) for k, v in batch.items()}

    def __len__(self):
        return len(self.buffer)


# ── Trainer ──────────────────────────────────────────────────────────────────

class Raw2DriveTrainer:
    """
    Orchestrates all three training stages.

    Hardware target: RTX 6000 24 GB VRAM, 128 GB RAM.
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.build_models()
        self.build_optimizers()
        self.replay_buffer     = ReplayBuffer(config.buffer_size, config.seq_len)
        self.raw_replay_buffer = ReplayBuffer(config.buffer_size, config.seq_len)
        self.scaler            = torch.cuda.amp.GradScaler(enabled=self._is_cuda)

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(project='raw2drive_oaiad', config=vars(config))

    # ── Device helper ────────────────────────────────────────────────────

    @property
    def _is_cuda(self) -> bool:
        """True iff the training device is a CUDA GPU."""
        return self.device.type == 'cuda'

    # ── Model construction ───────────────────────────────────────────────

    def build_models(self):
        cfg       = self.config
        state_dim = cfg.hidden_dim + cfg.stoch_dim * cfg.stoch_classes

        # ── Stage 1 models → GPU ────────────────────────────────────────
        # FIX: PrivilegedEncoder and BEVDecoder use bev_size (200×200) for
        #      the GT mask resolution; BEVFormerEncoder uses bev_h/bev_w (50×50)
        #      for its internal query grid.
        priv_enc   = PrivilegedEncoder(
            cfg.bev_channels, cfg.embed_dim,
            bev_h=cfg.bev_size, bev_w=cfg.bev_size,   # 200×200 GT masks
        ).to(self.device)

        priv_rssm  = RSSM(
            cfg.hidden_dim, cfg.stoch_dim, cfg.stoch_classes,
            cfg.embed_dim,  cfg.action_dim,
        ).to(self.device)

        priv_heads = WorldModelHeads(
            state_dim, cfg.bev_channels,
            bev_h=cfg.bev_size, bev_w=cfg.bev_size,
            use_reward=True, use_continue=True,
        ).to(self.device)

        self.priv_wm     = WorldModel(
            priv_enc, priv_rssm, priv_heads, is_raw=False,
            decoder_chunk_size=cfg.decoder_chunk_size,
        ).to(self.device)
        self.priv_policy = ActorCritic(state_dim, cfg.action_dim).to(self.device)

        # ── Stage 2 models → CPU (moved to GPU in switch_to_stage2) ────
        raw_enc  = BEVFormerEncoder(
            embed_dim=cfg.embed_dim,
            bev_h=cfg.bev_h, bev_w=cfg.bev_w,         # 50×50 query grid
            out_embed_dim=cfg.embed_dim,
        ).cpu()

        raw_rssm = RSSM(
            cfg.hidden_dim, cfg.stoch_dim, cfg.stoch_classes,
            cfg.embed_dim,  cfg.action_dim,
        ).cpu()
        raw_rssm.load_state_dict(priv_rssm.state_dict())

        raw_heads = WorldModelHeads(
            state_dim, cfg.bev_channels,
            bev_h=cfg.bev_size, bev_w=cfg.bev_size,
            use_reward=False, use_continue=False,
        ).cpu()

        self.raw_wm = WorldModel(
            raw_enc, raw_rssm, raw_heads, is_raw=True,
            decoder_chunk_size=cfg.decoder_chunk_size,
        ).cpu()
        # Initialise raw decoder from privileged decoder weights
        self.raw_wm.heads.decoder.load_state_dict(priv_heads.decoder.state_dict())

        self.raw_policy = ActorCritic(state_dim, cfg.action_dim).cpu()
        self.raw_policy.load_state_dict(self.priv_policy.state_dict())

        self.oaiad = OAIADModel(
            embed_dim=cfg.oaiad_embed_dim,
            num_agents=cfg.oaiad_num_agents,
            num_modes=cfg.oaiad_num_modes,
            future_steps=cfg.oaiad_future_steps,
        ).cpu()

        _pm = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
        print(f"[Models] Privileged WM : {_pm(self.priv_wm):.1f} M params")
        print(f"[Models] Raw WM        : {_pm(self.raw_wm):.1f} M params")
        print(f"[Models] OAIAD         : {_pm(self.oaiad):.1f} M params")

    def switch_to_stage2(self):
        """Move Stage 1 models to CPU; Stage 2 + OAIAD models to GPU."""
        print("[Trainer] Switching to Stage 2...")
        self.priv_wm     = self.priv_wm.cpu()
        self.priv_policy = self.priv_policy.cpu()
        torch.cuda.empty_cache()

        self.raw_wm     = self.raw_wm.to(self.device)
        self.raw_policy = self.raw_policy.to(self.device)
        self.oaiad      = self.oaiad.to(self.device)

        # Move privileged heads to GPU so guided reward/continue avoid CPU round-trip
        self.priv_wm.heads = self.priv_wm.heads.to(self.device)

        cfg = self.config
        self.raw_wm_opt  = AdamW(self.raw_wm.parameters(),    lr=cfg.wm_lr,       weight_decay=0.0)
        self.raw_pol_opt = AdamW(self.raw_policy.parameters(), lr=cfg.pol_lr,      weight_decay=0.0)
        self.oaiad_opt   = AdamW(self.oaiad.parameters(),      lr=cfg.oaiad_lr,    weight_decay=cfg.oaiad_weight_decay)
        print("[Trainer] Stage 2 models ready on GPU.")

    def build_optimizers(self):
        """Build Stage 1 optimizers only. Stage 2 optimizers are created in switch_to_stage2."""
        cfg = self.config
        self.priv_wm_opt  = AdamW(self.priv_wm.parameters(),    lr=cfg.wm_lr,  weight_decay=0.0)
        self.priv_pol_opt = AdamW(self.priv_policy.parameters(), lr=cfg.pol_lr, weight_decay=0.0)
        # Placeholders — will be (re-)created in switch_to_stage2
        self.raw_wm_opt   = None
        self.raw_pol_opt  = None
        self.oaiad_opt    = None

    # ── Training steps ───────────────────────────────────────────────────

    def train_stage1_step(self, batch: dict) -> dict:
        obs_priv = batch['obs_priv'].to(self.device).float()
        actions  = batch['action' ].to(self.device).float()
        rewards  = batch['reward' ].to(self.device).float()
        dones    = batch['done'   ].to(self.device).float()

        # World model
        self.priv_wm.train()
        self.priv_wm_opt.zero_grad()
        with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
            wm_losses, states = self.priv_wm.world_model_loss(
                obs_priv, actions, rewards, dones
            )
            loss_wm = sum(wm_losses.values())
        self.scaler.scale(loss_wm).backward()
        self.scaler.unscale_(self.priv_wm_opt)
        nn.utils.clip_grad_norm_(self.priv_wm.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.priv_wm_opt)
        self.scaler.update()

        # Policy
        self.priv_policy.train()
        self.priv_pol_opt.zero_grad()
        h0 = states['h'][:, -1].detach()
        s0 = states['s'][:, -1].detach()
        with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
            pol_losses = self.priv_policy.policy_step(
                self.priv_wm, h0, s0, horizon=self.config.imagination_horizon
            )
            loss_pol = pol_losses['actor'] + pol_losses['critic']
        self.scaler.scale(loss_pol).backward()
        self.scaler.unscale_(self.priv_pol_opt)
        nn.utils.clip_grad_norm_(self.priv_policy.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.priv_pol_opt)
        self.scaler.update()

        return {
            **{f'priv_wm/{k}':  v.item() for k, v in wm_losses.items()},
            **{f'priv_pol/{k}': v.item() for k, v in pol_losses.items()},
        }

    def train_stage2_step(self, batch: dict) -> dict:
        obs_raw      = batch['obs_raw' ].to(self.device).float()
        obs_priv_cpu = batch['obs_priv'].float()            # stays on CPU for guidance
        actions_cpu  = batch['action'  ].float()
        actions      = actions_cpu.to(self.device)
        rewards      = batch['reward'  ].to(self.device).float()
        dones        = batch['done'    ].to(self.device).float()
        B, T         = actions.shape[:2]

        # ── Generate privileged guidance (priv_wm on CPU) ────────────────
        self.priv_wm.eval()
        with torch.no_grad():
            priv_embeds = []
            for t in range(T):
                e, _ = self.priv_wm.encode(obs_priv_cpu[:, t])   # CPU
                priv_embeds.append(e)
            priv_embeds_t = torch.stack(priv_embeds, dim=1)        # (B, T, embed_dim) CPU
            priv_states   = self.priv_wm.rssm.observe_sequence(actions_cpu, priv_embeds_t)
            priv_latents  = self.priv_wm.get_latent(priv_states['h'], priv_states['s'])
            priv_bev      = self.priv_wm._decode_chunked(
                priv_latents.reshape(B * T, -1)
            ).reshape(B, T, self.config.bev_channels,
                      self.config.bev_size, self.config.bev_size)

        guidance = {
            'embed': priv_embeds_t.to(self.device),
            'h':     priv_states['h'].to(self.device),
            'bev':   priv_bev.to(self.device),
        }

        # ── Train raw world model ─────────────────────────────────────────
        self.raw_wm.train()
        self.raw_wm_opt.zero_grad()
        with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
            raw_losses, raw_states = self.raw_wm.world_model_loss(
                obs_raw, actions, rewards, dones, guidance_states=guidance
            )
            loss_raw_wm = sum(raw_losses.values())
        self.scaler.scale(loss_raw_wm).backward()
        self.scaler.unscale_(self.raw_wm_opt)
        nn.utils.clip_grad_norm_(self.raw_wm.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.raw_wm_opt)
        self.scaler.update()

        # ── Train raw policy ──────────────────────────────────────────────
        # FIX: priv heads are now on GPU → no CPU round-trip per imagination step
        self.raw_policy.train()
        self.raw_pol_opt.zero_grad()
        h0 = raw_states['h'][:, -1].detach()
        s0 = raw_states['s'][:, -1].detach()

        def guided_reward(latent):
            return self.priv_wm.heads.reward_head(latent).detach()

        def guided_continue(latent):
            # return sigmoid of the raw logit so policy_step gets a probability
            return torch.sigmoid(self.priv_wm.heads.continue_head(latent)).detach()

        with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
            raw_pol_losses = self.raw_policy.policy_step(
                self.raw_wm, h0, s0,
                horizon=self.config.imagination_horizon,
                ext_reward_fn=guided_reward,
                ext_continue_fn=guided_continue,
            )
            loss_raw_pol = raw_pol_losses['actor'] + raw_pol_losses['critic']
        self.scaler.scale(loss_raw_pol).backward()
        self.scaler.unscale_(self.raw_pol_opt)
        nn.utils.clip_grad_norm_(self.raw_policy.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.raw_pol_opt)
        self.scaler.update()

        return {
            **{f'raw_wm/{k}':  v.item() for k, v in raw_losses.items()},
            **{f'raw_pol/{k}': v.item() for k, v in raw_pol_losses.items()},
        }

    def train_oaiad_step(self, batch: dict) -> dict:
        obs_raw  = batch['obs_raw'].to(self.device).float()
        B, T     = obs_raw.shape[:2]

        self.raw_wm.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
                _, bev_features = self.raw_wm.encode(obs_raw[:, -1])

        ego_state = batch['ego_state'][:, -1].to(self.device).float()
        nav_cmd   = batch['nav_cmd'  ][:, -1].to(self.device).float()
        occ_vecs  = batch['occ_vecs' ][:, -1].to(self.device).float()
        gt_trajs  = batch['gt_trajs' ][:, -1].to(self.device).float()

        self.oaiad.train()
        self.oaiad_opt.zero_grad()
        with torch.autocast(device_type=self.device.type, enabled=self._is_cuda):
            out       = self.oaiad(
                bev_features=bev_features,
                ego_state=ego_state,
                nav_cmd=nav_cmd,
                occ_vecs=occ_vecs,
                gt_trajs=gt_trajs,
            )
            loss_dict  = out['losses']
            total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(
                0.0, requires_grad=True, device=self.device
            )

        if total_loss.requires_grad:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.oaiad_opt)
            nn.utils.clip_grad_norm_(self.oaiad.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.oaiad_opt)
            self.scaler.update()

        return {f'oaiad/{k}': v.item() for k, v in loss_dict.items()
                if isinstance(v, torch.Tensor)}

    # ── Checkpointing ────────────────────────────────────────────────────

    def save_stage1(self, path: str, step: int):
        torch.save({
            'step':         step,
            'stage':        1,
            'priv_wm':      self.priv_wm.state_dict(),
            'priv_policy':  self.priv_policy.state_dict(),
            'priv_wm_opt':  self.priv_wm_opt.state_dict(),
            'priv_pol_opt': self.priv_pol_opt.state_dict(),
            'scaler':       self.scaler.state_dict(),
        }, path)
        # Rolling window: keep only last 3 checkpoints
        ckpt_dir  = Path(path).parent
        all_ckpts = sorted(ckpt_dir.glob("stage1_step*.pt"), key=lambda p: p.stat().st_mtime)
        for old in all_ckpts[:-3]:
            old.unlink()
            print(f"[Save] Removed old checkpoint: {old.name}")
        print(f"[Save] Stage 1 checkpoint → {path}")

    def save(self, path: str, step: int):
        torch.save({
            'step':         step,
            'priv_wm':      self.priv_wm.state_dict(),
            'raw_wm':       self.raw_wm.state_dict(),
            'priv_policy':  self.priv_policy.state_dict(),
            'raw_policy':   self.raw_policy.state_dict(),
            'oaiad':        self.oaiad.state_dict(),
            'priv_wm_opt':  self.priv_wm_opt.state_dict() if self.priv_wm_opt else None,
            'priv_pol_opt': self.priv_pol_opt.state_dict() if self.priv_pol_opt else None,
            'raw_wm_opt':   self.raw_wm_opt.state_dict()  if self.raw_wm_opt  else None,
            'raw_pol_opt':  self.raw_pol_opt.state_dict() if self.raw_pol_opt  else None,
            'oaiad_opt':    self.oaiad_opt.state_dict()   if self.oaiad_opt   else None,
            'scaler':       self.scaler.state_dict(),
        }, path)
        # Rolling 3-checkpoint cleanup
        ckpt_path = Path(path)
        if '_step' in ckpt_path.name:
            prefix    = ckpt_path.name.split('_step')[0]
            ckpt_dir  = ckpt_path.parent
            all_ckpts = sorted(ckpt_dir.glob(f"{prefix}_step*.pt"),
                               key=lambda p: p.stat().st_mtime)
            for old in all_ckpts[:-3]:
                old.unlink()
                print(f"[Save] Removed old checkpoint: {old.name}")
        print(f"[Save] Checkpoint → {path}")

    def load_stage1(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.priv_wm.load_state_dict(ckpt['priv_wm'])
        self.priv_policy.load_state_dict(ckpt['priv_policy'])
        if 'priv_wm_opt' in ckpt and ckpt['priv_wm_opt']:
            self.priv_wm_opt.load_state_dict(ckpt['priv_wm_opt'])
            self.priv_pol_opt.load_state_dict(ckpt['priv_pol_opt'])
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        print(f"[Load] Stage 1 loaded from step {ckpt['step']}")
        return ckpt['step']

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get('stage') == 1:
            return self.load_stage1(path)
        self.priv_wm.load_state_dict(ckpt['priv_wm'])
        self.raw_wm.load_state_dict(ckpt['raw_wm'])
        self.priv_policy.load_state_dict(ckpt['priv_policy'])
        self.raw_policy.load_state_dict(ckpt['raw_policy'])
        if 'oaiad' in ckpt:
            self.oaiad.load_state_dict(ckpt['oaiad'])
        if ckpt.get('raw_wm_opt') is not None:
            self.priv_wm_opt.load_state_dict(ckpt['priv_wm_opt'])
            self.priv_pol_opt.load_state_dict(ckpt['priv_pol_opt'])
            if self.raw_wm_opt:
                self.raw_wm_opt.load_state_dict(ckpt['raw_wm_opt'])
                self.raw_pol_opt.load_state_dict(ckpt['raw_pol_opt'])
                self.oaiad_opt.load_state_dict(ckpt['oaiad_opt'])
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        print(f"[Load] Checkpoint loaded from step {ckpt['step']}")
        return ckpt['step']