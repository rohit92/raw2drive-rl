"""Default configuration for Raw2Drive + OAIAD.

Hardware target: NVIDIA RTX 6000 Ada 24 GB VRAM, 128 GB RAM, 7.5 TB storage.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────
    hidden_dim: int = 512
    stoch_dim: int = 32
    stoch_classes: int = 32
    embed_dim: int = 512        # global scene embedding fed into RSSM
    action_dim: int = 39

    # ── BEV dimensions ────────────────────────────────────────────────────
    # Privileged BEV semantic masks   → (B, bev_channels, bev_size, bev_size)
    bev_channels: int = 43
    bev_size: int = 200         # spatial size of GT / decoded BEV masks (H = W)

    # BEVFormer query grid (raw sensor encoder) — kept smaller to save VRAM
    bev_h: int = 50             # BEVFormer spatial query height
    bev_w: int = 50             # BEVFormer spatial query width

    # Camera input
    img_w: int = 400
    img_h: int = 225

    target_speed: float = 8.0   # m/s

    # ── OAIAD ─────────────────────────────────────────────────────────────
    oaiad_embed_dim: int = 256
    oaiad_num_agents: int = 20
    oaiad_num_modes: int = 6
    oaiad_future_steps: int = 12

    # ── Training ──────────────────────────────────────────────────────────
    stage1_steps: int = 100_000
    stage2_steps: int = 100_000
    oaiad_steps:  int = 100_000  # dedicated budget for Stage 3 (OAIAD fine-tune)

    # RTX 6000 24 GB — batch_size=8 + AMP + chunked decode is safe
    batch_size: int = 8
    seq_len: int = 32
    imagination_horizon: int = 15

    wm_lr:  float = 1e-5
    pol_lr: float = 3e-5
    oaiad_lr: float = 2e-4
    oaiad_weight_decay: float = 1e-2
    grad_clip_norm: float = 100.0

    # 128 GB RAM — 15 k steps ≈ 50 GB with uint8 image storage
    buffer_size: int = 15_000
    max_steps_per_episode: int = 500

    # ── CARLA ─────────────────────────────────────────────────────────────
    host: str = 'localhost'
    port: int = 2000
    town: str = 'Town01'
    num_envs: int = 4

    # ── Evaluation / checkpointing ────────────────────────────────────────
    eval_every: int = 5_000
    save_every: int = 500

    # ── Logging ───────────────────────────────────────────────────────────
    use_wandb: bool = True
    log_dir: str = 'logs'
    ckpt_dir: str = 'checkpoints'

    # ── Hardware helpers ──────────────────────────────────────────────────
    # Process BEV decoder in chunks to stay within 24 GB VRAM
    decoder_chunk_size: int = 16
