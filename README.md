# Raw2Drive v3 — Autonomous Driving with World Models & RL

MTech Thesis | IIIT Allahabad | 2025–2027

Implementation and improvement of the Raw2Drive paper — end-to-end autonomous driving from raw sensor inputs using World Models, Reinforcement Learning, and BEVFormer-based perception.

---

## What This Does

Trains an autonomous driving agent that:
- Learns a **World Model** from raw camera/sensor data (no privileged info at inference)
- Uses **RL (Actor-Critic)** to learn driving policy
- Follows a 3-stage curriculum: Privileged WM → Raw WM → OAIAD Policy

---

## Key Improvements Over Baseline

| Change | Impact |
|--------|--------|
| Replaced linear layer with attention-weighted pooling | Raw WM: 1467M → 156.1M params (10x reduction) |
| AMP (torch.autocast + GradScaler) | 2–3x faster training, ~50% VRAM saved |
| Optimizer state checkpointing | No loss spikes on resume |
| OAIAD Stage 3 pipeline | Agent tracking + trajectory computation added |

---

## Stack

`PyTorch` `CARLA` `RL (Actor-Critic)` `BEVFormer` `RSSM` `Mixed Precision` `Python`

---

## Status

🔄 Training in progress — results and evaluation coming soon.

---

## References

- Raw2Drive Paper *(link TBD)*
- [CARLA Simulator](https://carla.org/)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
