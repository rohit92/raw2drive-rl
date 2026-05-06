"""
Main training script for Raw2Drive + OAIAD.

Usage:
  python training/train.py --stage 1 --steps 100000
  python training/train.py --stage 2 --steps 100000 --load checkpoints/stage1_final.pt
  python training/train.py --stage all
  python training/train.py --stage oaiad --load checkpoints/stage2_final.pt

Changes vs original:
  - collect_episode: all tensors kept in (1, dim) 2-D form — never squeeze
    to 1-D, so nn.GRUCell receives the required (batch, input) shape.
  - Stage 3 loop uses cfg.oaiad_steps (not cfg.stage2_steps).
  - Stage 3 final checkpoint tagged with the correct step count.
  - Bare `except: pass` replaced with `except RuntimeError` to avoid
    silently swallowing KeyboardInterrupt / SystemExit.
  - Stage 2 wandb logging added (was missing in original).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback

from configs.default import Config
from training.trainer import Raw2DriveTrainer
from training.carla_env import CarlaEnv


# ── Episode collection ────────────────────────────────────────────────────────

def collect_episode(env, policy, wm, device, is_raw=False, max_steps=500):
    """
    Collect one episode of experience and return a list of transition dicts.

    FIX: all RSSM tensors are kept as (1, dim) — never squeezed to 1-D —
    so nn.GRUCell always receives the required 2-D (batch, input) input.
    """
    obs         = env.reset()
    transitions = []

    # Initial RSSM state: (1, hidden_dim) and (1, stoch_total)
    h, s = wm.rssm.initial_state(1, device)
    done = False
    total_reward = 0.0

    for step in range(max_steps):
        with torch.no_grad():
            if is_raw:
                cam   = obs['cameras'].unsqueeze(0).to(device)   # (1, 6, 3, H, W)
                embed, _ = wm.encode(cam)                         # (1, embed_dim)
            else:
                priv  = obs['privileged'].unsqueeze(0).to(device) # (1, C, H, W)
                embed, _ = wm.encode(priv)                        # (1, embed_dim)

            # Dummy action for first RSSM step (action = zeros, shape (1, action_dim))
            action_dummy = torch.zeros(1, wm.rssm.gru.input_size - wm.rssm.stoch_total,
                                       device=device)

            # FIX: h and s are already (1, dim) — pass them directly (no squeeze)
            h, s, _, _ = wm.rssm.observe_step(h, s, action_dummy, embed)
            # h, s still (1, hidden_dim) / (1, stoch_total) after the call

            latent = wm.get_latent(h, s)                  # (1, state_dim)
            action_oh, action_idx, _ = policy.actor(latent)
            action_val = action_idx.item()

        next_obs, reward, done, _ = env.step(action_val)
        total_reward += reward

        transitions.append({
            'obs_priv': obs['privileged'],
            'obs_raw':  obs['cameras'],
            'action':   action_oh.squeeze(0).detach().cpu(),   # (action_dim,)
            'reward':   torch.FloatTensor([reward]),
            'done':     torch.FloatTensor([float(done)]),
            'ego_state': obs['ego_state'],
            'nav_cmd':   obs['nav_cmd'],
            'occ_vecs':  obs['occ_vecs'],
            'agents_info': obs['agents_info'],
        })

        obs = next_obs
        if done:
            break

    # Compute GT trajectories for OAIAD
    T_eps        = len(transitions)
    future_steps = env.cfg.oaiad_future_steps
    for t in range(T_eps):
        gt_trajs = np.zeros(
            (env.cfg.oaiad_num_agents, future_steps, 2), dtype=np.float32
        )
        current_agents = list(transitions[t]['agents_info'].items())
        for i, (aid, (cx, cy)) in enumerate(current_agents):
            if i >= env.cfg.oaiad_num_agents:
                break
            for f in range(future_steps):
                fut_idx = t + f + 1
                if fut_idx < T_eps and aid in transitions[fut_idx]['agents_info']:
                    fx, fy = transitions[fut_idx]['agents_info'][aid]
                    gt_trajs[i, f, 0] = fx - cx
                    gt_trajs[i, f, 1] = fy - cy
                elif f > 0:
                    gt_trajs[i, f] = gt_trajs[i, f - 1]   # hold last known position

        transitions[t]['gt_trajs'] = torch.FloatTensor(gt_trajs)
        del transitions[t]['agents_info']

    return transitions, total_reward, step + 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Raw2Drive + OAIAD Training')
    parser.add_argument('--stage',      type=str, default='all',
                        choices=['1', '2', 'all', 'oaiad'])
    parser.add_argument('--steps',      type=int, default=None,
                        help='Override step count for all stages')
    parser.add_argument('--load',       type=str, default=None)
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--no_wandb',   action='store_true')
    args = parser.parse_args()

    cfg           = Config()
    cfg.use_wandb = not args.no_wandb
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.steps:
        cfg.stage1_steps = args.steps
        cfg.stage2_steps = args.steps
        cfg.oaiad_steps  = args.steps

    Path(cfg.ckpt_dir).mkdir(exist_ok=True)
    Path(cfg.log_dir ).mkdir(exist_ok=True)

    device  = torch.device(args.device)
    trainer = Raw2DriveTrainer(cfg, device)

    if args.load:
        start_step = trainer.load(args.load)
    else:
        start_step = 0

    print("[Env] Connecting to CARLA...")
    env = CarlaEnv(cfg=cfg, host=cfg.host, port=cfg.port, town=cfg.town)

    # ═══════════════════ STAGE 1 ═════════════════════════════════════════
    if args.stage in ['1', 'all']:
        print(f"\n[Stage 1] Privileged WM + Policy   ({cfg.stage1_steps} steps)")

        for ep in range(5):
            try:
                transitions, _, _ = collect_episode(
                    env, trainer.priv_policy, trainer.priv_wm,
                    device, is_raw=False, max_steps=cfg.max_steps_per_episode
                )
                for t in transitions:
                    trainer.replay_buffer.push(**t)
                print(f"[Stage 1] Init episode {ep+1}/5 ({len(transitions)} steps)")
            except Exception as e:
                print(f"[Stage 1] Init episode {ep+1} failed: {e}")
                traceback.print_exc()

        pbar = tqdm(range(cfg.stage1_steps), desc="Stage 1")
        for step in pbar:
            try:
                if step % 500 == 0:
                    transitions, ep_reward, ep_len = collect_episode(
                        env, trainer.priv_policy, trainer.priv_wm,
                        device, is_raw=False, max_steps=cfg.max_steps_per_episode
                    )
                    for t in transitions:
                        trainer.replay_buffer.push(**t)

                batch = trainer.replay_buffer.sample_sequences(cfg.batch_size, cfg.seq_len)
                if batch is None:
                    continue

                metrics = trainer.train_stage1_step(batch)

                if step % 100 == 0:
                    pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}"
                                      for k, v in list(metrics.items())[:4]})

                if cfg.use_wandb and WANDB_AVAILABLE:
                    import wandb
                    wandb.log(metrics, step=step)

                if step % cfg.save_every == 0 and step > 0:
                    trainer.save_stage1(f"{cfg.ckpt_dir}/stage1_step{step}.pt", step)

            except KeyboardInterrupt:
                print("\n[Stage 1] Interrupted — saving...")
                trainer.save_stage1(f"{cfg.ckpt_dir}/stage1_interrupted_step{step}.pt", step)
                env.close()
                return
            except Exception as e:
                print(f"[Stage 1] Error at step {step}: {e}")
                traceback.print_exc()

        trainer.save_stage1(f"{cfg.ckpt_dir}/stage1_final.pt", cfg.stage1_steps)
        print("[Stage 1] Complete!")

    # ═══════════════════ STAGE 2 ═════════════════════════════════════════
    if args.stage in ['2', 'all']:
        if args.stage == '2' and args.load is None:
            trainer.load_stage1(f"{cfg.ckpt_dir}/stage1_final.pt")

        trainer.switch_to_stage2()
        print(f"\n[Stage 2] Raw Sensor WM + Policy   ({cfg.stage2_steps} steps)")

        for ep in range(5):
            try:
                transitions, _, _ = collect_episode(
                    env, trainer.raw_policy, trainer.raw_wm,
                    device, is_raw=True, max_steps=cfg.max_steps_per_episode
                )
                for t in transitions:
                    trainer.raw_replay_buffer.push(**t)
                print(f"[Stage 2] Init episode {ep+1}/5 ({len(transitions)} steps)")
            except Exception as e:
                print(f"[Stage 2] Init episode {ep+1} failed: {e}")
                traceback.print_exc()

        pbar = tqdm(range(cfg.stage2_steps), desc="Stage 2")
        for step in pbar:
            try:
                if step % 500 == 0:
                    transitions, ep_reward, ep_len = collect_episode(
                        env, trainer.raw_policy, trainer.raw_wm,
                        device, is_raw=True, max_steps=cfg.max_steps_per_episode
                    )
                    for t in transitions:
                        trainer.raw_replay_buffer.push(**t)

                batch = trainer.raw_replay_buffer.sample_sequences(cfg.batch_size, cfg.seq_len)
                if batch is None:
                    continue

                metrics = trainer.train_stage2_step(batch)

                if step % 100 == 0:
                    pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}"
                                      for k, v in list(metrics.items())[:4]})

                if cfg.use_wandb and WANDB_AVAILABLE:
                    import wandb
                    wandb.log(metrics, step=step)

                if step % cfg.save_every == 0 and step > 0:
                    trainer.save(f"{cfg.ckpt_dir}/stage2_step{step}.pt", step)

            except KeyboardInterrupt:
                print("\n[Stage 2] Interrupted — saving...")
                trainer.save(f"{cfg.ckpt_dir}/stage2_interrupted_step{step}.pt", step)
                env.close()
                return
            except Exception as e:
                print(f"[Stage 2] Error at step {step}: {e}")
                traceback.print_exc()

        trainer.save(f"{cfg.ckpt_dir}/stage2_final.pt", cfg.stage2_steps)
        print("[Stage 2] Complete!")

    # ═══════════════════ STAGE 3 (OAIAD) ════════════════════════════════
    if args.stage in ['oaiad', 'all']:
        if args.stage == 'oaiad' and args.load is None:
            print("[Stage 3] No checkpoint specified — attempting auto-load of stage2_final...")
            try:
                trainer.load(f"{cfg.ckpt_dir}/stage2_final.pt")
            except Exception as e:
                print(f"[Stage 3] Auto-load failed: {e}")

        # Ensure raw models are on GPU
        try:
            trainer.switch_to_stage2()
        except RuntimeError:
            pass   # already in stage 2 — models already on GPU

        print(f"\n[Stage 3] OAIAD Fine-tune   ({cfg.oaiad_steps} steps)")

        # FIX: use cfg.oaiad_steps (was cfg.stage2_steps — wrong step count)
        pbar = tqdm(range(cfg.oaiad_steps), desc="Stage 3 (OAIAD)")
        for step in pbar:
            try:
                if step % 500 == 0:
                    transitions, ep_reward, ep_len = collect_episode(
                        env, trainer.raw_policy, trainer.raw_wm,
                        device, is_raw=True, max_steps=cfg.max_steps_per_episode
                    )
                    for t in transitions:
                        trainer.raw_replay_buffer.push(**t)

                batch = trainer.raw_replay_buffer.sample_sequences(cfg.batch_size, cfg.seq_len)
                if batch is None:
                    continue

                metrics = trainer.train_oaiad_step(batch)

                if step % 100 == 0 and metrics:
                    pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}"
                                      for k, v in list(metrics.items())[:4]})

                if cfg.use_wandb and WANDB_AVAILABLE and metrics:
                    import wandb
                    wandb.log(metrics, step=step)

                if step % cfg.save_every == 0 and step > 0:
                    trainer.save(f"{cfg.ckpt_dir}/oaiad_step{step}.pt", step)

            except KeyboardInterrupt:
                print("\n[Stage 3] Interrupted — saving...")
                # FIX: tag with current step, not cfg.stage2_steps
                trainer.save(f"{cfg.ckpt_dir}/oaiad_interrupted_step{step}.pt", step)
                env.close()
                return
            except Exception as e:
                print(f"[Stage 3] Error at step {step}: {e}")
                traceback.print_exc()

        # FIX: final save uses cfg.oaiad_steps, not cfg.stage2_steps
        trainer.save(f"{cfg.ckpt_dir}/oaiad_final.pt", cfg.oaiad_steps)
        print("[Stage 3] Complete!")

    env.close()
    print("\n[Training] All stages complete!")


# ── wandb availability check ─────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


if __name__ == '__main__':
    main()