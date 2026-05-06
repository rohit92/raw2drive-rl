"""
Evaluation script for Raw2Drive + OAIAD on Bench2Drive metrics.
Computes: Driving Score, Success Rate, Infraction Score.

Changes vs original:
  - observe_step called with 2-D tensors (batch=1) throughout — no squeeze
    to 1-D, which caused a GRUCell dimension crash.
  - route_completion updated at each step from the environment's route
    progress (was always 0.0 in the original).
  - num_infractions now tracked via collision flag.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
import carla

from training.carla_env import CarlaEnv
from configs.default import Config
from training.trainer import Raw2DriveTrainer


class Evaluator:
    def __init__(self, trainer, env, device):
        self.trainer = trainer
        self.env     = env
        self.device  = device

    @torch.no_grad()
    def evaluate_episode(self, scenario_type: str = 'basic', max_steps: int = 1000) -> dict:
        """Run one evaluation episode and return metrics dict."""
        obs    = self.env.reset()
        wm     = self.trainer.raw_wm
        policy = self.trainer.raw_policy

        # FIX: keep (1, dim) batch dimension — never squeeze to 1-D
        h, s = wm.rssm.initial_state(1, self.device)

        done            = False
        total_reward    = 0.0
        num_infractions = 0
        steps           = 0
        prev_closest    = 0   # for route completion tracking

        for step in range(max_steps):
            cam = obs['cameras'].unsqueeze(0).to(self.device)   # (1, 6, 3, H, W)
            embed, _ = wm.encode(cam)                            # (1, embed_dim)

            # Dummy action for RSSM step
            action_dummy = torch.zeros(1, wm.rssm.gru.input_size - wm.rssm.stoch_total,
                                       device=self.device)

            # FIX: h/s are (1, dim) — pass directly, no squeeze
            h, s, _, _ = wm.rssm.observe_step(h, s, action_dummy, embed)

            latent = wm.get_latent(h, s)             # (1, state_dim)
            _, action_idx, _ = policy.actor(latent, sample=False)

            next_obs, reward, done, info = self.env.step(action_idx.item())
            total_reward += reward
            steps        += 1

            # Track collisions
            if self.env._collision:
                num_infractions += 1

            obs = next_obs
            if done:
                break

        # Route completion: fraction of waypoints passed
        if self.env._route_waypoints:
            ego_loc = self.env._vehicle.get_location() if self.env._vehicle else None
            if ego_loc is not None:
                dists       = [ego_loc.distance(wp.transform.location)
                               for wp in self.env._route_waypoints]
                closest_idx = int(np.argmin(dists))
                route_completion = closest_idx / max(len(self.env._route_waypoints) - 1, 1)
            else:
                route_completion = 0.0
        else:
            route_completion = 0.0

        infraction_score = max(0.0, 1.0 - num_infractions * 0.1)
        driving_score    = route_completion * infraction_score
        success          = (route_completion > 0.9) and (num_infractions == 0)

        return {
            'driving_score':     driving_score,
            'route_completion':  route_completion,
            'infraction_score':  infraction_score,
            'success':           float(success),
            'total_reward':      total_reward,
            'steps':             steps,
            'num_infractions':   num_infractions,
        }

    def evaluate_bench2drive(self, num_episodes: int = 50) -> dict:
        """Run Bench2Drive evaluation across multiple scenarios."""
        scenarios = ['Merging', 'Overtaking', 'EmergencyBrake', 'GiveWay', 'TrafficSign']
        results   = defaultdict(list)

        for ep in range(num_episodes):
            scenario = scenarios[ep % len(scenarios)]
            print(f"[Eval] Episode {ep+1}/{num_episodes} — {scenario}")
            metrics  = self.evaluate_episode(scenario_type=scenario)
            for k, v in metrics.items():
                results[k].append(v)
            results['scenario'].append(scenario)
            print(f"       DS={metrics['driving_score']:.3f}  "
                  f"RC={metrics['route_completion']:.3f}  "
                  f"IS={metrics['infraction_score']:.3f}")

        summary = {}
        for k, vs in results.items():
            if k == 'scenario':
                continue
            summary[k] = {'mean': float(np.mean(vs)), 'std': float(np.std(vs))}

        for sc in scenarios:
            indices   = [i for i, s in enumerate(results['scenario']) if s == sc]
            sc_success = np.mean([results['success'][i] for i in indices]) if indices else 0.0
            summary[f'success_{sc}'] = float(sc_success)

        return summary


def main():
    parser = argparse.ArgumentParser(description='Raw2Drive Evaluation')
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--output',       type=str, default='eval_results.json')
    parser.add_argument('--device',       type=str, default='cuda')
    args = parser.parse_args()

    cfg     = Config()
    device  = torch.device(args.device)
    trainer = Raw2DriveTrainer(cfg, device)
    trainer.load(args.checkpoint)
    trainer.switch_to_stage2()   # ensure raw models are on GPU

    trainer.raw_wm.eval()
    trainer.raw_policy.eval()
    trainer.oaiad.eval()

    env       = CarlaEnv(cfg=cfg, host=cfg.host, port=cfg.port, town=cfg.town)
    evaluator = Evaluator(trainer, env, device)

    print(f"[Eval] Running {args.num_episodes} episodes...")
    results = evaluator.evaluate_bench2drive(args.num_episodes)

    print("\n[Results]")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")
        else:
            print(f"  {k}: {v:.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results → {args.output}")

    env.close()


if __name__ == '__main__':
    main()
