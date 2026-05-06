"""
Actor-Critic Policy for MBRL (DreamerV3-style).

Changes vs original:
  - policy_step now stores the action distribution from the rollout loop and
    passes it to actor_loss — avoids a second full forward pass through the
    actor network (was doubling compute cost).
  - actor_loss updated to accept pre-computed distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscreteActor(nn.Module):
    """
    Discrete action actor.
    Maps latent state → action logits over `action_dim` discrete actions.
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int = 39,
        hidden:     int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        layers  = []
        in_dim  = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, action_dim))
        self.net        = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(self, latent: torch.Tensor, sample: bool = True, temperature: float = 1.0):
        logits    = self.net(latent) / temperature
        dist      = Categorical(logits=logits)
        action    = dist.sample() if sample else logits.argmax(-1)
        action_oh = F.one_hot(action, self.action_dim).float()
        return action_oh, action, dist

    def get_action_onehot(self, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        action_oh, _, _ = self(latent, **kwargs)
        return action_oh


class Critic(nn.Module):
    """Critic: predicts scalar value V(z) for a latent state z."""

    def __init__(self, state_dim: int, hidden: int = 512, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)   # (B,) scalar


class ActorCritic(nn.Module):
    """Combined actor-critic with DreamerV3-style training."""

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int   = 39,
        gamma:        float = 0.99,
        lam:          float = 0.95,
        entropy_coef: float = 3e-4,
    ):
        super().__init__()
        self.actor         = DiscreteActor(state_dim, action_dim)
        self.critic        = Critic(state_dim)
        self.target_critic = Critic(state_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.gamma        = gamma
        self.lam          = lam
        self.entropy_coef = entropy_coef

    @torch.no_grad()
    def update_target(self, tau: float = 0.98):
        """Soft update: target ← τ·target + (1-τ)·online."""
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.data.lerp_(p.data, 1.0 - tau)

    def compute_returns(
        self,
        rewards:    torch.Tensor,
        continues:  torch.Tensor,
        values:     torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        λ-return computation (DreamerV3).

        rewards, continues, values: (T, B)
        last_value: (B,)
        """
        T, B    = rewards.shape
        returns = torch.zeros_like(rewards)
        last    = last_value
        for t in reversed(range(T)):
            last      = rewards[t] + self.gamma * continues[t] * (
                (1.0 - self.lam) * values[t] + self.lam * last
            )
            returns[t] = last
        return returns

    def actor_loss(
        self,
        latents:   torch.Tensor,
        actions:   torch.Tensor,
        returns:   torch.Tensor,
        values:    torch.Tensor,
        dists=None,             # pre-computed distributions (avoids double forward)
    ) -> torch.Tensor:
        """Policy-gradient loss with entropy regularisation."""
        if dists is None:
            _, _, dist = self.actor(latents)
        else:
            dist = dists

        log_probs = dist.log_prob(actions.argmax(-1))
        adv       = returns - values.detach()
        S         = torch.clamp(
            torch.quantile(adv.abs().float(), 0.95) - torch.quantile(adv.abs().float(), 0.05),
            min=1.0,
        )
        adv_norm  = adv / S
        pol_loss  = -(adv_norm.detach() * log_probs).mean()
        ent_loss  = -self.entropy_coef * dist.entropy().mean()
        return pol_loss + ent_loss

    def critic_loss(self, latents: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        values = self.critic(latents)
        return F.mse_loss(values, returns.detach())

    def policy_step(
        self,
        world_model,
        h0: torch.Tensor,
        s0: torch.Tensor,
        horizon: int = 15,
        ext_reward_fn=None,
        ext_continue_fn=None,
    ) -> dict:
        """Full imagination rollout + loss computation."""
        h, s = h0.detach(), s0.detach()
        latents, actions_oh, rewards, continues, dists = [], [], [], [], []

        for _ in range(horizon):
            latent              = torch.cat([h, s], dim=-1)
            # FIX: store dist here — reused in actor_loss (no second forward pass)
            action_oh, _, dist  = self.actor(latent)

            if ext_reward_fn is not None:
                r = ext_reward_fn(latent)
                c = ext_continue_fn(latent)
            else:
                r = world_model.heads.reward_head(latent)
                # continue_head returns raw logit — sigmoid for actual probability
                c = torch.sigmoid(world_model.heads.continue_head(latent))

            latents.append(latent)
            actions_oh.append(action_oh)
            dists.append(dist)
            rewards.append(r.squeeze(-1))    # ensure (B,)
            continues.append(c.squeeze(-1))  # ensure (B,)

            h, s = world_model.rssm.imagine_step(h, s, action_oh)

        last_latent = torch.cat([h, s], dim=-1)
        T           = horizon

        latents_t   = torch.stack(latents)       # (T, B, state_dim)
        rewards_t   = torch.stack(rewards)       # (T, B)
        continues_t = torch.stack(continues)     # (T, B)

        with torch.no_grad():
            values_t   = self.target_critic(
                latents_t.reshape(-1, latents_t.shape[-1])
            ).reshape(T, -1)                     # (T, B)
            last_value = self.target_critic(last_latent)   # (B,)

        returns = self.compute_returns(rewards_t, continues_t, values_t, last_value)

        # Flatten for loss computation
        lat_flat  = latents_t.reshape(-1, latents_t.shape[-1])
        act_flat  = torch.stack(actions_oh).reshape(-1, self.actor.action_dim)
        ret_flat  = returns.reshape(-1)
        val_flat  = values_t.reshape(-1)

        # Merge per-step distributions into a single batched distribution
        # by stacking logits from each step
        all_logits = torch.stack([d.logits for d in dists])   # (T, B, action_dim)
        merged_dist = Categorical(logits=all_logits.reshape(-1, self.actor.action_dim))

        a_loss = self.actor_loss(lat_flat, act_flat, ret_flat, val_flat, merged_dist)
        c_loss = self.critic_loss(lat_flat.detach(), ret_flat)

        self.update_target()
        return {'actor': a_loss, 'critic': c_loss}