"""
Recurrent State Space Model (RSSM) for Raw2Drive.
Implements both Privileged and Raw Sensor World Models.

Changes vs original:
  - Fixed LayerNormGRUCell.forward — was using a wrong weight-sum instead of
    a proper reset-gated candidate state. Now uses separate ih/hh projections
    with individual layer-norms for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class LayerNormGRUCell(nn.Module):
    """
    GRU cell with per-gate layer normalisation.

    Implements the standard GRU update equations:
        r = σ( LN(Wir·x + Whr·h) )
        u = σ( LN(Wiu·x + Whu·h) )
        n =tanh( LN(Win·x + r ⊙ Whn·h) )
        h' = (1 - u) ⊙ n  +  u ⊙ h

    Using LN instead of bias for stability (DreamerV3 style).
    Note: currently not used by RSSM (which uses nn.GRUCell directly),
    but available as a drop-in upgrade.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Separate input→hidden and hidden→hidden projections for r and u gates
        self.ih_ru = nn.Linear(input_size,  2 * hidden_size, bias=False)
        self.hh_ru = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.ln_ru = nn.LayerNorm(2 * hidden_size)

        # Separate projections for the candidate state n
        self.ih_n  = nn.Linear(input_size,  hidden_size, bias=False)
        self.hh_n  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_n  = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # r, u gates
        ru   = torch.sigmoid(self.ln_ru(self.ih_ru(x) + self.hh_ru(h)))
        r, u = ru.chunk(2, dim=-1)

        # Candidate state — reset gate applied to recurrent branch only
        n = torch.tanh(self.ln_n(self.ih_n(x) + r * self.hh_n(h)))

        return (1.0 - u) * n + u * h


class RSSM(nn.Module):
    """
    Recurrent State Space Model from DreamerV3.

    h_t = deterministic state (GRU hidden)
    s_t = stochastic state (straight-through categorical)

    Args:
        hidden_dim:    Size of deterministic state h
        stoch_dim:     Number of categorical variables
        stoch_classes: Classes per categorical variable
        embed_dim:     Encoder embedding dimension
        action_dim:    Action space dimension
    """

    def __init__(
        self,
        hidden_dim:    int = 512,
        stoch_dim:     int = 32,
        stoch_classes: int = 32,
        embed_dim:     int = 512,
        action_dim:    int = 39,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.stoch_dim    = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_total  = stoch_dim * stoch_classes

        # Deterministic state: GRU
        self.gru = nn.GRUCell(
            input_size=self.stoch_total + action_dim,
            hidden_size=hidden_dim,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Posterior: q(s_t | h_t, e_t)  — uses observation embedding
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_total),
        )

        # Prior: p(s_t | h_t)  — no observation, used for imagination & KL
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_total),
        )

        self.embed_norm = nn.LayerNorm(embed_dim)

    # ── Initialisation ───────────────────────────────────────────────────

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_total, device=device)
        return h, s

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_stoch(self, logits: torch.Tensor, sample: bool = True):
        """
        Compute stochastic state from logits.
        Returns (s_flat, dist) where s_flat is straight-through if sample=True.
        """
        logits = logits.view(*logits.shape[:-1], self.stoch_dim, self.stoch_classes)
        dist   = OneHotCategorical(logits=logits)
        if sample:
            s = dist.sample()
            # Straight-through estimator: gradients flow through probs
            s = s + (dist.probs - dist.probs.detach())
        else:
            s = F.one_hot(logits.argmax(-1), self.stoch_classes).float()
        return s.view(*s.shape[:-2], self.stoch_total), dist

    # ── Single steps ─────────────────────────────────────────────────────

    def observe_step(self, h, s, action, embed):
        """
        One step with observation (training).

        All tensors must be 2-D: (B, dim).
        Returns: (h_new, s_new, prior_dist, post_dist)
        """
        x     = torch.cat([s, action], dim=-1)        # (B, stoch_total + action_dim)
        h_new = self.gru_norm(self.gru(x, h))         # (B, hidden_dim)

        embed = self.embed_norm(embed)
        post_logits = self.posterior_net(torch.cat([h_new, embed], dim=-1))
        s_new, post_dist = self._get_stoch(post_logits, sample=True)

        prior_logits = self.prior_net(h_new)
        _, prior_dist = self._get_stoch(prior_logits, sample=False)

        return h_new, s_new, prior_dist, post_dist

    def imagine_step(self, h, s, action):
        """
        One step without observation (rollout / imagination).
        Returns: (h_new, s_new)
        """
        x     = torch.cat([s, action], dim=-1)
        h_new = self.gru_norm(self.gru(x, h))
        prior_logits = self.prior_net(h_new)
        s_new, _     = self._get_stoch(prior_logits, sample=True)
        return h_new, s_new

    # ── Sequence processing ───────────────────────────────────────────────

    def observe_sequence(self, actions, embeds):
        """
        Process a full sequence during training.

        actions: (B, T, action_dim)
        embeds:  (B, T, embed_dim)
        Returns dict of stacked states for loss computation.
        """
        B, T, _ = actions.shape
        device  = actions.device
        h, s    = self.initial_state(B, device)

        hs, ss, prior_dists, post_dists = [], [], [], []

        for t in range(T):
            h, s, prior_d, post_d = self.observe_step(
                h, s, actions[:, t], embeds[:, t]
            )
            hs.append(h)
            ss.append(s)
            prior_dists.append(prior_d)
            post_dists.append(post_d)

        return {
            'h':     torch.stack(hs, dim=1),    # (B, T, hidden_dim)
            's':     torch.stack(ss, dim=1),    # (B, T, stoch_total)
            'prior': prior_dists,
            'post':  post_dists,
        }

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self.hidden_dim + self.stoch_total
