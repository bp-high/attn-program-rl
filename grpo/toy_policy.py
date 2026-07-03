"""
A self-contained PyTorch policy over the program DSL (grpo/program_dsl.py).

This is what makes the repo *runnable end-to-end with no GPU and no model
download*: a small autoregressive network that, conditioned on a head's
attention-motif features, emits a two-token program `(template, sharpness)`.
Sampling, per-token log-probs, and an exact categorical KL to a frozen
reference are all real -- they feed the same grpo/core.py optimizer step the
HF causal-LM policy uses. Training this against the verifiable env reward
visibly converges (the policy learns which program each head archetype
wants), which is the point: it demonstrates the full RL loop working, not a
stub.

`ToyProgramPolicy` is a plain wrapper (not an nn.Module) holding a trainable
`ToyNet` and a frozen reference `ToyNet`, so `parameters()` never leaks the
reference into the optimizer.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grpo.core import Rollout
from grpo.program_dsl import (
    FEATURE_DIM, NUM_SHARPNESS, NUM_TEMPLATES, compile_action,
)


class ToyNet(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 64,
                 template_emb: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.template_head = nn.Linear(hidden, NUM_TEMPLATES)
        self.template_emb = nn.Embedding(NUM_TEMPLATES, template_emb)
        self.sharp_head = nn.Linear(hidden + template_emb, NUM_SHARPNESS)

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.encoder(feats)

    def template_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.template_head(h)

    def sharp_logits(self, h: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        return self.sharp_head(torch.cat([h, self.template_emb(template)], dim=-1))


def _step_kl(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Exact per-example KL( current || reference ) for one categorical step."""
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)


class ToyProgramPolicy:
    """Policy protocol impl for the DSL backend."""

    def __init__(self, prompt_features: np.ndarray, hidden: int = 64,
                 device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.features = torch.tensor(np.asarray(prompt_features, dtype=np.float32),
                                     device=self.device)  # [P, F]
        self.net = ToyNet(feature_dim=self.features.shape[1], hidden=hidden).to(self.device)
        self.ref = copy.deepcopy(self.net).to(self.device)
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()

    def parameters(self):
        return self.net.parameters()

    # -- sampling (no grad; produces detached behavior-policy log-probs) --
    def sample(self, prompt_indices: list[int], group_size: int) -> list[Rollout]:
        flat = [pi for pi in prompt_indices for _ in range(group_size)]
        idx = torch.tensor(flat, device=self.device, dtype=torch.long)
        with torch.no_grad():
            h = self.net.encode(self.features[idx])                 # [R, H]
            t_dist = torch.distributions.Categorical(logits=self.net.template_logits(h))
            t = t_dist.sample()                                     # [R]
            lp_t = t_dist.log_prob(t)
            s_dist = torch.distributions.Categorical(logits=self.net.sharp_logits(h, t))
            s = s_dist.sample()                                     # [R]
            lp_s = s_dist.log_prob(s)

        rollouts = []
        for k, pi in enumerate(flat):
            ti, si = int(t[k]), int(s[k])
            old_logps = torch.stack([lp_t[k], lp_s[k]]).detach().cpu()
            rollouts.append(Rollout(prompt_index=pi, action=(ti, si),
                                    text=compile_action(ti, si), old_logps=old_logps))
        return rollouts

    # -- re-evaluation (with grad; recomputes log-probs, KL, entropy) --
    def evaluate(self, rollouts: list[Rollout]):
        idx = torch.tensor([r.prompt_index for r in rollouts], device=self.device, dtype=torch.long)
        t = torch.tensor([r.action[0] for r in rollouts], device=self.device, dtype=torch.long)
        s = torch.tensor([r.action[1] for r in rollouts], device=self.device, dtype=torch.long)

        h = self.net.encode(self.features[idx])
        t_logits = self.net.template_logits(h)
        s_logits = self.net.sharp_logits(h, t)
        lp_t = F.log_softmax(t_logits, dim=-1).gather(1, t[:, None]).squeeze(1)
        lp_s = F.log_softmax(s_logits, dim=-1).gather(1, s[:, None]).squeeze(1)

        with torch.no_grad():
            h_ref = self.ref.encode(self.features[idx])
            t_logits_ref = self.ref.template_logits(h_ref)
            s_logits_ref = self.ref.sharp_logits(h_ref, t)
        kl_t = _step_kl(t_logits, t_logits_ref)
        kl_s = _step_kl(s_logits, s_logits_ref)

        ent_t = torch.distributions.Categorical(logits=t_logits).entropy()
        ent_s = torch.distributions.Categorical(logits=s_logits).entropy()

        n = len(rollouts)
        new_logps = [torch.stack([lp_t[k], lp_s[k]]) for k in range(n)]
        kls = [torch.stack([kl_t[k], kl_s[k]]) for k in range(n)]
        entropies = [torch.stack([ent_t[k], ent_s[k]]) for k in range(n)]
        return new_logps, kls, entropies

    @torch.no_grad()
    def greedy_actions(self) -> list[tuple[int, int]]:
        """Argmax (template, sharpness) per prompt -- for reporting what the
        policy has learned to pick for each head."""
        h = self.net.encode(self.features)
        t = self.net.template_logits(h).argmax(dim=-1)
        s = self.net.sharp_logits(h, t).argmax(dim=-1)
        return [(int(t[i]), int(s[i])) for i in range(self.features.shape[0])]
