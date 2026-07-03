"""
GRPO (Group Relative Policy Optimization) from scratch in PyTorch.

This is the "not scaffolded" core: the algorithm of Shao et al. (2024,
DeepSeekMath) implemented directly, not delegated to a library trainer. It
is deliberately backend-agnostic -- it knows nothing about attention heads,
programs, tokenizers, or transformers. It talks to a `Policy` (below) that
can (a) sample grouped rollouts and (b) recompute per-token log-probs and a
KL-to-reference term with gradient. Both the self-contained toy policy
(grpo/toy_policy.py) and the HF causal-LM policy (grpo/hf_policy.py) satisfy
that interface, so the exact same optimizer step trains both.

GRPO in one paragraph: for each prompt, sample a *group* of G completions;
score each with a verifiable reward; use the group's mean/std as a baseline
so the advantage of completion i is (r_i - mean)/std -- no learned value
network. Then take the standard PPO-style clipped surrogate on the
per-token importance ratio, plus a KL penalty toward a frozen reference
policy. Here the reward is the attention-program env's IoU/JSD/degeneracy
score (env/reward.py), which is verifiable and not LM-judged.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Protocol

import torch


@dataclasses.dataclass
class Rollout:
    """One sampled completion and everything needed to score and reweight it."""
    prompt_index: int          # which prompt/head this came from (groups share this)
    action: object             # backend-specific (toy: (template, sharpness); hf: token ids)
    text: str                  # the program string to execute for reward
    old_logps: torch.Tensor    # [T] per-token log-prob under the behavior policy (detached)
    reward: float = 0.0        # filled in by the driver via reward_fn


class Policy(Protocol):
    """The only surface grpo/core.py needs from a policy backend."""

    def sample(self, prompt_indices: list[int], group_size: int) -> list[Rollout]:
        """Return len(prompt_indices) * group_size rollouts, each tagged with
        its originating prompt_index. `old_logps` must be detached."""
        ...

    def evaluate(self, rollouts: list[Rollout]
                 ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
        """Recompute, WITH gradient, per-token log-probs, per-token KL to the
        frozen reference policy, and (optionally) per-token policy entropy, in
        the same token order as `old_logps`. Returns
        (new_logps, kl, entropies); entropies may be None if the backend
        does not provide an entropy bonus (e.g. large-vocab LM)."""
        ...

    def parameters(self):
        ...


@dataclasses.dataclass
class GRPOConfig:
    steps: int = 80
    group_size: int = 12         # G: completions sampled per prompt
    inner_epochs: int = 1        # PPO-style reuse of each batch of rollouts
    lr: float = 3e-3
    clip_eps: float = 0.2        # PPO clip range on the importance ratio
    kl_coef: float = 0.02        # weight on KL-to-reference
    ent_coef: float = 0.01       # entropy bonus; guards against exploration collapse
    adv_eps: float = 1e-4        # std floor when standardizing group advantages
    max_grad_norm: float = 1.0
    seed: int = 0


# --------------------------------------------------------------------------
# Core algorithm
# --------------------------------------------------------------------------

def compute_group_advantages(rewards: torch.Tensor, group_ids: list[int],
                             eps: float = 1e-4) -> torch.Tensor:
    """GRPO advantage: standardize each prompt-group's rewards to (r-mean)/std.

    The group mean is the value baseline (no critic); the group std is a
    per-prompt scale. Groups where every completion scored identically get
    zero advantage (nothing to learn from that prompt this step)."""
    adv = torch.zeros_like(rewards)
    groups: dict[int, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idxs in groups.values():
        r = rewards[idxs]
        std = r.std(unbiased=False)
        adv[idxs] = (r - r.mean()) / (std + eps)
    return adv


def grpo_loss(new_logps: list[torch.Tensor], old_logps: list[torch.Tensor],
              advantages: torch.Tensor, kls: list[torch.Tensor],
              clip_eps: float, kl_coef: float,
              entropies: list[torch.Tensor] | None = None,
              ent_coef: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Token-mean PPO-clipped surrogate + KL penalty - entropy bonus.

    For rollout i with scalar advantage A_i and per-token ratio
    r_t = exp(logp_new - logp_old):
        L_t = -min(r_t*A_i, clip(r_t, 1-eps, 1+eps)*A_i) + kl_coef*KL_t - ent_coef*H_t
    averaged over all tokens in the batch. Clipping caps how far one update
    can move the policy toward a high-advantage completion; the KL term
    anchors it to the frozen reference so it doesn't drift into reward hacking
    the degeneracy penalty; the entropy bonus keeps sampling diverse so the
    policy doesn't prematurely collapse onto a locally-good program before
    discovering a better one."""
    # Align everything to the policy's compute device: a GPU policy returns
    # new_logps/kls on cuda, while old_logps (cached in sample()) and the
    # advantages (built from CPU rewards) live on CPU.
    device = new_logps[0].device if new_logps else advantages.device
    advantages = advantages.to(device)
    pol_sum = torch.zeros((), device=device)
    kl_sum = torch.zeros((), device=device)
    ent_sum = torch.zeros((), device=device)
    n_tokens = 0
    for i, (lp_new, lp_old, adv_i, kl_i) in enumerate(zip(new_logps, old_logps, advantages, kls)):
        lp_old = lp_old.to(device)
        kl_i = kl_i.to(device)
        ratio = torch.exp(lp_new - lp_old)                 # [T]
        unclipped = ratio * adv_i
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
        pol_sum = pol_sum - torch.min(unclipped, clipped).sum()
        kl_sum = kl_sum + kl_i.sum()
        if entropies is not None and entropies[i] is not None:
            ent_sum = ent_sum + entropies[i].to(device).sum()
        n_tokens += lp_new.numel()
    n_tokens = max(n_tokens, 1)
    loss = (pol_sum + kl_coef * kl_sum - ent_coef * ent_sum) / n_tokens
    stats = {
        "policy_loss": float(pol_sum.detach() / n_tokens),
        "kl": float(kl_sum.detach() / n_tokens),
        "entropy": float(ent_sum.detach() / n_tokens),
    }
    return loss, stats


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

RewardFn = Callable[[int, str], float]


def train_grpo(policy: Policy, num_prompts: int, reward_fn: RewardFn,
               cfg: GRPOConfig, on_step: Callable[[dict], None] | None = None
               ) -> list[dict]:
    """Run GRPO for `cfg.steps` steps over `num_prompts` prompts.

    reward_fn(prompt_index, program_text) -> float is the verifiable reward
    (env/reward.py). on_step, if given, receives a per-step metrics dict --
    used by callers to print reward curves or track per-prompt best programs.
    Returns the list of per-step metrics dicts.
    """
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    prompt_indices = list(range(num_prompts))
    history: list[dict] = []

    for step in range(cfg.steps):
        rollouts = policy.sample(prompt_indices, cfg.group_size)
        for r in rollouts:
            r.reward = reward_fn(r.prompt_index, r.text)

        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
        group_ids = [r.prompt_index for r in rollouts]
        advantages = compute_group_advantages(rewards, group_ids, cfg.adv_eps)

        last_stats: dict = {}
        for _ in range(cfg.inner_epochs):
            new_logps, kls, entropies = policy.evaluate(rollouts)
            loss, last_stats = grpo_loss(
                new_logps, [r.old_logps for r in rollouts], advantages, kls,
                cfg.clip_eps, cfg.kl_coef, entropies, cfg.ent_coef,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()), cfg.max_grad_norm)
            opt.step()

        metrics = {
            "step": step,
            "mean_reward": float(rewards.mean()),
            "max_reward": float(rewards.max()),
            "frac_executable": float((rewards > -1.0).float().mean()),
            "loss": float(loss.detach()),
            **last_stats,
        }
        history.append(metrics)
        if on_step is not None:
            on_step(metrics)
    return history
