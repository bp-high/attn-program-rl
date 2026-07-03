"""From-scratch PyTorch GRPO for attention-program synthesis.

Public surface:
  - core:      GRPOConfig, Rollout, train_grpo, grpo_loss, compute_group_advantages
  - rewarding: ProgramScorer (verifiable IoU/JSD/degeneracy reward)
  - toy_policy: ToyProgramPolicy (self-contained DSL policy, no GPU/download)
  - train:     run_toy_training, make_reward_fn, HeadPrompt

The HF causal-LM policy lives in grpo.hf_policy and is imported lazily by
train_grpo_torch.py so `import grpo` never requires transformers.
"""
from grpo.core import (
    GRPOConfig, Policy, Rollout, compute_group_advantages, grpo_loss, train_grpo,
)
from grpo.rewarding import ProgramScorer
from grpo.train import HeadPrompt, make_reward_fn, run_toy_training

__all__ = [
    "GRPOConfig", "Policy", "Rollout", "compute_group_advantages", "grpo_loss",
    "train_grpo", "ProgramScorer", "HeadPrompt", "make_reward_fn", "run_toy_training",
]
