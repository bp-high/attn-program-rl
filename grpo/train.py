"""
Orchestration glue shared by the CLI (train_grpo_torch.py) and the tests:
build the reward function from a set of head-prompts, and run a full toy
GRPO training session end to end.

Kept backend-agnostic where it can be: `make_reward_fn` and `HeadPrompt`
know nothing about the policy. `run_toy_training` wires the DSL policy,
verifiable reward, and grpo/core.py together and reports what the policy
learned to pick for each head.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from grpo.core import GRPOConfig, train_grpo
from grpo.rewarding import ProgramScorer


@dataclasses.dataclass
class HeadPrompt:
    head_id: str
    examples: list                       # examples used to compute the reward
    expected_template: int | None = None  # ground-truth template idx (toy eval only)


def make_reward_fn(prompts: list[HeadPrompt], scorer: ProgramScorer
                   ) -> Callable[[int, str], float]:
    def reward_fn(prompt_index: int, code: str) -> float:
        return scorer.mean_reward(code, prompts[prompt_index].examples)
    return reward_fn


def run_toy_training(archetypes: list[str] | None = None,
                     cfg: GRPOConfig | None = None,
                     n_examples: int = 6, hidden: int = 64,
                     multi_sentence: bool = True, device: str = "cpu",
                     verbose: bool = True) -> dict:
    """Train one DSL policy to pick the right program for each head archetype.

    Returns a dict with the metrics history, the trained policy, the prompts,
    the policy's final greedy action per head, and template-selection accuracy
    (fraction of heads for which the argmax template matches ground truth).
    """
    # Lazy import avoids a module-level cycle (toy_data imports HeadPrompt).
    from grpo.program_dsl import TEMPLATE_NAMES, action_str, mean_features
    from grpo.toy_data import build_toy_prompts
    from grpo.toy_policy import ToyProgramPolicy

    cfg = cfg or GRPOConfig()
    prompts = build_toy_prompts(archetypes, n_examples=n_examples,
                                multi_sentence=multi_sentence, seed=cfg.seed)
    features = np.stack([mean_features(p.examples) for p in prompts])

    policy = ToyProgramPolicy(features, hidden=hidden, device=device, seed=cfg.seed)
    scorer = ProgramScorer(trusted=True)
    reward_fn = make_reward_fn(prompts, scorer)

    def _log(m: dict) -> None:
        if verbose and (m["step"] % max(cfg.steps // 10, 1) == 0 or m["step"] == cfg.steps - 1):
            print(f"  step {m['step']:3d} | mean_reward {m['mean_reward']:+.3f} "
                  f"| max {m['max_reward']:+.3f} | exec {m['frac_executable']:.2f} "
                  f"| kl {m.get('kl', 0):.4f}")

    history = train_grpo(policy, num_prompts=len(prompts), reward_fn=reward_fn,
                         cfg=cfg, on_step=_log)

    greedy = policy.greedy_actions()
    correct = 0
    report = []
    for p, (t, s) in zip(prompts, greedy):
        ok = (p.expected_template is None) or (t == p.expected_template)
        correct += int(ok)
        report.append({
            "head_id": p.head_id, "chose": action_str(t, s),
            "expected": TEMPLATE_NAMES[p.expected_template] if p.expected_template is not None else None,
            "reward": reward_fn(prompts.index(p), _compiled(t, s)),
            "correct": ok,
        })
    accuracy = correct / len(prompts)
    if verbose:
        print(f"\n  template-selection accuracy: {accuracy:.0%} "
              f"({correct}/{len(prompts)} heads)")
        for r in report:
            mark = "OK " if r["correct"] else "XX "
            print(f"    {mark}{r['head_id']:28s} chose {r['chose']:22s} "
                  f"(reward {r['reward']:+.3f})")

    return {"history": history, "policy": policy, "prompts": prompts,
            "greedy": greedy, "accuracy": accuracy, "report": report}


def _compiled(t: int, s: int) -> str:
    from grpo.program_dsl import compile_action
    return compile_action(t, s)
