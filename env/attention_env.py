"""
AttentionProgramEnv: an OpenEnv-style RL environment where the policy
writes Python programs that approximate a target attention head, and the
environment scores them with the paper's own metrics (IoU / JSD) plus a
complexity penalty, with verifiable, non-LM-judged rewards.

This directly targets the paper's stated gap: they use one-shot generation
plus a single refinement round and explicitly say closing the fit gap
"likely requires richer synthesis strategies such as multi-round refinement
with stronger feedback signals." This env generalizes their fixed
one-round refinement into an N-round MDP so a policy can be trained with
GRPO (or any group-relative RL method) instead of relying on a single
prompted refinement pass.

API mirrors OpenEnv's Environment / Observation / StepResult conventions:
    obs = env.reset()
    ...
    step = env.step(Action(code=candidate_program))
    step.observation, step.reward, step.done, step.info
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .data import AttentionExample
from .executor import run_program
from .reward import RewardBreakdown, compute_reward, positional_collapse_score


@dataclasses.dataclass
class Observation:
    head_id: str
    tokens: list[str]
    round_idx: int
    max_rounds: int
    # Feedback from the previous round, mirroring the paper's refinement
    # step: "identify representative best and worst-scoring examples ...
    # constructing structured error feedback by contrasting real and
    # predicted attention patterns". None on round 0.
    feedback: Optional[dict] = None
    # NOTE: target attention is intentionally NOT included here. The policy
    # only ever sees `tokens` (matching the paper's X -> A signature) plus
    # the top-k edge summary used to build the synthesis prompt upstream
    # (see scripts/build_prompt.py). Held-out attention is only used by the
    # env internally to score submissions.


@dataclasses.dataclass
class Action:
    code: str


@dataclasses.dataclass
class StepResult:
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict


class AttentionProgramEnv:
    def __init__(self, examples: list[AttentionExample], max_rounds: int = 3,
                 reward_weights: Optional[dict] = None, timeout: float = 5.0,
                 val_fraction: float = 0.3, seed: int = 0):
        """
        examples: all examples for ONE head (same head_id), used as both the
            fitting set (shown to the policy across rounds) and the held-out
            set (used only for scoring, never revealed as raw numbers).
        max_rounds: generalizes the paper's fixed single refinement round
            into a configurable budget.
        val_fraction: fraction of `examples` withheld purely for scoring,
            so a policy can't just memorize per-example feedback and must
            infer the underlying rule -- this matches the paper's own
            train/held-out split for selecting pi*.
        """
        if len({e.head_id for e in examples}) != 1:
            raise ValueError("AttentionProgramEnv expects examples from a single head")
        self.head_id = examples[0].head_id
        self.max_rounds = max_rounds
        self.reward_weights = reward_weights
        self.timeout = timeout

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(examples))
        n_val = max(1, int(len(examples) * val_fraction))
        val_idx = set(idx[:n_val].tolist())
        self.fit_examples = [e for i, e in enumerate(examples) if i not in val_idx]
        self.val_examples = [e for i, e in enumerate(examples) if i in val_idx]
        if not self.fit_examples:
            self.fit_examples = examples
        if not self.val_examples:
            self.val_examples = examples

        self._round = 0
        self._current_example: Optional[AttentionExample] = None
        self._best_reward = -float("inf")
        self._best_code: Optional[str] = None

    def reset(self, example: Optional[AttentionExample] = None) -> Observation:
        self._round = 0
        self._current_example = example or self.fit_examples[
            np.random.randint(len(self.fit_examples))
        ]
        self._best_reward = -float("inf")
        self._best_code = None
        return Observation(
            head_id=self.head_id,
            tokens=self._current_example.tokens,
            round_idx=0,
            max_rounds=self.max_rounds,
            feedback=None,
        )

    def step(self, action: Action) -> StepResult:
        if self._current_example is None:
            raise RuntimeError("call reset() before step()")

        ex = self._current_example
        Ahat, executable, error = run_program(action.code, ex.tokens, timeout=self.timeout)
        collapse = positional_collapse_score(Ahat) if executable else 0.0
        breakdown: RewardBreakdown = compute_reward(
            ex.attention, Ahat, action.code, executable, error, self.reward_weights,
            collapse_score=collapse,
        )

        if breakdown.reward > self._best_reward:
            self._best_reward = breakdown.reward
            self._best_code = action.code

        self._round += 1
        done = self._round >= self.max_rounds

        feedback = None
        if not done:
            feedback = self._build_feedback(ex, Ahat, breakdown)

        obs = None
        if not done:
            obs = Observation(
                head_id=self.head_id,
                tokens=ex.tokens,
                round_idx=self._round,
                max_rounds=self.max_rounds,
                feedback=feedback,
            )

        info = {
            "iou": breakdown.iou,
            "jsd": breakdown.jsd,
            "complexity_penalty": breakdown.complexity_penalty,
            "executable": breakdown.executable,
            "error": breakdown.error,
            "best_reward_so_far": self._best_reward,
        }
        return StepResult(observation=obs, reward=breakdown.reward, done=done, info=info)

    def evaluate_held_out(self, code: str) -> dict:
        """Score `code` on the held-out split. This is what should decide
        pi* across a group of GRPO rollouts, not the in-loop training
        reward, to avoid rewarding overfit to the fit examples shown during
        refinement rounds.
        """
        ious, jsds = [], []
        n_fail = 0
        for ex in self.val_examples:
            Ahat, executable, error = run_program(code, ex.tokens, timeout=self.timeout)
            if not executable:
                n_fail += 1
                continue
            r = compute_reward(ex.attention, Ahat, code, executable, error, self.reward_weights)
            ious.append(r.iou)
            jsds.append(r.jsd)
        return {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "mean_jsd": float(np.mean(jsds)) if jsds else float(np.log(2.0)),
            "fail_rate": n_fail / max(len(self.val_examples), 1),
            "n_scored": len(ious),
        }

    @staticmethod
    def _build_feedback(ex: AttentionExample, Ahat: Optional[np.ndarray],
                         breakdown: RewardBreakdown, top_k: int = 6) -> dict:
        """Structured error feedback contrasting real vs. predicted attention
        on the worst-fit token pairs, mirroring the paper's refinement step
        (section 2.2). If the program failed to execute, feedback is just
        the error message -- that alone is often enough signal to fix a
        shape/naming bug on the next round.
        """
        if Ahat is None:
            return {"status": "execution_failed", "error": breakdown.error}

        diff = np.abs(ex.attention - Ahat)
        n = diff.shape[0]
        flat_idx = np.argsort(diff, axis=None)[::-1][:top_k]
        worst_edges = []
        for fi in flat_idx:
            i, j = np.unravel_index(fi, diff.shape)
            worst_edges.append({
                "query": ex.tokens[i], "query_idx": int(i),
                "key": ex.tokens[j], "key_idx": int(j),
                "real": round(float(ex.attention[i, j]), 4),
                "predicted": round(float(Ahat[i, j]), 4),
            })
        return {
            "status": "scored",
            "iou": round(breakdown.iou, 4),
            "jsd": round(breakdown.jsd, 4),
            "complexity_penalty": round(breakdown.complexity_penalty, 4),
            "worst_edges": worst_edges,
        }
