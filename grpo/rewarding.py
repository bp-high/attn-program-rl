"""
Turns a program string into a scalar GRPO reward, fast.

Wraps env/reward.py's paper-faithful IoU/JSD + degeneracy penalty. The only
thing added here is a compile cache: an RL step scores G-completions x
prompts x steps programs, and for the toy DSL those collapse to a handful of
distinct source strings, so we compile each once (in-process, trusted path)
and reuse the callable. The reward *value* is identical to what
AttentionProgramEnv.step would compute; this just skips the per-call spawn
subprocess that isolates untrusted policy-LM output.
"""
from __future__ import annotations

import numpy as np

from env.executor import compile_program, run_program, run_program_inproc
from env.reward import FAILURE_REWARD, compute_reward, positional_collapse_score


class ProgramScorer:
    def __init__(self, weights: dict | None = None, trusted: bool = True):
        """trusted=True executes candidates in-process (fast; use for the
        template-generated toy DSL). trusted=False routes through the
        subprocess sandbox with a timeout (use for untrusted policy-LM code).
        """
        self.weights = weights
        self.trusted = trusted
        self._fn_cache: dict = {}

    def _run(self, code: str, tokens: list[str]):
        if not self.trusted:
            return run_program(code, tokens)
        cached = self._fn_cache.get(code)
        if cached is None:
            try:
                cached = compile_program(code)
            except Exception as e:  # noqa: BLE001 - static-check / syntax failures
                cached = e
            self._fn_cache[code] = cached
        if isinstance(cached, Exception):
            return None, False, f"compile failed: {cached}"
        try:
            from env.executor import _postprocess
            return _postprocess(cached(tokens), len(tokens))
        except Exception as e:  # noqa: BLE001 - runtime failures in candidate
            return None, False, f"runtime error: {e}"

    def score_example(self, code: str, example) -> float:
        A_hat, executable, error = self._run(code, example.tokens)
        collapse = positional_collapse_score(A_hat) if executable else 0.0
        return compute_reward(example.attention, A_hat, code, executable, error,
                              self.weights, collapse_score=collapse).reward

    def mean_reward(self, code: str, examples: list) -> float:
        """Reward averaged over a head's examples -- a lower-variance RL
        signal than scoring a single sampled sentence, and it rewards
        programs that describe the head's *general* rule, not one input."""
        if not examples:
            return FAILURE_REWARD
        return float(np.mean([self.score_example(code, e) for e in examples]))
