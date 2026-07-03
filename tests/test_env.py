"""
Sanity tests against synthetic ground-truth heads. No torch, no model
weights, no network -- these should pass anywhere numpy is installed, and
are meant to be the first thing you run after cloning, before spending any
API budget on a real policy LM.

Run: python -m pytest tests/test_env.py -v
  or: python tests/test_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import (
    Action, AttentionProgramEnv, compute_reward, iou_score, jsd_score,
    synthetic_head_dataset,
)
from env.executor import run_program


PREVIOUS_TOKEN_PROGRAM = """
import numpy as np

def predict_attention(tokens):
    n = len(tokens)
    attention = np.zeros((n, n))
    for i in range(n):
        j = max(i - 1, 0)
        attention[i, j] = 0.92
        attention[i, i] += 0.08
    row_sums = attention.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return attention / row_sums
"""

FIRST_TOKEN_PROGRAM = """
import numpy as np

def predict_attention(tokens):
    n = len(tokens)
    attention = np.zeros((n, n))
    attention[:, 0] = 1.0
    return attention
"""

RANDOM_PROGRAM = """
import numpy as np

def predict_attention(tokens):
    n = len(tokens)
    rng = np.random.default_rng(42)
    m = rng.random((n, n))
    return m / m.sum(axis=1, keepdims=True)
"""

SYNTAX_ERROR_PROGRAM = "def predict_attention(tokens:\n    return None"

UNSAFE_PROGRAM = """
import os

def predict_attention(tokens):
    os.system("echo pwned")
    return None
"""

SENTENCE_BOUNDARY_PROGRAM = """
import numpy as np

def predict_attention(tokens):
    n = len(tokens)
    attention = np.zeros((n, n)) + 0.01
    for i in range(n):
        for j in range(i + 1):
            is_start = (j == 0) or (tokens[j - 1] in {'.', '!', '?'})
            if is_start and tokens[j] not in {'.', '!', '?', ','}:
                attention[i, j] += 3.0
        attention[i, i] += 0.5
    row_sums = attention.sum(axis=1, keepdims=True)
    return attention / row_sums
"""


def test_correct_previous_token_program_scores_high():
    data = synthetic_head_dataset("previous_token", n_examples=10, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.3)
    obs = env.reset(example=data[0])
    result = env.step(Action(code=PREVIOUS_TOKEN_PROGRAM))
    assert result.info["executable"], result.info["error"]
    assert result.info["iou"] > 0.75, f"expected high IoU, got {result.info['iou']}"
    print(f"[previous_token, correct program] reward={result.reward:.3f} "
          f"iou={result.info['iou']:.3f} jsd={result.info['jsd']:.3f} "
          f"complexity_penalty={result.info['complexity_penalty']:.3f}")


def test_wrong_archetype_scores_low():
    data = synthetic_head_dataset("previous_token", n_examples=10, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.3)
    env.reset(example=data[0])
    result = env.step(Action(code=FIRST_TOKEN_PROGRAM))
    print(f"[previous_token, first-token program] reward={result.reward:.3f} "
          f"iou={result.info['iou']:.3f}")
    assert result.info["iou"] < 0.5


def test_random_program_scores_lowest():
    data = synthetic_head_dataset("previous_token", n_examples=10, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.3)
    env.reset(example=data[0])
    correct = env.step(Action(code=PREVIOUS_TOKEN_PROGRAM))

    env.reset(example=data[0])
    rand = env.step(Action(code=RANDOM_PROGRAM))
    print(f"[previous_token] correct reward={correct.reward:.3f} vs "
          f"random reward={rand.reward:.3f}")
    assert correct.reward > rand.reward


def test_syntax_error_gets_floor_reward():
    data = synthetic_head_dataset("previous_token", n_examples=5, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.3)
    env.reset(example=data[0])
    result = env.step(Action(code=SYNTAX_ERROR_PROGRAM))
    assert not result.info["executable"]
    assert result.reward == -1.0
    print(f"[syntax error] reward={result.reward} (expected floor -1.0)")


def test_unsafe_program_is_rejected_not_executed():
    data = synthetic_head_dataset("previous_token", n_examples=5, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.3)
    env.reset(example=data[0])
    result = env.step(Action(code=UNSAFE_PROGRAM))
    assert not result.info["executable"]
    assert "unsafe" in (result.info["error"] or "").lower()
    print(f"[unsafe program] correctly rejected: {result.info['error']}")


def _multi_sentence_example():
    # Two sentences in one example -- deliberately breaks the degenerate
    # equivalence between "sentence boundary" and "first token" that holds
    # on any single-sentence input (both start-tokens coincide at index 0).
    from env.data import AttentionExample, _sentence_boundary_attention
    tokens = "Tim was happy . He ran outside and played with his dog .".split(" ")
    attn = _sentence_boundary_attention(tokens)
    return AttentionExample(head_id="synthetic:sentence_boundary_multi",
                             tokens=tokens, attention=attn)


def test_trivial_first_token_penalized_even_with_comparable_iou():
    """This is the paper's own flagged failure mode: a policy optimizing IoU
    alone can collapse onto degenerate first-token attenders. Uses a
    multi-sentence example where "attend to token 0" and "attend to sentence
    starts" genuinely diverge (unlike single-sentence inputs, where they're
    accidentally identical), then checks the content-invariance probe
    actually suppresses the trivial program relative to a structured one.
    """
    ex = _multi_sentence_example()
    env = AttentionProgramEnv([ex], max_rounds=1, val_fraction=0.99)

    env.reset(example=ex)
    trivial = env.step(Action(code=FIRST_TOKEN_PROGRAM))

    env.reset(example=ex)
    structured = env.step(Action(code=SENTENCE_BOUNDARY_PROGRAM))

    print(f"[multi-sentence] trivial first-token: iou={trivial.info['iou']:.3f} "
          f"reward={trivial.reward:.3f} | structured: iou={structured.info['iou']:.3f} "
          f"reward={structured.reward:.3f}")
    assert structured.info["iou"] > trivial.info["iou"], (
        "structured program should fit the two-sentence-start pattern "
        "better than a program that only ever looks at index 0"
    )
    assert structured.reward > trivial.reward


def test_multi_round_refinement_feedback_and_best_tracking():
    data = synthetic_head_dataset("previous_token", n_examples=10, noise=0.0)
    env = AttentionProgramEnv(data, max_rounds=3, val_fraction=0.3)
    obs = env.reset(example=data[0])
    assert obs.feedback is None and obs.round_idx == 0

    r1 = env.step(Action(code=RANDOM_PROGRAM))
    assert not r1.done
    assert r1.observation.feedback is not None
    assert r1.observation.feedback["status"] == "scored"
    assert "worst_edges" in r1.observation.feedback
    print(f"[round 1] feedback worst edge example: {r1.observation.feedback['worst_edges'][0]}")

    r2 = env.step(Action(code=PREVIOUS_TOKEN_PROGRAM))
    assert not r2.done
    assert r2.info["best_reward_so_far"] >= r1.reward

    r3 = env.step(Action(code=PREVIOUS_TOKEN_PROGRAM))
    assert r3.done
    assert r3.observation is None
    print(f"[round 3, terminal] best_reward_so_far={r3.info['best_reward_so_far']:.3f}")


def test_held_out_evaluation_generalizes():
    data = synthetic_head_dataset("previous_token", n_examples=20, noise=0.01)
    env = AttentionProgramEnv(data, max_rounds=1, val_fraction=0.4)
    stats = env.evaluate_held_out(PREVIOUS_TOKEN_PROGRAM)
    print(f"[held-out eval] mean_iou={stats['mean_iou']:.3f} "
          f"mean_jsd={stats['mean_jsd']:.3f} fail_rate={stats['fail_rate']:.2f}")
    assert stats["mean_iou"] > 0.7
    assert stats["fail_rate"] == 0.0


def test_iou_and_jsd_metrics_directly():
    import numpy as np
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(iou_score(A, B) - 1.0) < 1e-9
    assert jsd_score(A, B) < 1e-9

    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert iou_score(A, C) < 1e-9
    assert jsd_score(A, C) > 0.6  # near ln(2), fully disjoint distributions


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS: {t.__name__}\n")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}\n")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
