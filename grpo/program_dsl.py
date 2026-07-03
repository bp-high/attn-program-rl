"""
A tiny program-synthesis action space for the *self-contained* GRPO backend.

The real research setting (scripts/grpo_train.py, grpo/hf_policy.py) has a
code LM emit arbitrary Python `predict_attention` functions. That needs a
GPU and a capable policy model to produce *executable* code at all, which
makes it a poor smoke test. This module gives a from-scratch PyTorch policy
(grpo/toy_policy.py) something it can actually learn end-to-end on a laptop
in seconds, while exercising the *exact same* GRPO machinery, verifiable
reward, sandboxed executor, and IoU/JSD metrics as the LM path.

An action is a length-2 token sequence `(template_idx, sharpness_idx)`:
  - template_idx picks one of a handful of parametric attention motifs
    (previous-token, first-token, diagonal/self, two-back, sentence-start),
  - sharpness_idx picks how peaked the softmax over target positions is.
`compile_action` turns that into a genuine Python program string in the same
`predict_attention(tokens) -> np.ndarray` contract the executor and reward
expect -- so a learned toy action is scored by identical code to a learned
LM completion. This mirrors the paper's own "program library" framing, but
the retrieval/composition policy is *learned by RL* instead of fixed.
"""
from __future__ import annotations

import numpy as np

# Sharpness = logit added at the target position(s) over a zero baseline on
# all other causal positions; larger -> more peaked row-softmax. Chosen to
# straddle the synthetic archetypes' own peak strength (~5-6 in env/data.py),
# so the policy has a real parameter to tune, not a no-op dimension.
SHARPNESS_CHOICES = (2.0, 4.0, 6.0)

# Each template is (name, body) where `body` is inserted inside a
# `for i in range(n):` loop and may set logits[i, j] += SCALE. `{scale}` is
# substituted with the chosen sharpness value at compile time.
_TEMPLATES: list[tuple[str, str]] = [
    ("previous_token",
     "        logits[i, max(i - 1, 0)] += {scale}"),
    ("first_token",
     "        logits[i, 0] += {scale}"),
    ("diagonal_self",
     "        logits[i, i] += {scale}"),
    ("two_back",
     "        logits[i, max(i - 2, 0)] += {scale}"),
    ("sentence_boundary",
     "        for j in range(i + 1):\n"
     "            is_start = (j == 0) or (tokens[j - 1] in ('.', '!', '?'))\n"
     "            if is_start and tokens[j] not in ('.', '!', '?', ','):\n"
     "                logits[i, j] += {scale}"),
]

TEMPLATE_NAMES: list[str] = [name for name, _ in _TEMPLATES]
NUM_TEMPLATES: int = len(_TEMPLATES)
NUM_SHARPNESS: int = len(SHARPNESS_CHOICES)

_PROGRAM_SKELETON = '''\
import numpy as np

def predict_attention(tokens):
    """{name} (sharpness={scale})."""
    n = len(tokens)
    logits = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j > i:
                logits[i, j] = -1e9
    for i in range(n):
{body}
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)
'''


def compile_action(template_idx: int, sharpness_idx: int) -> str:
    """Map a (template_idx, sharpness_idx) action to a real Python program
    string in the `predict_attention(tokens)` contract."""
    name, body = _TEMPLATES[template_idx]
    scale = SHARPNESS_CHOICES[sharpness_idx]
    return _PROGRAM_SKELETON.format(name=name, scale=scale, body=body.format(scale=scale))


def action_str(template_idx: int, sharpness_idx: int) -> str:
    return f"{TEMPLATE_NAMES[template_idx]}@{SHARPNESS_CHOICES[sharpness_idx]:g}"


# --------------------------------------------------------------------------
# Policy-facing features
# --------------------------------------------------------------------------
# The LM policy is shown the top attention edges as text (scripts/build_prompt.py).
# The toy policy is shown a compact numeric summary of the *same* information:
# how much mass a head places on each positional motif. A head archetype maps
# to a distinctive feature vector, so a small MLP can learn template selection
# from reward alone. Kept deliberately motif-aligned (one feature per template
# family) rather than hand-mapping features to templates -- the mapping from
# features to the best action is what GRPO has to discover.

FEATURE_NAMES = [
    "mass_self", "mass_prev", "mass_two_back", "mass_first",
    "mass_sentence_start", "col_spread",
]
FEATURE_DIM = len(FEATURE_NAMES)


def attention_features(attention: np.ndarray, tokens: list[str]) -> np.ndarray:
    """Compact positional-motif summary of one attention matrix, in [0, 1]."""
    A = np.clip(np.asarray(attention, dtype=np.float64), 0.0, None)
    n = A.shape[0]
    rows = np.arange(n)
    mass_self = A[rows, rows].mean()
    mass_prev = A[rows, np.maximum(rows - 1, 0)].mean()
    mass_two_back = A[rows, np.maximum(rows - 2, 0)].mean()
    mass_first = A[:, 0].mean()

    is_start = np.zeros(n, dtype=bool)
    for j in range(n):
        is_start[j] = (j == 0) or (tokens[j - 1] in (".", "!", "?"))
    start_cols = np.where(is_start)[0]
    causal_start = np.zeros(n)
    for i in range(n):
        cols = start_cols[start_cols <= i]
        causal_start[i] = A[i, cols].sum() if len(cols) else 0.0
    mass_sentence_start = causal_start.mean()

    # Normalized entropy of the column marginal (the same behavioral signal
    # reward.positional_collapse_score uses). This is what separates a head
    # that dumps all mass on one column (first_token -> col_spread ~ 0) from
    # one that spreads across several sentence-start columns
    # (sentence_boundary -> col_spread well above 0), even though both put
    # high mass on token 0. Without it the two archetypes are near-degenerate
    # in feature space and the policy conflates them.
    col_marginal = A.mean(axis=0)
    total = col_marginal.sum()
    if total <= 1e-10 or n <= 1:
        col_spread = 0.0
    else:
        p = col_marginal / total
        entropy = -np.sum(p * np.log(p + 1e-12))
        col_spread = float(np.clip(entropy / np.log(n), 0.0, 1.0))

    return np.array([mass_self, mass_prev, mass_two_back, mass_first,
                     mass_sentence_start, col_spread], dtype=np.float32)


def mean_features(examples) -> np.ndarray:
    """Average feature vector over a head's examples (the policy input)."""
    feats = [attention_features(e.attention, e.tokens) for e in examples]
    return np.mean(feats, axis=0).astype(np.float32)
