"""
Reward functions for RL-based attention-program synthesis.

Implements the two metrics from Hayes, Li & Andreas (2026), "Explaining
Attention with Program Synthesis" (arXiv:2606.19317):

  - IoU(A, Ahat)  = sum(min(A,Ahat)) / sum(max(A,Ahat))          [paper eq. 3]
  - JSD(A, Ahat)  = 0.5*KL(A||M) + 0.5*KL(Ahat||M), M=(A+Ahat)/2  [paper eq. 2]

plus a complexity penalty that the paper's pipeline does *not* have, and
which is the main thing this environment adds. The paper's own limitations
section names two failure modes we're explicitly guarding against:

  1. "many high-scoring programs are not particularly complex ... the
     improvement in QA seen ... at low replacement levels may be akin to
     a pruning effect" -> a policy optimizing IoU alone can collapse onto
     degenerate solutions (e.g. "always attend to token 0") that fit the
     coarse metric without capturing real head logic.
  2. Non-well-formed / non-executable programs are assigned maximal
     divergence (the paper's rule) -> we mirror that as a hard reward floor.

Reward = w_iou * IoU - w_jsd * JSD - w_cx * complexity_penalty(program)
with a hard floor for syntax/execution failures.
"""
from __future__ import annotations

import ast
import dataclasses
import numpy as np


EPS = 1e-10


# --------------------------------------------------------------------------
# Core similarity metrics (paper-faithful)
# --------------------------------------------------------------------------

def _as_nonneg_matrix(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    M = np.clip(M, 0.0, None)
    return M


def iou_score(A: np.ndarray, Ahat: np.ndarray) -> float:
    """Intersection-over-Union between two nonnegative attention matrices.

    Paper eq. 3: IoU(A, Ahat) = sum_ij min(A_ij, Ahat_ij) / sum_ij max(A_ij, Ahat_ij)
    """
    A = _as_nonneg_matrix(A)
    Ahat = _as_nonneg_matrix(Ahat)
    if A.shape != Ahat.shape:
        return 0.0
    inter = np.minimum(A, Ahat).sum()
    union = np.maximum(A, Ahat).sum()
    if union < EPS:
        return 1.0 if inter < EPS else 0.0
    return float(inter / union)


def _normalize_to_distribution(M: np.ndarray) -> np.ndarray:
    M = _as_nonneg_matrix(M)
    total = M.sum()
    if total < EPS:
        # degenerate -> uniform distribution, treated as maximally uninformative
        return np.full_like(M, 1.0 / M.size)
    return M / total


def jsd_score(A: np.ndarray, Ahat: np.ndarray) -> float:
    """Jensen-Shannon distance between two attention matrices treated as
    flattened joint distributions over token pairs.

    Paper eq. 2: JSD(A, Ahat) = 0.5*KL(A||M) + 0.5*KL(Ahat||M), M = (A+Ahat)/2
    Returns the JS *divergence* (bounded in [0, ln 2]); callers that want the
    JS *distance* (bounded in [0, 1]) should take sqrt().
    """
    if A.shape != Ahat.shape:
        return float(np.log(2.0))  # maximal divergence for shape mismatch
    P = _normalize_to_distribution(A)
    Q = _normalize_to_distribution(Ahat)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * (np.log(P + EPS) - np.log(M + EPS)))
    kl_qm = np.sum(Q * (np.log(Q + EPS) - np.log(M + EPS)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(max(jsd, 0.0))


# --------------------------------------------------------------------------
# Complexity penalty
# --------------------------------------------------------------------------

# AST node types that count as "real" logic vs. boilerplate. Kept small and
# legible on purpose -- this is a knob you should tune per experiment.
_LOGIC_NODES = (
    ast.If, ast.For, ast.While, ast.BoolOp, ast.Compare,
    ast.FunctionDef, ast.Call, ast.ListComp, ast.DictComp,
    ast.IfExp, ast.Lambda,
)


@dataclasses.dataclass
class ComplexityReport:
    node_count: int
    logic_node_count: int
    is_trivial_constant: bool
    collapse_score: float  # 0 = attention spread across columns, 1 = collapsed onto one fixed column
    penalty: float


def positional_collapse_score(Ahat: np.ndarray) -> float:
    """Intrinsic degeneracy signal computed directly from a candidate's own
    output -- no ground truth, no second execution needed.

    A program that ignores its input and always routes attention to one
    fixed column (e.g. `attention[:, 0] = 1.0`) produces a column-marginal
    distribution (mean attention received by each column, across all query
    rows) that's a spike. A program with real positional or content logic
    -- even a *purely* positional one like "attend to token i-1", which is
    legitimate and shouldn't be penalized -- spreads attention across many
    columns as the query index moves, so its column marginal is closer to
    uniform. We measure this via normalized Shannon entropy of the column
    marginal: 1 - H(marginal)/log(n).

    Verified against the four synthetic archetypes in env/data.py:
    previous-token and diagonal-self (both purely positional, both
    legitimate) score ~0.04 and ~0.00 collapse (i.e. NOT flagged); a
    constant first-token program scores ~0.998 collapse (correctly
    flagged). This is deliberately behavioral rather than AST-based: no
    static pattern-match on the source catches `attention[:, 0] = 1.0`,
    since it has no conditionals to match on at all.
    """
    n = Ahat.shape[0]
    if n <= 1:
        return 0.0
    col_marginal = np.clip(Ahat, 0, None).mean(axis=0)
    total = col_marginal.sum()
    if total < EPS:
        return 1.0  # all-zero output is maximally degenerate
    col_marginal = col_marginal / total
    entropy = -np.sum(col_marginal * np.log(col_marginal + EPS))
    max_entropy = np.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > EPS else 1.0
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


def analyze_complexity(code: str, min_nodes: int = 6, target_nodes: int = 40,
                        max_nodes: int = 220,
                        collapse_score: float = 0.0) -> ComplexityReport:
    """Score a candidate program's complexity and structural triviality.

    Penalizes two failure modes:
      - too trivial (below `min_nodes`) or too complex (above `max_nodes`,
        shallow U around `target_nodes`) -- discourages both no-op programs
        and unreadable ones that overfit rather than describe general logic.
      - positional collapse (see `positional_collapse_score`): this is
        precisely the "pruning effect" confound the paper's own limitations
        section flags -- high-IoU programs that fit the metric by ignoring
        the query entirely rather than capturing real head logic. Pass the
        candidate's own output through `positional_collapse_score` and feed
        it in here; this function alone can't compute it from source.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ComplexityReport(0, 0, True, 1.0, penalty=1.0)

    node_count = sum(1 for _ in ast.walk(tree))
    logic_count = sum(1 for n in ast.walk(tree) if isinstance(n, _LOGIC_NODES))
    is_trivial_constant = node_count < min_nodes

    if node_count <= min_nodes:
        penalty = 1.0
    elif node_count >= max_nodes:
        overshoot = (node_count - max_nodes) / max(target_nodes, 1)
        penalty = min(1.0, 0.3 + 0.1 * overshoot)
    else:
        # shallow U, 0 at target, rising toward both edges, capped well below 1
        span = max(target_nodes - min_nodes, max_nodes - target_nodes, 1)
        dist = abs(node_count - target_nodes) / span
        penalty = min(0.35, 0.35 * dist)

    # collapse_score in [0,1] scales an additional penalty up to 0.6 at full
    # collapse -- large enough to flip the ranking against a comparably-fit
    # program that actually varies its attention with the query.
    penalty = max(penalty, 0.6 * collapse_score)

    return ComplexityReport(node_count, logic_count, is_trivial_constant,
                             collapse_score, penalty=penalty)


# --------------------------------------------------------------------------
# Combined reward
# --------------------------------------------------------------------------

@dataclasses.dataclass
class RewardBreakdown:
    reward: float
    iou: float
    jsd: float
    complexity_penalty: float
    executable: bool
    error: str | None = None


DEFAULT_WEIGHTS = dict(w_iou=1.0, w_jsd=0.5, w_cx=0.3)
FAILURE_REWARD = -1.0  # paper's "non-well-formed -> maximal divergence" rule


def compute_reward(A: np.ndarray | None, Ahat: np.ndarray | None, code: str,
                    executable: bool, error: str | None = None,
                    weights: dict | None = None,
                    collapse_score: float = 0.0) -> RewardBreakdown:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    if not executable or Ahat is None:
        return RewardBreakdown(FAILURE_REWARD, 0.0, float(np.log(2.0)), 1.0,
                                executable=False, error=error)

    iou = iou_score(A, Ahat)
    jsd = jsd_score(A, Ahat)
    cx = analyze_complexity(code, collapse_score=collapse_score).penalty

    reward = w["w_iou"] * iou - w["w_jsd"] * jsd - w["w_cx"] * cx
    return RewardBreakdown(reward=float(reward), iou=iou, jsd=jsd,
                            complexity_penalty=cx, executable=True, error=None)
