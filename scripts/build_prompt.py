"""
Builds the synthesis prompt shown to the policy LM, reproducing the format
described in the paper (section 2.2):

  "our pipeline first extracts attention patterns, filtering for the top
  2.5% of attention weights by magnitude to isolate the most salient
  token-pair interactions. These filtered patterns are formatted as
  token-pair weight summaries and embedded into structured prompts
  (approximately 4,000 tokens in length). The synthesis agent S is
  provided access to NumPy, spaCy and NLTK."

Two entry points:
  - build_round0_prompt(example): the initial synthesis prompt, matching
    the paper's one-shot generation step.
  - build_refinement_prompt(example, prior_code, feedback): appends the
    previous attempt and the env's structured error feedback (worst-fit
    token pairs) to the prompt, extending the paper's *fixed one-round*
    refinement into an arbitrary-round MDP. Feed the full accumulated
    transcript as the "prompt" to a stock single-turn GRPO trainer -- see
    scripts/grpo_train.py for why that's the practical way to get
    multi-round refinement out of off-the-shelf TRL without a custom
    multi-turn rollout loop.
"""
from __future__ import annotations

from env.data import AttentionExample

SYSTEM_PREAMBLE = """You are analyzing a single attention head from a transformer language model. \
You will be shown its attention pattern on an example sentence: which tokens (queries) attend to \
which other tokens (keys), and how strongly. Your job is to write a Python function that \
approximates this attention pattern given only the input tokens -- no access to model weights.

Function signature (required, exact name):

    def predict_attention(tokens: list[str]) -> np.ndarray:
        n = len(tokens)
        attention = np.zeros((n, n))
        # attention[i, j] = how much query token i attends to key token j
        ...
        return attention

Constraints:
  - Only `np` (numpy) and `re` are available -- no other imports.
  - Causal heads should respect j <= i (queries cannot attend to future tokens); \
bidirectional heads may attend to any j.
  - Output does not need to be row-normalized; the harness will normalize for you.
  - Start with a one-line docstring naming the linguistic or positional pattern \
you believe this head implements.

Respond with ONLY the Python function, no explanation before or after."""


def _top_k_percent_edges(attention, tokens, top_pct: float = 2.5, max_edges: int = 40):
    """Filter for the top `top_pct`% of attention weights by magnitude,
    matching the paper's stated extraction step, capped at `max_edges` to
    keep the prompt within a practical token budget.
    """
    n = attention.shape[0]
    flat = attention.flatten()
    k = max(1, int(len(flat) * top_pct / 100))
    k = min(k, max_edges)
    top_idx = flat.argsort()[::-1][:k]
    edges = []
    for idx in top_idx:
        i, j = divmod(int(idx), n)
        edges.append((i, j, float(attention[i, j])))
    return edges


def _format_edges(edges, tokens) -> str:
    lines = []
    for i, j, w in edges:
        qi = tokens[i] if i < len(tokens) else "?"
        kj = tokens[j] if j < len(tokens) else "?"
        lines.append(f"  '{qi}'[{i}] -> '{kj}'[{j}]  ({w:.3f})")
    return "\n".join(lines)


def build_round0_prompt(example: AttentionExample, top_pct: float = 2.5) -> str:
    edges = _top_k_percent_edges(example.attention, example.tokens, top_pct=top_pct)
    token_str = " ".join(f"'{t}'[{i}]" for i, t in enumerate(example.tokens))
    edge_str = _format_edges(edges, example.tokens)
    return (
        f"{SYSTEM_PREAMBLE}\n\n"
        f"Head: {example.head_id}\n\n"
        f"Input tokens:\n  {token_str}\n\n"
        f"Top {top_pct}% attention edges by weight (query -> key):\n{edge_str}\n"
    )


def build_refinement_prompt(example: AttentionExample, prior_code: str, feedback: dict,
                             top_pct: float = 2.5) -> str:
    base = build_round0_prompt(example, top_pct=top_pct)
    if feedback.get("status") == "execution_failed":
        fb_str = f"Your previous attempt failed to execute: {feedback['error']}"
    else:
        worst = feedback.get("worst_edges", [])
        worst_str = "\n".join(
            f"  '{e['query']}'[{e['query_idx']}] -> '{e['key']}'[{e['key_idx']}]: "
            f"real={e['real']:.3f}, your program predicted={e['predicted']:.3f}"
            for e in worst
        )
        fb_str = (
            f"Your previous attempt scored IoU={feedback.get('iou', 0):.3f}, "
            f"JSD={feedback.get('jsd', 0):.3f}.\n"
            f"Worst-predicted token pairs (largest error vs. real attention):\n{worst_str}"
        )
    return (
        f"{base}\n\n"
        f"--- Your previous attempt ---\n{prior_code}\n\n"
        f"--- Feedback ---\n{fb_str}\n\n"
        f"Write a revised version of predict_attention that fixes these errors. "
        f"Respond with ONLY the revised Python function."
    )
