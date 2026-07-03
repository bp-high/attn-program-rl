"""
Attention-example data structures.

`load_dataset` reads the schema documented below, which you should point at
whatever your `scripts/extract_attention_maps.py` run produces for real
model heads. `synthetic_head_dataset` needs no model weights or network
access at all -- it fabricates known-ground-truth heads (previous-token,
first-token, sentence-boundary, diagonal) so you can unit-test the env,
reward shaping, and a synthesis agent's prompting before spending any API
budget or GPU time on the real thing. Treat it as a fixture, not a
replacement for real extracted attention maps.

Expected on-disk schema (one JSON object per line):
    {
      "head_id": "gpt2:L0H5",
      "tokens": ["And", " so", ",", " the", ...],
      "attention": [[...], [...], ...]   # n x n, row-stochastic
    }
Adjust `load_dataset` if the upstream repo's notebooks serialize attention
maps differently (e.g. .npz per head, or a HF Dataset) -- I couldn't fetch
that repo's directory contents at build time to confirm the exact format;
this is deliberately kept as the single seam to edit.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class AttentionExample:
    head_id: str
    tokens: list[str]
    attention: np.ndarray  # (n, n), row-stochastic


def load_dataset(path: str | Path) -> list[AttentionExample]:
    path = Path(path)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(AttentionExample(
                head_id=obj["head_id"],
                tokens=obj["tokens"],
                attention=np.asarray(obj["attention"], dtype=np.float64),
            ))
    return examples


def group_by_head(examples: list[AttentionExample]) -> dict[str, list[AttentionExample]]:
    """extract_attention_maps.py writes one flat JSONL spanning every head in
    a model; AttentionProgramEnv expects examples from a single head. Group
    once after loading, then build one env per head_id you care about.
    """
    grouped: dict[str, list[AttentionExample]] = {}
    for e in examples:
        grouped.setdefault(e.head_id, []).append(e)
    return grouped


def split_train_val(examples: list[AttentionExample], val_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[AttentionExample], list[AttentionExample]]:
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(examples) * val_frac))
    val_idx = set(idx[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    return train, val


# --------------------------------------------------------------------------
# Synthetic archetypes for testing (no model / network required)
# --------------------------------------------------------------------------

_SENTENCES = [
    "And so the daughter played on the slide and had a great time .",
    "Tim was happy because his mom gave him a new toy .",
    "The dog ran fast but it could not catch the cat .",
    "She asked her mom for a paper and a pen .",
    "Once upon a time there was a small red house .",
    "He looked at the sky and saw a bright star .",
]

# Multi-sentence inputs. On a single sentence, "attend to the sentence start"
# and "attend to token 0" are indistinguishable (both land on index 0) -- the
# degeneracy tests/test_env.py::_multi_sentence_example calls out. Passing
# multi_sentence=True to synthetic_head_dataset uses these instead so the
# sentence_boundary archetype is genuinely separable from first_token, which
# matters when training one policy to discriminate archetypes by their
# attention (see grpo/toy_data.py).
_MULTI_SENTENCES = [
    "Tim was happy . He ran outside and played with his dog .",
    "The dog barked loudly . A cat ran away fast . It was gone now .",
    "She smiled at him . Then she left the room . Nobody saw her go .",
    "Rain fell all day . The river rose high . People left their homes .",
    "He opened the box . Inside was a ring . His hands began to shake .",
    "Birds flew south . Winter came early . Snow covered the whole town .",
]


def _tokenize(sentence: str) -> list[str]:
    return sentence.split(" ")


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _previous_token_attention(tokens: list[str]) -> np.ndarray:
    n = len(tokens)
    logits = np.full((n, n), -5.0)
    for i in range(n):
        logits[i, max(i - 1, 0)] = 5.0
        logits[i, i] = 1.0
    return _row_softmax(logits)


def _first_token_attention(tokens: list[str]) -> np.ndarray:
    n = len(tokens)
    logits = np.full((n, n), -5.0)
    logits[:, 0] = 5.0
    return _row_softmax(logits)


def _sentence_boundary_attention(tokens: list[str]) -> np.ndarray:
    n = len(tokens)
    logits = np.full((n, n), -5.0)
    for i in range(n):
        for j in range(i + 1):
            is_sent_start = (j == 0) or (tokens[j - 1] in {".", "!", "?"})
            if is_sent_start and tokens[j] not in {".", "!", "?", ","}:
                logits[i, j] = 3.0
        logits[i, i] = 0.5
    return _row_softmax(logits)


def _diagonal_self_attention(tokens: list[str]) -> np.ndarray:
    n = len(tokens)
    logits = np.eye(n) * 6.0 - 5.0
    return _row_softmax(logits)


ARCHETYPES = {
    "previous_token": _previous_token_attention,
    "first_token": _first_token_attention,
    "sentence_boundary": _sentence_boundary_attention,
    "diagonal_self": _diagonal_self_attention,
}


def synthetic_head_dataset(kind: str, n_examples: int = 20, seed: int = 0,
                            noise: float = 0.02,
                            multi_sentence: bool = False) -> list[AttentionExample]:
    """Generate a fake but internally-consistent dataset for one head
    archetype, useful as ground truth to sanity-check the env/reward before
    touching real extracted attention maps.

    multi_sentence: draw from multi-sentence inputs so archetypes that only
        diverge across sentence boundaries (sentence_boundary vs first_token)
        are separable -- used by the toy GRPO convergence demo.
    """
    if kind not in ARCHETYPES:
        raise ValueError(f"unknown archetype {kind!r}, choose from {list(ARCHETYPES)}")
    fn = ARCHETYPES[kind]
    corpus = _MULTI_SENTENCES if multi_sentence else _SENTENCES
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n_examples):
        sentence = corpus[i % len(corpus)]
        tokens = _tokenize(sentence)
        attn = fn(tokens)
        if noise > 0:
            attn = attn + rng.normal(0, noise, attn.shape)
            attn = np.clip(attn, 1e-6, None)
            attn = attn / attn.sum(axis=1, keepdims=True)
        examples.append(AttentionExample(head_id=f"synthetic:{kind}", tokens=tokens, attention=attn))
    return examples
