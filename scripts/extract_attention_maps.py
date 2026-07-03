"""
Extract per-head attention matrices from a real HF model and write them to
the JSONL schema `env/data.py::load_dataset` expects.

NOT RUNNABLE IN THIS SANDBOX -- torch/transformers aren't installed here
(no network access to fetch them or model weights). This is written against
the standard HF `output_attentions=True` API and should run as-is in a
normal GPU environment; if the original repo's own notebooks serialize
attention differently, treat that as the source of truth over this script's
assumptions and adjust `load_dataset` in env/data.py accordingly (that
function is deliberately the one seam to edit).

Usage:
    python scripts/extract_attention_maps.py \
        --model gpt2 --dataset tinystories --n-examples 200 \
        --out data/gpt2_attention.jsonl

Mirrors the paper's own choice of TinyStories as the extraction corpus
(section 2.3: "selected for its relative simplicity ... simplifies the act
of isolating specific head behaviors").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_for_model(model_name: str, sentences: list[str], out_path: Path,
                       max_length: int = 64, device: str = "cpu") -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, attn_implementation="eager"
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as f, torch.no_grad():
        for sentence in sentences:
            enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            n = len(tokens)
            if n < 3:
                continue

            outputs = model(**enc)
            # outputs.attentions: tuple of (n_layers) tensors, each
            # (batch=1, n_heads, seq, seq)
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                for head_idx in range(n_heads):
                    A = layer_attn[0, head_idx].cpu().numpy()  # (n, n)
                    head_id = f"{model_name}:L{layer_idx}H{head_idx}"
                    f.write(json.dumps({
                        "head_id": head_id,
                        "tokens": tokens,
                        "attention": A.tolist(),
                    }) + "\n")
                    n_written += 1
    print(f"wrote {n_written} (sentence x head) examples across "
          f"{n_layers} layers x {n_heads} heads to {out_path}")


def load_tinystories_sentences(n_examples: int, split: str = "train") -> list[str]:
    ds = hf_load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    sentences = []
    for ex in ds:
        text = ex["text"].strip()
        if text:
            sentences.append(text)
        if len(sentences) >= n_examples:
            break
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories"])
    ap.add_argument("--n-examples", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.dataset == "tinystories":
        sentences = load_tinystories_sentences(args.n_examples)
    else:
        raise ValueError(args.dataset)

    extract_for_model(args.model, sentences, Path(args.out), device=args.device)


if __name__ == "__main__":
    main()
