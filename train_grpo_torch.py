#!/usr/bin/env python3
"""
GRPO training entrypoint for attention-program synthesis, PyTorch-native and
from scratch (grpo/core.py) -- no TRL, no library trainer.

Two backends behind one loop:

  toy  self-contained. A small PyTorch policy (grpo/toy_policy.py) learns to
       pick, per synthetic head archetype, the program template that best
       reproduces its attention. Runs on CPU in seconds, no downloads. This
       is the fastest way to see the whole RL loop -- sampling, verifiable
       reward, group-relative advantage, clipped update, KL -- actually work
       and converge.

           python train_grpo_torch.py --backend toy

  hf   the real setting. A HuggingFace code LM writes arbitrary Python
       `predict_attention` functions; the env scores them. Point it at
       attention maps extracted by scripts/extract_attention_maps.py.

           python train_grpo_torch.py --backend hf \\
               --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
               --data data/gpt2_attention.jsonl --max-heads 8

Both call the identical grpo.core.train_grpo; only the Policy differs.
"""
from __future__ import annotations

import argparse
import json


def run_toy(args) -> None:
    from grpo import GRPOConfig, run_toy_training
    cfg = GRPOConfig(steps=args.steps, group_size=args.group_size, lr=args.lr,
                     kl_coef=args.kl_coef, ent_coef=args.ent_coef, seed=args.seed)
    archetypes = args.archetypes.split(",") if args.archetypes else None
    print(f"[toy] GRPO over {archetypes or 'default'} archetypes | "
          f"steps={cfg.steps} G={cfg.group_size} lr={cfg.lr} "
          f"kl={cfg.kl_coef} ent={cfg.ent_coef}\n")
    out = run_toy_training(archetypes=archetypes, cfg=cfg, device=args.device, verbose=True)
    if args.save_curve:
        with open(args.save_curve, "w") as f:
            json.dump(out["history"], f, indent=2)
        print(f"\nsaved reward curve to {args.save_curve}")


def run_hf(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from env import group_by_head, load_dataset
    from grpo import GRPOConfig, ProgramScorer, make_reward_fn, HeadPrompt, train_grpo
    from grpo.hf_policy import HFPolicy
    from scripts.build_prompt import build_round0_prompt

    grouped = group_by_head(load_dataset(args.data))
    head_ids = list(grouped)[: args.max_heads] if args.max_heads else list(grouped)
    prompts_txt, head_prompts = [], []
    for hid in head_ids:
        exs = grouped[hid]
        prompts_txt.append(build_round0_prompt(exs[0]))
        head_prompts.append(HeadPrompt(hid, exs))
    print(f"[hf] {len(head_ids)} heads from {args.data} | policy={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    policy = HFPolicy(model, tok, prompts_txt, device=args.device,
                      max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    scorer = ProgramScorer(trusted=False)  # untrusted LM code -> subprocess sandbox
    base_reward = make_reward_fn(head_prompts, scorer)
    best: dict[int, tuple[float, str]] = {}

    def reward_fn(pi: int, code: str) -> float:
        r = base_reward(pi, code)
        if pi not in best or r > best[pi][0]:
            best[pi] = (r, code)
        return r

    cfg = GRPOConfig(steps=args.steps, group_size=args.group_size, lr=args.lr,
                     kl_coef=args.kl_coef, seed=args.seed)
    train_grpo(policy, len(head_prompts), reward_fn, cfg,
               on_step=lambda m: print(f"  step {m['step']:3d} | mean_reward "
                                       f"{m['mean_reward']:+.3f} | exec {m['frac_executable']:.2f}"))

    print("\nbest program found per head:")
    for pi, hid in enumerate(head_ids):
        r, _ = best.get(pi, (float("nan"), ""))
        print(f"  {hid:24s} best_reward={r:+.3f}")
    if args.save_programs:
        dump = {head_ids[pi]: {"reward": best[pi][0], "code": best[pi][1]} for pi in best}
        with open(args.save_programs, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"saved best programs to {args.save_programs}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["toy", "hf"], default="toy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    # shared GRPO knobs
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--group-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    # toy
    ap.add_argument("--archetypes", default=None, help="comma-separated, e.g. previous_token,first_token")
    ap.add_argument("--save-curve", default=None, help="write per-step metrics JSON here")
    # hf
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default=None, help="JSONL from scripts/extract_attention_maps.py")
    ap.add_argument("--max-heads", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save-programs", default=None)
    args = ap.parse_args()

    if args.backend == "toy":
        run_toy(args)
    else:
        if not args.data:
            ap.error("--backend hf requires --data (run scripts/extract_attention_maps.py first)")
        # hf uses a smaller default lr than the toy policy
        if args.lr == 3e-3:
            args.lr = 1e-6
        run_hf(args)


if __name__ == "__main__":
    main()
