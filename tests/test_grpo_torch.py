"""
Tests for the from-scratch PyTorch GRPO trainer (grpo/).

Covers three layers, all offline (no model download, no network):
  - the program DSL compiles to executable programs and separates archetypes;
  - the GRPO primitives (group advantages, clipped loss) are correct;
  - the *toy* backend actually trains -- reward climbs and the policy learns
    the right program per head archetype;
  - the *HF causal-LM* backend runs end to end on a locally-built tiny GPT-2:
    the loop steps, log-probs are differentiable and wired to the optimizer,
    and old/new log-probs agree at inner-epoch 0 (importance ratio == 1).

Run: python -m pytest tests/test_grpo_torch.py -v
  or: python tests/test_grpo_torch.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from grpo.core import GRPOConfig, compute_group_advantages, grpo_loss, train_grpo
from grpo.program_dsl import (
    NUM_SHARPNESS, NUM_TEMPLATES, compile_action, mean_features,
)
from grpo.rewarding import ProgramScorer
from grpo.toy_data import build_toy_prompts


# --------------------------------------------------------------------------
# Program DSL
# --------------------------------------------------------------------------

def test_every_dsl_action_compiles_and_executes():
    from env import run_program_inproc
    tokens = "Tim was happy . He ran outside .".split(" ")
    for t in range(NUM_TEMPLATES):
        for s in range(NUM_SHARPNESS):
            A, ok, err = run_program_inproc(compile_action(t, s), tokens)
            assert ok, f"action ({t},{s}) failed: {err}"
            assert A.shape == (len(tokens), len(tokens))


def test_matching_template_wins_per_archetype():
    """The verifiable reward should rank a head's own template highest, so
    there is a real signal for the policy to learn."""
    scorer = ProgramScorer(trusted=True)
    for p in build_toy_prompts():
        rewards = {(t, s): scorer.mean_reward(compile_action(t, s), p.examples)
                   for t in range(NUM_TEMPLATES) for s in range(NUM_SHARPNESS)}
        best_t = max(rewards, key=rewards.get)[0]
        assert best_t == p.expected_template, (
            f"{p.head_id}: best template {best_t} != expected {p.expected_template}")


# --------------------------------------------------------------------------
# GRPO primitives
# --------------------------------------------------------------------------

def test_group_advantages_are_standardized_per_group():
    rewards = torch.tensor([0.0, 1.0, 2.0, 10.0, 10.0, 10.0])
    groups = [0, 0, 0, 1, 1, 1]
    adv = compute_group_advantages(rewards, groups, eps=1e-6)
    # group 0 standardized to ~zero mean, unit std; group 1 all-equal -> ~0
    assert abs(float(adv[:3].mean())) < 1e-5
    assert abs(float(adv[:3].std(unbiased=False)) - 1.0) < 1e-3
    assert torch.allclose(adv[3:], torch.zeros(3), atol=1e-4)


def test_grpo_loss_finite_and_entropy_reduces_loss():
    logps = [torch.tensor([-1.0, -2.0], requires_grad=True)]
    old = [torch.tensor([-1.0, -2.0])]
    adv = torch.tensor([1.5])
    kl = [torch.tensor([0.1, 0.1])]
    ent = [torch.tensor([0.5, 0.5])]
    loss_no_ent, _ = grpo_loss(logps, old, adv, kl, 0.2, 0.02, ent, ent_coef=0.0)
    loss_ent, stats = grpo_loss(logps, old, adv, kl, 0.2, 0.02, ent, ent_coef=0.1)
    assert torch.isfinite(loss_ent)
    assert float(loss_ent.detach()) < float(loss_no_ent.detach())  # entropy bonus lowers loss
    assert stats["entropy"] > 0


# --------------------------------------------------------------------------
# Toy backend: end-to-end learning
# --------------------------------------------------------------------------

def test_toy_grpo_training_converges():
    from grpo import run_toy_training
    cfg = GRPOConfig(steps=60, group_size=12, lr=3e-3, kl_coef=0.02, ent_coef=0.02, seed=0)
    out = run_toy_training(cfg=cfg, verbose=False)
    hist = out["history"]
    init, final = hist[0]["mean_reward"], hist[-1]["mean_reward"]
    assert final > init + 0.4, f"reward barely moved: {init:.3f} -> {final:.3f}"
    assert final > 0.5, f"final reward too low: {final:.3f}"
    assert out["accuracy"] >= 0.75, f"template accuracy too low: {out['accuracy']}"
    # every rollout executed (DSL programs are always well-formed)
    assert all(h["frac_executable"] == 1.0 for h in hist)


# --------------------------------------------------------------------------
# HF backend: runs offline on a tiny locally-built GPT-2
# --------------------------------------------------------------------------

def _tiny_hf_policy(prompts, max_new_tokens=16):
    transformers = pytest.importorskip("transformers")
    from grpo.hf_policy import CharTokenizer, HFPolicy
    torch.manual_seed(0)
    tok = CharTokenizer()
    cfg = transformers.GPT2Config(vocab_size=tok.vocab_size, n_positions=256,
                                  n_embd=32, n_layer=2, n_head=2)
    model = transformers.GPT2LMHeadModel(cfg)
    model.config.pad_token_id = tok.pad_token_id
    return HFPolicy(model, tok, prompts, device="cpu", max_new_tokens=max_new_tokens)


def test_hf_backend_train_loop_runs_offline():
    from grpo import HeadPrompt, make_reward_fn
    from env.data import synthetic_head_dataset
    policy = _tiny_hf_policy(["Write predict_attention for head A:"])
    head_prompts = [HeadPrompt("A", synthetic_head_dataset("previous_token", n_examples=3, noise=0.0))]
    reward_fn = make_reward_fn(head_prompts, ProgramScorer(trusted=False))
    hist = train_grpo(policy, 1, reward_fn, GRPOConfig(steps=2, group_size=4, seed=0))
    assert len(hist) == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in hist)


def test_hf_backend_is_differentiable_and_ratio_one_at_epoch0():
    policy = _tiny_hf_policy(["Write predict_attention:"])
    rollouts = policy.sample([0], 4)
    new_logps, kls, ent = policy.evaluate(rollouts)

    # (a) old == new log-probs at epoch 0 -> importance ratio is exactly 1
    for nl, r in zip(new_logps, rollouts):
        assert torch.allclose(nl, r.old_logps, atol=1e-5)

    # (b) with an injected non-zero advantage, a step actually moves params
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, _ = grpo_loss(new_logps, [r.old_logps for r in rollouts], adv, kls, 0.2, 0.02, ent, 0.0)
    assert torch.isfinite(loss)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in policy.parameters()]
    opt.zero_grad()
    loss.backward()
    opt.step()
    moved = sum(float((a - b).abs().sum()) for a, b in zip(policy.parameters(), before))
    assert moved > 0, "optimizer step did not change any parameters"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in fns:
        try:
            t()
            passed += 1
            print(f"PASS: {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(fns)}")
    if failed:
        sys.exit(1)
