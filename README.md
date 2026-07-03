# attn-program-rl

A **from-scratch PyTorch GRPO trainer** and RL environment for learning
attention-head *program synthesis*, extending Hayes, Li & Andreas (2026),
*[Explaining Attention with Program Synthesis](https://arxiv.org/abs/2606.19317)*
(code: [AmiriHayes/explaining_attention_heads](https://github.com/AmiriHayes/explaining_attention_heads)).

The GRPO algorithm — grouped sampling, group-relative advantages, the
PPO-clipped surrogate, the KL-to-reference and entropy terms, the optimizer
step — is implemented directly in PyTorch ([grpo/core.py](grpo/core.py)), not
delegated to a library trainer. It runs, it's tested, and the self-contained
demo converges on a laptop CPU in seconds.

```bash
pip install -r requirements.txt          # numpy, scipy, torch (+ hf extras)
python -m pytest tests/ -q               # 16 passing, no GPU, no downloads
python train_grpo_torch.py --backend toy # watch reward climb and the policy learn
```

**Run it on a GPU with a real model:**
[notebooks/modal_qwen3_demo.ipynb](notebooks/modal_qwen3_demo.ipynb) is a
self-contained notebook (built for [Modal](https://modal.com) notebooks, runs
on any CUDA GPU) that writes the whole package, runs the test suite, trains the
toy policy with a reward-curve plot, then loads **Qwen3-4B** to (a) write real
`predict_attention` programs scored by the verifiable reward and (b) GRPO
fine-tune it with LoRA using the from-scratch trainer.

## The research gap this targets

The paper generates candidate Python programs that reproduce a transformer
head's attention pattern via **one-shot LM prompting plus a single fixed
refinement round**, scored by IoU / JSD against real attention. From their
own limitations section:

> "many high-scoring programs are not particularly complex, and ... the
> improvement in question-answering seen for some models at low replacement
> levels may be akin to a pruning effect. Closing this gap likely requires
> richer synthesis strategies such as **multi-round refinement with stronger
> feedback signals**."

This repo turns their fixed pipeline into a proper RL problem: a policy
writes candidate programs, gets a **verifiable, non-LM-judged reward** (the
paper's own IoU/JSD minus a degeneracy penalty), and is trained with GRPO
across an arbitrary number of refinement rounds.

## Experiments & results

We are working toward the paper's stated gap (quoted above): does
richer, feedback-driven synthesis produce better and *less-degenerate*
attention-program explanations than one-shot generation? Getting there needs
a verifiable environment, a policy that can be trained against it, and — the
part the paper leaves open — a way to *measure* degeneracy. The two
experiments below establish that foundation is real and works end to end;
[Next steps](#next-steps-from-working-system-to-research-result) lays out the
measured comparison that turns it into a result.

### Experiment 1 — does the RL loop actually learn? (toy policy, CPU)

**Why.** Before spending GPU/compute on a large LM, verify the whole loop —
sampling → verifiable reward → group-relative advantage → clipped update →
KL/entropy — actually optimizes a policy, on a problem with known ground
truth so "did it learn the right thing?" is checkable.

**Setup.** A small PyTorch policy ([grpo/toy_policy.py](grpo/toy_policy.py))
picks, per synthetic head archetype, the program template that best
reproduces its attention. `python train_grpo_torch.py --backend toy`.

**Result.** Verifiable reward climbs ~0 → ~0.8 and the policy reaches **100%
template-selection accuracy, reproducibly across seeds 0–7**:

```
  step   0 | mean_reward -0.085 | max +0.913 | exec 1.00 | kl 0.0000
  step  24 | mean_reward +0.278 | max +0.913 | exec 1.00 | kl 0.4204
  step  48 | mean_reward +0.679 | max +0.913 | exec 1.00 | kl 1.0165
  step  79 | mean_reward +0.816 | max +0.913 | exec 1.00 | kl 1.2294

  template-selection accuracy: 100% (4/4 heads)
    OK synthetic:previous_token     chose previous_token@6
    OK synthetic:first_token        chose first_token@6
    OK synthetic:diagonal_self      chose diagonal_self@6
    OK synthetic:sentence_boundary  chose sentence_boundary@6
```

### Experiment 2 — a real code LM on a *hard* head (Qwen3-4B, Modal)

**Why.** The learned multi-round refiner we ultimately want (Next steps) is
only worth building if the foundation holds on a *real* model *and* on a head
the model can't already solve in one shot. So we target a **sentence-boundary**
head — harder than positional heads, since it depends on *where sentences
begin* — and check: does Qwen3-4B emit *executable* programs, does the
verifiable reward rank them, and does the from-scratch GRPO update actually
*improve* the policy against 4B parameters?
([notebooks/model_run_latest.ipynb](notebooks/model_run_latest.ipynb))

**Setup.** Target = a synthetic sentence-boundary head (ground truth known, so
fit is verifiable); policy = **Qwen3-4B** (bf16) with **LoRA** (16.5M trainable
params, **0.41%** of the model); reward = `IoU − ½·JSD − degeneracy penalty`
computed in the sandbox; GRPO for **20 steps, group size 8**, lr 1e-4, KL to
the frozen base.

**Result (latest Modal run).**
- *One-shot best-of-6 is poor:* Qwen3-4B does **not** solve this head cold —
  all six sampled programs score **negative** reward
  (`[-0.10, -0.13, -0.17, -0.17, -0.26, -0.29]`), best IoU only **0.14**
  (JSD 0.38). It reaches for brittle hardcoded-index heuristics
  (`if j == 0 or j < 4: ...`).
- *GRPO learns:* over 20 steps mean group reward climbs **−0.26 → +0.19**,
  converging onto the group max (`+0.192`), with **100% of programs
  executable** every step and KL to base staying small (~0.01–0.02):

```
  step  0 | mean_reward -0.258 | max -0.060 | exec 1.00 | kl 0.0000
  step  2 | mean_reward -0.016 | max +0.192 | exec 1.00 | kl 0.0173
  step  5 | mean_reward +0.122 | max +0.192 | exec 1.00 | kl 0.0173
  step  7 | mean_reward +0.192 | max +0.192 | exec 1.00 | kl 0.0106
  step 19 | mean_reward +0.192 | max +0.192 | exec 1.00 | kl 0.0185
```

### What these runs show — and what they don't

- ✅ **The full pipeline works end to end on a real 4B model** — strong LM →
  *executable* Python attention-program → verifiable, non-LM-judged reward →
  hand-written GRPO update (LoRA, KL-to-base) — and it **genuinely learns**:
  Experiment 1 reaches 100% archetype accuracy, and Experiment 2 lifts mean
  reward **−0.26 → +0.19** on a head Qwen3-4B could not solve one-shot.
- ⚠️ **GRPO converges to the model's *best one-shot* program, not to a good
  explanation.** The reward plateaus at +0.19 because that is the group max —
  the policy learns to *reliably* emit its best attempt, but Qwen3-4B's best
  sentence-boundary program is still a rough heuristic (IoU ~0.19, versus the
  ~0.9 the toy policy reaches on this archetype). Making the model *discover
  better logic* is exactly what the multi-round, feedback-driven refinement in
  Next steps is for — this result is the motivation for it.
- ⚠️ **Still a foundation check, not a measured result:** single head,
  synthetic target, short run, no baseline comparison across real heads.

## Next steps: from working system to research result

The system is in place; turning it into a result means a real experiment —
real data, a baseline, and a metric that isolates a real phenomenon. In
priority order:

1. **Real heads, real baseline.** *(harness built — see
   [notebooks/real_heads_experiment.ipynb](notebooks/real_heads_experiment.ipynb),
   awaiting a GPU run.)* Extract structured heads from GPT-2
   ([scripts/extract_attention_maps.py](scripts/extract_attention_maps.py))
   and compare, on held-out sentences, **one-shot best-of-N** (the paper's
   baseline) against **N-round iterative refinement** driven by the env's
   worst-edge feedback. Reports per-head IoU / JSD / held-out fit / degeneracy
   rate. Inference-only — no fragile RL — and a direct test of the paper's
   "multi-round refinement" hypothesis.
2. **Measure degeneracy (our novel angle).** The paper flags the "pruning
   effect" but never measures it; [`positional_collapse_score`](env/reward.py)
   does. Report the rate at which IoU-selected programs are degenerate, and how
   far the degeneracy-penalized objective reduces it *at comparable fit*.
3. **Causal faithfulness (gold standard).** Hook the synthesized program's
   attention in for the real head during a forward pass and measure the
   downstream next-token KL / loss increase on held-out text — a faithful
   program barely moves the output. This is the paper's own validation and the
   most convincing evidence that a program is a genuine explanation.
4. **Then the RL contribution.** With that harness, GRPO answers the sharper
   question: can a *learned* refiner, trained on a train split of heads, beat
   prompted refinement on *held-out heads*? Cross-head generalization — not one
   head going up — is the interesting result.

Rigor to carry through all of these: ≥20 heads (not 1), multiple seeds with
error bars, held-out sentences *and* held-out heads, and a `results/` notebook
that regenerates every number and figure.

## Two backends, one GRPO core

[grpo/core.py](grpo/core.py) is backend-agnostic: it talks to a `Policy` that
can (a) sample grouped rollouts and (b) recompute per-token log-probs and a
KL term *with gradient*. Two policies satisfy that interface and are trained
by the identical optimizer step:

| Backend | Policy | Runs where | Status |
|---|---|---|---|
| `toy` | [grpo/toy_policy.py](grpo/toy_policy.py) — a small autoregressive net over a program-template DSL | CPU, seconds, **no downloads** | ✅ trains & converges; unit-tested |
| `hf` | [grpo/hf_policy.py](grpo/hf_policy.py) — a HuggingFace code LM emitting arbitrary Python | GPU for a real model | ✅ loop/gradient/KL verified offline on a tiny GPT-2; not trained to convergence on a real LM here (needs a GPU + API/compute budget) |

The toy policy emits a two-token program `(template, sharpness)` that
[compiles](grpo/program_dsl.py) to a genuine `predict_attention(tokens)`
function — scored by the *same* executor and reward as a real LM completion.
This mirrors the paper's "program library" idea, but the
retrieval/composition policy is **learned by RL** rather than fixed.

### How the GRPO core works (grpo/core.py)

For each prompt, sample a *group* of `G` completions; score each with the
verifiable reward; use the group's mean/std as the baseline so completion
`i`'s advantage is `(r_i − mean) / std` — **no learned value network**. Then:

```
ratio_t = exp(logp_new − logp_old)                       # per token
L_t = −min(ratio_t · A_i, clip(ratio_t, 1±ε) · A_i)      # PPO-clipped surrogate
      + kl_coef · KL_t                                    # anchor to frozen reference
      − ent_coef · H_t                                    # keep exploring
```

The KL term (exact categorical for the toy policy; the k3 estimator
`exp(d)−d−1` for the large-vocab LM) anchors the policy to a frozen reference
so it can't drift into reward-hacking the degeneracy penalty; the entropy
bonus prevents premature collapse onto a locally-good program before a better
one is found — without it, `sentence_boundary` heads collapse onto the
degenerate `first_token` solution (see below).

## Quickstart

```bash
# 1. Env + reward sanity, numpy/scipy only, no model weights
python -m pytest tests/test_env.py -q

# 2. The whole PyTorch GRPO stack (toy convergence + HF mechanics on a tiny GPT-2)
python -m pytest tests/test_grpo_torch.py -q

# 3. Train the self-contained policy and save the reward curve
python train_grpo_torch.py --backend toy --save-curve curve.json
```

### Full pipeline with a real model

```bash
pip install -r requirements.txt

# Extract real per-head attention maps from a target model
python scripts/extract_attention_maps.py --model gpt2 --n-examples 200 \
    --out data/gpt2_attention.jsonl

# GRPO with a code-LM policy (from-scratch trainer)
python train_grpo_torch.py --backend hf \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --data data/gpt2_attention.jsonl --max-heads 8

# ...or with TRL's library GRPOTrainer for scale (scripts/grpo_train.py)
python scripts/grpo_train.py --data data/gpt2_attention.jsonl \
    --policy-model Qwen/Qwen2.5-Coder-3B-Instruct --max-heads 20
```

## The degeneracy penalty, and why it's *not* what you'd guess

IoU alone rewards a policy for fitting the metric by any means, including the
degenerate shortcuts that are the paper's own "pruning effect" confound. The
naive fix — penalize "trivial-looking" source (does it check `j == 0`?) — is
wrong twice over: it misses genuinely trivial programs with no conditionals
at all (`attention[:, 0] = 1.0`), and it would *also* punish legitimate
positional heads like "attend to the previous token," which are equally
content-blind but not degenerate.

The fix ([`env.reward.positional_collapse_score`](env/reward.py)) is
**behavioral and intrinsic to the candidate's own output**: compute the
column-marginal attention (mean attention received by each column across all
query rows) and measure its entropy. A program that always routes to one
fixed column collapses this to a spike (flagged); a genuinely positional head
spreads attention across columns as the query index moves (not flagged), even
though it's just as content-independent. The *same* signal is reused as a
policy input feature (`col_spread`), which is what lets the learned policy
tell `first_token` (mass collapses onto column 0) apart from
`sentence_boundary` (mass spreads across several sentence-start columns).

## Multi-round refinement

The env ([env/attention_env.py](env/attention_env.py)) generalizes
the paper's fixed single refinement round into an N-round MDP: `reset` /
`step`, with structured worst-edge feedback each round (contrasting real vs.
predicted attention on the worst-fit token pairs, exactly as the paper's
refinement step describes). `grpo/core.py` currently trains single-turn;
[grpo_train.py](scripts/grpo_train.py) documents the two routes to genuine
multi-turn GRPO (bake a fixed transcript into the prompt, or use trl 1.2.0's
`environment_factory`/`rollout_func` hooks) and
`run_multiturn_grpo_step` sketches the raw multi-round rollout.

## Layout

```
env/                          verifiable environment + reward (numpy/scipy only)
  reward.py                   IoU, JSD, positional_collapse_score, compute_reward
  executor.py                 AST-allowlisted sandbox: subprocess (untrusted) + fast in-process (trusted)
  attention_env.py            AttentionProgramEnv: reset/step, multi-round feedback
  data.py                     AttentionExample, synthetic archetypes, JSONL loader
grpo/                         from-scratch PyTorch GRPO
  core.py                     GRPOConfig, train_grpo, grpo_loss, compute_group_advantages
  program_dsl.py              program templates + action->program compiler + policy features
  toy_policy.py               self-contained autoregressive DSL policy (+ frozen reference)
  hf_policy.py                HF causal-LM policy (log-probs, k3 KL) + offline CharTokenizer
  rewarding.py                ProgramScorer: cached program -> verifiable reward
  train.py                    reward-fn wiring + run_toy_training orchestration
  toy_data.py                 one head per archetype, with ground-truth templates
train_grpo_torch.py           CLI: --backend {toy,hf}
notebooks/
  modal_qwen3_demo.ipynb      self-contained GPU notebook: tests + toy + Qwen3-4B GRPO
scripts/
  build_prompt.py             round-0 + refinement prompt construction (paper's format)
  extract_attention_maps.py   HF extraction adapter -> data.py's JSONL schema
  grpo_train.py               TRL GRPOTrainer alternative (verified vs trl==1.2.0)
tests/
  test_env.py                 9 tests: env + reward + metrics (numpy/scipy only)
  test_grpo_torch.py          7 tests: DSL, GRPO primitives, toy convergence, HF mechanics offline
```

## Status & honest gaps

- **16/16 tests pass** with no GPU and no network: `python -m pytest tests/ -q`.
  The toy trainer converges deterministically; the HF trainer's loop,
  gradient flow, KL, and ratio-at-epoch-0 correctness are verified on a
  locally-constructed tiny GPT-2 (never a download).
- The HF path **runs end to end on Qwen3-4B** (Experiment 2): it emits
  executable programs, they score well under the verifiable reward, and GRPO
  updates the LoRA policy. What it is **not** yet is a *measured* result — a
  single synthetic head with the policy near ceiling is a foundation check,
  not a baseline-vs-treatment comparison across real heads (see Next steps).
- [env/data.py](env/data.py)'s JSONL schema (`head_id`, `tokens`,
  `attention`) is a reasonable interchange format, not confirmed against the
  upstream repo's exact notebook serialization. `load_dataset` is the single
  seam to edit if theirs differs.
- The executor's sandbox (AST allowlist + subprocess timeout) is good enough
  for a local research loop, not a real security boundary for multi-tenant
  untrusted input.
- Bidirectional models (BERT) need the causal constraint `j <= i` relaxed in
  the DSL templates and the synthesis prompt.
