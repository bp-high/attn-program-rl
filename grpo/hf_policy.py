"""
HF causal-LM policy backend for the same from-scratch GRPO core.

This is the "real" research setting: the policy is a code LM that emits an
arbitrary Python `predict_attention` function as text, sampled with
`model.generate`. It implements the identical Policy protocol the toy DSL
policy does -- sample grouped rollouts, then recompute per-token log-probs
and a KL-to-reference term with gradient -- so grpo/core.py trains it with
the exact same optimizer step. The reward is still the verifiable env score
(untrusted code, so ProgramScorer routes it through the subprocess sandbox).

The per-token log-prob and KL math is written out by hand rather than pulled
from a trainer:
  - logp of completion token c_k comes from the logits at the *previous*
    position (P-1+k for a length-P prompt), gathered and log-softmaxed.
  - KL to the reference uses the k3 estimator kl = exp(d) - d - 1 with
    d = logp_ref - logp_new -- unbiased, non-negative, and cheap (no full
    vocab sum), which matters when the vocab is 50k+.

`CharTokenizer` + a locally-constructed tiny GPT-2 let the whole path run
offline (see tests/test_grpo_torch.py). In production you pass
`AutoTokenizer.from_pretrained(m)` and `AutoModelForCausalLM.from_pretrained(m)`.
"""
from __future__ import annotations

import copy
import re

import torch
import torch.nn.functional as F

from grpo.core import Rollout

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Pull a Python code block from an LM completion; fall back to raw text
    (some models emit the bare function when told 'respond with ONLY ...')."""
    m = _CODE_BLOCK_RE.search(completion)
    return (m.group(1) if m else completion).strip()


def _is_dispatched(model) -> bool:
    """True if accelerate has already placed the model across device(s)
    (device_map=...); in that case we must not call .to(device) on it."""
    return getattr(model, "hf_device_map", None) is not None


def _sequence_logps(model, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of `completion_ids` continuing `prompt_ids`.

    logits at position t predict token t+1, so the k-th completion token
    (absolute position P+k) is predicted by logits[P-1+k]. log_softmax is
    taken in float32 so bf16/fp16 policies stay numerically stable."""
    ids = torch.cat([prompt_ids, completion_ids])[None]     # [1, P+T]
    logits = model(ids).logits[0]                           # [P+T, V]
    P, T = prompt_ids.shape[0], completion_ids.shape[0]
    pred = logits[P - 1: P - 1 + T].float()                # [T, V]
    return F.log_softmax(pred, dim=-1).gather(1, completion_ids[:, None]).squeeze(1)


class HFPolicy:
    def __init__(self, model, tokenizer, prompts: list[str], ref_model=None,
                 device: str = "cpu", max_new_tokens: int = 256,
                 temperature: float = 1.0, top_p: float = 1.0,
                 add_special_tokens: bool = True):
        # device_map-loaded / accelerate models place themselves; only .to()
        # when the caller passes a plain device string for a single-device model.
        if _is_dispatched(model):
            self.model = model
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        # add_special_tokens=False when `prompts` are already chat-templated
        # strings (they carry their own special tokens as text).
        self.add_special_tokens = add_special_tokens

        # Frozen reference for the KL anchor.
        self._peft_ref = False
        if ref_model is not None:
            self.ref = ref_model if _is_dispatched(ref_model) else ref_model.to(self.device)
            for p in self.ref.parameters():
                p.requires_grad_(False)
            self.ref.eval()
        elif hasattr(model, "disable_adapter") and callable(model.disable_adapter):
            # PEFT/LoRA: the reference is just the base model with adapters
            # switched off -- no second full copy of the weights in memory.
            # (LoRA is zero-initialized, so ref == policy at step 0: KL 0.)
            self.ref = None
            self._peft_ref = True
        else:
            self.ref = copy.deepcopy(model).to(self.device)
            for p in self.ref.parameters():
                p.requires_grad_(False)
            self.ref.eval()
        # Keep the policy in eval mode throughout (dropout off) so log-probs
        # are a deterministic function of the params: the behavior-policy
        # `old_logps` and the re-evaluated `new_logps` then agree exactly at
        # inner-epoch 0, making the importance ratio 1 as GRPO assumes.
        # eval() only disables dropout/batchnorm; gradients still flow.
        self.model.eval()

        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = getattr(tokenizer, "pad_token_id", None) or self.eos_id

    def parameters(self):
        # Only the params that actually train (all of them for full FT; just
        # the adapters for LoRA) -- so the optimizer never allocates state for
        # a frozen 4B base model.
        return [p for p in self.model.parameters() if p.requires_grad]

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _truncate_completion(self, comp_ids: torch.Tensor) -> torch.Tensor:
        """Keep completion tokens up to and including the first eos; drop the
        trailing padding `generate` emits so log-probs aren't computed over pad."""
        if self.eos_id is not None:
            hit = (comp_ids == self.eos_id).nonzero()
            if hit.numel() > 0:
                return comp_ids[: int(hit[0]) + 1]
        return comp_ids

    def sample(self, prompt_indices: list[int], group_size: int) -> list[Rollout]:
        rollouts: list[Rollout] = []
        self.model.eval()
        for pi in prompt_indices:
            prompt_ids = self._encode(self.prompts[pi])
            with torch.no_grad():
                gen = self.model.generate(
                    prompt_ids[None],
                    attention_mask=torch.ones_like(prompt_ids)[None],  # silence pad==eos warning
                    do_sample=True, num_return_sequences=group_size,
                    max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                    top_p=self.top_p, pad_token_id=self.pad_id,
                )
            P = prompt_ids.shape[0]
            for g in range(group_size):
                comp_ids = self._truncate_completion(gen[g, P:])
                if comp_ids.numel() == 0:
                    comp_ids = gen[g, P:P + 1]  # guard: never zero-length
                with torch.no_grad():
                    old_lp = _sequence_logps(self.model, prompt_ids, comp_ids).detach().cpu()
                # skip_special_tokens=True: instruct models end a turn with an
                # eos like <|im_end|>; left in, it glues onto `return attention`
                # and the program fails to parse.
                decoded = self.tokenizer.decode(comp_ids.tolist(), skip_special_tokens=True)
                text = extract_code(decoded)
                rollouts.append(Rollout(
                    prompt_index=pi, action=(prompt_ids.cpu(), comp_ids.cpu()),
                    text=text, old_logps=old_lp,
                ))
        return rollouts

    def _ref_logps(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor) -> torch.Tensor:
        if self._peft_ref:
            # Reference = base model with LoRA adapters disabled.
            with torch.no_grad(), self.model.disable_adapter():
                return _sequence_logps(self.model, prompt_ids, comp_ids)
        with torch.no_grad():
            return _sequence_logps(self.ref, prompt_ids, comp_ids)

    def evaluate(self, rollouts: list[Rollout]):
        new_logps, kls = [], []
        for r in rollouts:
            prompt_ids = r.action[0].to(self.device)
            comp_ids = r.action[1].to(self.device)
            lp_new = _sequence_logps(self.model, prompt_ids, comp_ids)         # grad
            lp_ref = self._ref_logps(prompt_ids, comp_ids)                     # frozen
            d = lp_ref - lp_new
            kl = torch.exp(d) - d - 1.0                                        # k3 estimator
            new_logps.append(lp_new)
            kls.append(kl)
        return new_logps, kls, None  # no entropy bonus for large-vocab LM


# --------------------------------------------------------------------------
# Offline test aid: a trivial char tokenizer so the HF path runs with a
# locally-constructed tiny model and zero network access.
# --------------------------------------------------------------------------

class CharTokenizer:
    """Byte-ish char tokenizer implementing the .encode/.decode/.eos_token_id/
    .pad_token_id surface HFPolicy needs. Real runs use AutoTokenizer."""
    eos_token_id = 256
    pad_token_id = 257
    vocab_size = 258

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [min(ord(c), 255) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if i < 256)
