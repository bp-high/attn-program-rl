"""
Builds the toy GRPO training problem: one head per attention archetype, each
with known ground truth, so we can both compute a verifiable reward and check
whether the learned policy selects the correct program template.

Archetype names in env/data.py line up 1:1 with template names in
grpo/program_dsl.py (previous_token, first_token, diagonal_self,
sentence_boundary), so the expected template index is just the template whose
name matches the archetype. two_back exists as a template but has no matching
archetype -- it is a distractor action the policy must learn *not* to pick.
"""
from __future__ import annotations

from env.data import synthetic_head_dataset
from grpo.program_dsl import TEMPLATE_NAMES
from grpo.train import HeadPrompt

DEFAULT_ARCHETYPES = ["previous_token", "first_token", "diagonal_self", "sentence_boundary"]


def build_toy_prompts(archetypes: list[str] | None = None, n_examples: int = 6,
                      multi_sentence: bool = True, seed: int = 0) -> list[HeadPrompt]:
    archetypes = archetypes or DEFAULT_ARCHETYPES
    prompts: list[HeadPrompt] = []
    for k, arch in enumerate(archetypes):
        examples = synthetic_head_dataset(arch, n_examples=n_examples, noise=0.0,
                                          multi_sentence=multi_sentence, seed=seed + k)
        expected = TEMPLATE_NAMES.index(arch) if arch in TEMPLATE_NAMES else None
        prompts.append(HeadPrompt(head_id=f"synthetic:{arch}", examples=examples,
                                  expected_template=expected))
    return prompts
