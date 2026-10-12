# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional

import torch
from transformers.generation.logits_process import LogitsProcessor


class ThinkGatedGrammarProcessor(LogitsProcessor):
    """
    Wraps a GrammarConstrainedLogitsProcessor so that it is a no-op
    (unconstrained generation) until the model has generated
    `think_close_tag` (e.g. "</think>") in its decoded output, after which
    all subsequent tokens are constrained by the wrapped grammar processor.

    `max_think_tokens` is an optional safety valve: if the model never
    emits the close tag, the grammar kicks in anyway after this many
    generated tokens so generation doesn't run away unconstrained.
    """

    def __init__(
        self,
        grammar_processor,
        tokenizer,
        think_close_tag: str,
        prompt_len: int,
        max_think_tokens: Optional[int] = None,
    ):
        self.grammar_processor = grammar_processor
        self.tokenizer = tokenizer
        self.think_close_tag = think_close_tag
        self.prompt_len = prompt_len
        self.max_think_tokens = max_think_tokens
        self.activated = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not self.activated:
            generated_ids = input_ids[0, self.prompt_len :]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
            timed_out = (
                self.max_think_tokens is not None
                and generated_ids.shape[0] >= self.max_think_tokens
            )
            if self.think_close_tag in generated_text or timed_out:
                self.activated = True
                print(
                    f"[ThinkGatedGrammarProcessor] ACTIVATED (timed_out={timed_out}). "
                    f"generated_text tail: {generated_text[-60:]!r}"
                )
            else:
                return scores
        return self.grammar_processor(input_ids, scores)
