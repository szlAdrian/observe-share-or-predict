#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from habitat_llm.llm.hf_model import HFModel
from habitat_llm.llm.llama import Llama

logger = logging.getLogger(__name__)


class VLLMModel(HFModel):
    """Language model using vLLM's OpenAI-compatible API server for fast inference"""

    def __init__(self, conf: DictConfig):
        """
        Initialize the vLLM Model
        :param conf: The Language Model config
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.max_tokens = self.generation_params.max_tokens
        self.inference_mode = self.llm_conf.inference_mode
        self._hf_fallback_model = None
        self.init_vllm_model()

    def init_vllm_model(self):
        """
        Initialize the vLLM OpenAI-compatible client
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for vLLM mode. "
                'Install it with: pip install "openai>=1.55.3"'
            )

        # Get vLLM server configuration
        host = self.llm_conf.get("host", "localhost")
        port = self.llm_conf.get("port", 8000)
        timeout = self.llm_conf.get("timeout", 600)

        base_url = f"http://{host}:{port}/v1"
        self.model_name = self.generation_params.engine

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(api_key="not-used", base_url=base_url, timeout=timeout)
        logger.info(f"vLLM client initialized: {base_url}, model: {self.model_name}")

        # Set HF fallback model name. Defaults to the same model actually being
        # served by vLLM (not a hardcoded model) so a fallback never silently
        # loads the wrong checkpoint.
        self.hf_fallback_model_name = self.llm_conf.get(
            "hf_fallback_model", self.model_name
        )

    def _init_hf_fallback(self):
        """
        Lazily initialize HF model for fallback when vLLM generation fails
        (invalid/unsupported grammar, guided decoding not enabled server-side,
        or think-gated grammar which has no vLLM equivalent).
        """
        if self._hf_fallback_model is None:
            logger.warning(
                f"Initializing HF fallback model '{self.hf_fallback_model_name}' "
                "for this request. This loads the full model into GPU memory "
                "in addition to the running vLLM server."
            )
            fallback_conf = OmegaConf.create(OmegaConf.to_container(self.llm_conf))
            fallback_conf.inference_mode = "hf"
            fallback_conf.generation_params.engine = self.hf_fallback_model_name
            # Propagate 4-bit quantization to the fallback so a 70B fallback
            # doesn't load a full fp16 model on top of the vLLM server.
            fallback_conf.load_in_4bit = self.llm_conf.get("load_in_4bit", False)
            self._hf_fallback_model = Llama(fallback_conf)

    def generate_vllm(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generate using vLLM OpenAI API
        :param prompt: A string with the input to the language model
        :param stop: A string or list of strings that determine when to stop generation
        :param max_length: The max number of tokens to generate
        :param generation_args: Additional generation arguments (e.g., grammar_definition)
        """

        # Prepare parameters
        if max_length is None:
            max_length = self.generation_params.max_tokens
        if stop is None:
            stop = self.generation_params.stop

        temperature = self.generation_params.temperature
        do_sample = self.generation_params.do_sample

        grammar = None
        think_gated = False
        if generation_args is not None and "grammar_definition" in generation_args:
            if generation_args.get("think_gate_close_tag"):
                # Think-gated grammar (grammar only activates after a reasoning
                # model emits e.g. "</think>") has no vLLM guided-decoding
                # equivalent here. Route straight to the HF/transformers_cfg
                # fallback, which does support it.
                think_gated = True
                logger.info(
                    "think_gate_close_tag requested; this has no vLLM guided "
                    "decoding equivalent, using HF fallback for this request."
                )
            else:
                grammar = generation_args["grammar_definition"]

        if think_gated:
            self._init_hf_fallback()
            self._hf_fallback_model.generate(
                prompt=prompt,
                stop=stop,
                max_length=max_length,
                generation_args=generation_args,
            )
            self.response = self._hf_fallback_model.response
            self.batch_response = self._hf_fallback_model.batch_response
            self.response_raw = self._hf_fallback_model.response_raw
            return

        try:
            # Attempt vLLM generation (guided decoding if `grammar` is set)
            self._generate_vllm_api(
                prompt=prompt,
                stop=stop,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                grammar=grammar,
            )
        except Exception as e:
            from openai import APIConnectionError

            if isinstance(e, APIConnectionError):
                # Connection-level failure (server down, restarting, or
                # unreachable) — not a grammar problem, so don't fall back to
                # HF: reloading this checkpoint from scratch takes minutes,
                # and doing so alongside a still-resident vLLM server risks
                # OOM. Re-raise so the eval loop skips this episode cleanly;
                # rerun with +resume=true once the vLLM server is confirmed
                # healthy again.
                logger.error(
                    f"vLLM server unreachable at {self.client.base_url} "
                    f"(model={self.model_name}): {e}. Not falling back to "
                    "HF — check whether the vLLM server process is still "
                    "running (it may have crashed, e.g. from an OOM)."
                )
                raise
            if grammar is not None:
                logger.warning(
                    f"vLLM guided-decoding generation failed: {e}. "
                    "Falling back to HF mode for this request. "
                    "Make sure the vLLM server was started with "
                    "--guided-decoding-backend xgrammar."
                )
                logger.warning(f"grammar_definition that vLLM rejected:\n{grammar}")
                self._init_hf_fallback()
                self._hf_fallback_model.generate(
                    prompt=prompt,
                    stop=stop,
                    max_length=max_length,
                    generation_args=generation_args,
                )
                # Copy response from HF fallback model
                self.response = self._hf_fallback_model.response
                self.batch_response = self._hf_fallback_model.batch_response
                self.response_raw = self._hf_fallback_model.response_raw
                return
            else:
                # Re-raise if no grammar constraints (actual error)
                logger.error(
                    f"vLLM inference error against {self.client.base_url} "
                    f"(model={self.model_name}): {e}"
                )
                raise

    def _generate_vllm_api(
        self,
        prompt: str,
        stop: Optional[str],
        max_length: int,
        temperature: float,
        do_sample: bool,
        grammar: Optional[str] = None,
    ) -> None:
        """
        Internal method to call vLLM OpenAI API
        """
        self.response_raw: Dict[str, Any] = {}

        # Prepare stop sequences
        stop_sequences = []
        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = stop

        # Standard OpenAI Completions API fields
        generation_kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": self.generation_params.top_p,
            "stop": stop_sequences if stop_sequences else None,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        # vLLM-only sampling fields must go through extra_body: the openai
        # client's Completions.create() only accepts standard OpenAI fields
        # as direct kwargs and raises TypeError on anything else.
        extra_body: Dict[str, Any] = {
            "top_k": self.generation_params.top_k,
            "repetition_penalty": self.generation_params.repetition_penalty,
            # Pin BOS behavior to match HF's AutoTokenizer default so vLLM
            # output doesn't silently diverge from "hf" mode.
            "add_special_tokens": True,
        }
        if grammar is not None:
            extra_body["guided_grammar"] = grammar

        if self.generation_params.batch_response:
            # Generate multiple responses
            self.batch_response = []
            for _ in range(self.generation_params.n):
                response = self.client.completions.create(
                    **generation_kwargs, extra_body=extra_body
                )
                self.response_raw = response.model_dump() if hasattr(response, 'model_dump') else response.dict()

                # Extract text from response
                generation_text = response.choices[0].text

                # Apply stop word filtering
                if isinstance(stop, str):
                    generation_text = generation_text.split(stop)[0]
                else:
                    for s in stop_sequences:
                        if s in generation_text:
                            generation_text = generation_text.split(s)[0]
                            break

                # Remove trailing whitespace
                generation_text = generation_text.rstrip()
                self.batch_response.append(generation_text)
        else:
            # Generate single response
            response = self.client.completions.create(
                **generation_kwargs, extra_body=extra_body
            )
            self.response_raw = response.model_dump() if hasattr(response, 'model_dump') else response.dict()

            # Extract text from response
            generation_text = response.choices[0].text

            # Apply stop word filtering
            if isinstance(stop, str):
                generation_text = generation_text.split(stop)[0]
            else:
                for s in stop_sequences:
                    if s in generation_text:
                        generation_text = generation_text.split(s)[0]
                        break

            # Remove trailing whitespace
            self.response = generation_text.rstrip()

        logger.debug("vLLM generation completed successfully")

    def generate_hf_llm(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Override to use vLLM instead of HF for LLM-only generation
        :param prompt: A string with the input to the language model
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate
        :param generation_args: Additional generation arguments
        """
        self.generate_vllm(prompt, stop, max_length, generation_args)

    def get_logprobs(self) -> List[float]:
        """
        Get log probabilities of generated text
        vLLM provides limited logprob support, so return empty list
        """
        # vLLM doesn't expose detailed logprobs like HF does
        return []
