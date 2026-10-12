#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
from typing import Dict, List, Optional

import httpx
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from habitat_llm.llm.base_llm import BaseLLM, Prompt
from habitat_llm.llm.openai_chat import generate_message


class GenericOpenAIChat(BaseLLM):
    """
    LLM backend for any OpenAI-compatible API (OpenRouter, standard OpenAI, etc.).
    Configured via YAML fields `api_key_env` and optional `base_url`.
    """

    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

        api_key_env = self.llm_conf.get("api_key_env", "OPENAI_API_KEY")
        try:
            api_key = os.getenv(api_key_env)
            assert api_key and len(api_key) > 0
        except Exception:
            raise ValueError(f"No {api_key_env} environment variable provided")

        base_url = self.llm_conf.get("base_url", None)
        client_kwargs = {"api_key": api_key}
        http_kwargs = {"follow_redirects": True}
        if base_url:
            client_kwargs["base_url"] = base_url
            http_kwargs["base_url"] = base_url

        # Pass http_client explicitly to avoid the openai+httpx>=0.28 proxies conflict.
        client_kwargs["http_client"] = httpx.Client(**http_kwargs)
        self.client = OpenAI(**client_kwargs)

        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 60,
    ):
        """
        Generate a response via an OpenAI-compatible API.
        :param prompt: A string (or multimodal list) input to the language model.
        :param stop: A string that determines when to stop generation.
        :param max_length: The max number of tokens to generate.
        :param request_timeout: Maximum time before timeout.
        :param generation_args: Grammar constraints — not used for API models.
        """
        params = OmegaConf.to_object(self.generation_params)

        if stop is None and len(self.generation_params.stop) > 0:
            stop = self.generation_params.stop
        params["stop"] = stop

        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        if type(prompt) is str:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append(generate_message(prompt, image_detail="low"))

        response = self.client.chat.completions.create(
            model=params["model"],
            messages=messages,
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            stop=params.get("stop") or None,
            frequency_penalty=params.get("frequency_penalty", 0),
            presence_penalty=params.get("presence_penalty", 0),
            timeout=request_timeout,
        )
        text_response = response.choices[0].message.content
        self.response = text_response

        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            text_response = text_response.split(stop)[0]
        return text_response
