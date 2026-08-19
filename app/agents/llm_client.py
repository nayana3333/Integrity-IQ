"""Thin, swappable client around IBM Granite.

Two backends, selected by environment variables, so the project runs both
in the IBM Cloud Lite / watsonx.ai setup the assignment mandates, AND
locally without cloud credentials (e.g. for CI, or a quick demo on a plane):

- "watsonx"  -> calls IBM watsonx.ai's hosted Granite model (production path).
- "local"    -> runs a small Granite Instruct checkpoint locally via
                transformers (ibm-granite/granite-3.0-2b-instruct).

Either way, callers only ever see `GraniteClient.generate(prompt)` - the
rest of the codebase doesn't know or care which backend is active.
"""
from __future__ import annotations

import functools
import os


class GraniteClient:
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.environ.get("GRANITE_BACKEND", "local")

    def generate(self, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2) -> str:
        if self.backend == "watsonx":
            return self._generate_watsonx(prompt, max_new_tokens, temperature)
        return self._generate_local(prompt, max_new_tokens, temperature)

    # -- watsonx.ai (IBM Cloud Lite) --------------------------------------
    def _generate_watsonx(self, prompt: str, max_new_tokens: int, temperature: float) -> str:

        api_key = os.environ["WATSONX_API_KEY"]
        project_id = os.environ["WATSONX_PROJECT_ID"]
        url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        model_id = os.environ.get("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct")

        model = _watsonx_model(api_key, project_id, url, model_id)
        response = model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "decoding_method": "greedy" if temperature == 0 else "sample",
            },
        )
        return response.strip()

    # -- local Granite checkpoint (dev / offline demo) ---------------------
    def _generate_local(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        tokenizer, model = _local_granite_model()
        import torch

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


@functools.lru_cache(maxsize=1)
def _watsonx_model(api_key: str, project_id: str, url: str, model_id: str):
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    return ModelInference(
        model_id=model_id,
        credentials=Credentials(api_key=api_key, url=url),
        project_id=project_id,
    )


@functools.lru_cache(maxsize=1)
def _local_granite_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("LOCAL_GRANITE_MODEL", "ibm-granite/granite-3.0-2b-instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model
