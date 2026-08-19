"""Perplexity/burstiness features for AI-generated text detection.

Core idea (well established in the AI-text-detection literature, e.g. the
GLTR / DetectGPT line of work): LLM-generated text tends to sit in the
high-probability region of a language model's own distribution far more
consistently than human text does. Humans are "bursty" - some sentences are
very predictable, others are surprising word choices; LLM output is more
uniformly predictable. So we measure:

  1. mean perplexity under a reference LM (lower = more "typical"/predictable)
  2. burstiness = variance of per-sentence log-likelihood (lower = more
     uniform = more LLM-like)
  3. lexical diversity (type-token ratio) as a cheap secondary signal

These are *features* fed into a trained classifier (see classifier.py), not
a standalone verdict - perplexity alone is a weak, easily-gamed signal, which
is exactly why we fuse it with the semantic-similarity and stylometric-drift
signals downstream (see app.fusion).
"""
from __future__ import annotations

import functools
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.ingestion.parser import split_sentences

_REF_MODEL_NAME = "gpt2"

AI_FEATURE_NAMES = [
    "mean_perplexity",
    "burstiness",
    "type_token_ratio",
    "avg_sentence_len",
]


@functools.lru_cache(maxsize=1)
def _load_reference_model():
    tokenizer = AutoTokenizer.from_pretrained(_REF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(_REF_MODEL_NAME)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def _sentence_log_likelihood(sentence: str) -> float | None:
    tokenizer, model = _load_reference_model()
    encodings = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encodings["input_ids"]
    if input_ids.shape[1] < 2:
        return None
    outputs = model(input_ids, labels=input_ids)
    # outputs.loss is the mean per-token negative log-likelihood
    return -outputs.loss.item()


def extract_ai_features(text: str, max_sentences: int = 30) -> dict[str, float]:
    """Compute perplexity/burstiness features for a document.

    Sampling is capped at `max_sentences` for speed - a stable estimate
    doesn't need every sentence in a long essay, and this keeps per-document
    latency roughly constant regardless of submission length.
    """
    sentences = split_sentences(text)[:max_sentences]
    log_likelihoods = [ll for s in sentences if (ll := _sentence_log_likelihood(s)) is not None]

    if not log_likelihoods:
        return dict.fromkeys(AI_FEATURE_NAMES, 0.0)

    mean_ll = sum(log_likelihoods) / len(log_likelihoods)
    mean_perplexity = math.exp(-mean_ll)

    if len(log_likelihoods) > 1:
        mean = sum(log_likelihoods) / len(log_likelihoods)
        variance = sum((x - mean) ** 2 for x in log_likelihoods) / (len(log_likelihoods) - 1)
        burstiness = variance ** 0.5
    else:
        burstiness = 0.0

    words = text.split()
    n_words = max(len(words), 1)
    type_token_ratio = len({w.lower() for w in words}) / n_words
    avg_sentence_len = n_words / max(len(sentences), 1)

    return {
        "mean_perplexity": mean_perplexity,
        "burstiness": burstiness,
        "type_token_ratio": type_token_ratio,
        "avg_sentence_len": avg_sentence_len,
    }
