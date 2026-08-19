"""Stylometric feature extraction.

The idea (classic authorship-attribution / Burrows' Delta style): a person's
use of common *function words* (the, of, and, but, ...) and basic rhythm
(sentence length, punctuation habits) is remarkably stable and hard to fake,
even when the *content* of what they write changes completely. So instead of
asking "does this look like plagiarism", we ask "does this still sound like
THIS student" - which also catches contract cheating / ghostwriting that
web-similarity search would miss entirely (because the ghostwritten text
matches nothing online).
"""
from __future__ import annotations

import re
from collections import Counter

import textstat

from app.ingestion.parser import split_sentences

# A curated function-word list (articles, prepositions, conjunctions,
# pronouns, common auxiliaries). Deliberately content-free - these words
# don't change with topic, only with a writer's habits.
_FUNCTION_WORDS = [
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
    "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "she", "do", "how", "their", "if",
    "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
    "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
    "however", "although", "therefore", "moreover", "thus", "furthermore",
]

FEATURE_NAMES: list[str] = [
    "avg_sentence_len",
    "sentence_len_std",
    "avg_word_len",
    "type_token_ratio",
    "flesch_reading_ease",
    "comma_per_100w",
    "semicolon_per_100w",
    "exclaim_per_100w",
    "avg_paragraph_len",
] + [f"fw_{w}" for w in _FUNCTION_WORDS]

_WORD_RE = re.compile(r"[A-Za-z']+")


def extract_style_vector(text: str) -> dict[str, float]:
    """Compute a fixed-order dict of stylometric features for one document.

    Returns a dict (not a bare list) so callers can inspect *which* features
    drove a drift flag - important for producing an explainable report
    instead of an opaque score.
    """
    sentences = split_sentences(text)
    words = _WORD_RE.findall(text.lower())
    n_words = max(len(words), 1)

    sent_lens = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
    avg_sentence_len = sum(sent_lens) / len(sent_lens)
    sentence_len_std = _std(sent_lens)

    avg_word_len = sum(len(w) for w in words) / n_words if words else 0.0
    type_token_ratio = len(set(words)) / n_words if words else 0.0

    try:
        flesch = textstat.flesch_reading_ease(text) if text.strip() else 0.0
    except Exception:  # noqa: BLE001 - textstat's syllable counter can raise on
        # unusual/degenerate input (e.g. no vowels); readability is one signal
        # of ~15, not worth failing the whole feature vector over.
        flesch = 0.0

    comma = text.count(",") / n_words * 100
    semicolon = text.count(";") / n_words * 100
    exclaim = text.count("!") / n_words * 100

    paragraphs = [p for p in text.split("\n\n") if p.strip()] or [text]
    avg_paragraph_len = sum(len(_WORD_RE.findall(p)) for p in paragraphs) / len(paragraphs)

    counts = Counter(words)
    fw_freqs = {f"fw_{w}": counts.get(w, 0) / n_words * 100 for w in _FUNCTION_WORDS}

    features = {
        "avg_sentence_len": avg_sentence_len,
        "sentence_len_std": sentence_len_std,
        "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "flesch_reading_ease": flesch,
        "comma_per_100w": comma,
        "semicolon_per_100w": semicolon,
        "exclaim_per_100w": exclaim,
        "avg_paragraph_len": avg_paragraph_len,
        **fw_freqs,
    }
    return {name: features[name] for name in FEATURE_NAMES}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5
