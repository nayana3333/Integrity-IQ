"""Document ingestion: pull raw text out of PDF/DOCX/TXT submissions and
split it into sentence-aware, overlapping chunks suitable for embedding.

Chunking (not raw whole-document comparison) matters here: plagiarism and
paraphrase detection needs to catch a *copied paragraph* inside an otherwise
original essay, not just a whole-document similarity score that would dilute
the signal.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass
class Chunk:
    index: int
    text: str
    start_sentence: int
    end_sentence: int


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF, DOCX, or TXT submission."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext in ("txt", "md"):
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    # Collapse paragraph breaks to spaces before splitting so a sentence
    # that happens to wrap a newline isn't cut in half.
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(flat)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, sentences_per_chunk: int = 5, overlap: int = 2) -> list[Chunk]:
    """Sliding-window chunking over sentences.

    Overlap ensures a plagiarized passage that straddles a chunk boundary
    still shows up fully within at least one chunk.
    """
    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be smaller than sentences_per_chunk")

    sentences = split_sentences(text)
    if not sentences:
        return []

    step = sentences_per_chunk - overlap
    chunks: list[Chunk] = []
    idx = 0
    i = 0
    while i < len(sentences):
        window = sentences[i : i + sentences_per_chunk]
        if not window:
            break
        chunks.append(
            Chunk(
                index=idx,
                text=" ".join(window),
                start_sentence=i,
                end_sentence=i + len(window) - 1,
            )
        )
        idx += 1
        if i + sentences_per_chunk >= len(sentences):
            break
        i += step
    return chunks
