"""Chunk-level semantic similarity search across all past submissions.

We embed every submission chunk-by-chunk (see app.ingestion.parser.chunk_text)
and keep a persistent index of every chunk ever submitted in a course. A new
submission's chunks are then compared against that index: any chunk that
scores highly similar to a chunk from a *different* student is exactly the
"paraphrased-from-a-classmate's-essay" case that simple string/n-gram
matching misses, because the wording can be completely rewritten while the
meaning - and therefore the embedding - stays close.

Index backend: a plain NumPy brute-force cosine-similarity search, not a
vector database. That's a deliberate choice, not a missing feature: a single
course's submission history tops out at a few thousand chunks (dozens of
students x a handful of assignments x ~10 chunks each), and at that scale an
ANN index (Chroma/FAISS/Milvus) buys nothing - a single (N, 384) @ (384,)
matrix-vector product is sub-millisecond even at N=50,000 on CPU, and it
removes a compiled-native-extension dependency entirely (we initially used
ChromaDB here and hit a reproducible crash in its Rust extension on this
Windows setup - swapping to NumPy sidesteps that class of problem completely
rather than chasing a native-dependency bug we don't control). If a
deployment ever needs to search across tens of millions of chunks - e.g. a
university-wide, multi-course, multi-year index - swap this class for a
Chroma/FAISS-backed one behind the same interface; nothing else in the
codebase would need to change.
"""
from __future__ import annotations

import functools
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.parser import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


@dataclass
class SimilarityMatch:
    query_chunk_index: int
    query_text: str
    matched_submission_id: str
    matched_student_id: str
    matched_chunk_index: int
    matched_text: str
    similarity: float  # 0..1, 1 = identical meaning


class VectorStore:
    """Persists to `{persist_dir}/{collection_name}.npy` (embeddings, float32,
    L2-normalized) + `{collection_name}.meta.json` (parallel metadata/text).

    Not safe for concurrent multi-process writers (a single in-process lock
    guards against races within one process, but two separate `uvicorn`
    workers would each hold their own copy) - see README "Future Work" for
    the same caveat already documented for the per-course fusion-model cache.
    A single-worker deployment (fine for a course-scale tool) has no issue.
    """

    def __init__(self, persist_dir: str = "vector_store_data", collection_name: str = "submissions"):
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_path = self._dir / f"{collection_name}.npy"
        self._meta_path = self._dir / f"{collection_name}.meta.json"
        self._lock = threading.Lock()
        self._embedder = get_embedder()

        self._embeddings: np.ndarray = np.zeros((0, self._embedder.get_embedding_dimension()), dtype=np.float32)
        self._meta: list[dict] = []  # one dict per row: course_id, submission_id, student_id, chunk_index, text
        self._load()

    def _load(self) -> None:
        if self._embeddings_path.exists() and self._meta_path.exists():
            self._embeddings = np.load(self._embeddings_path)
            self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        np.save(self._embeddings_path, self._embeddings)
        self._meta_path.write_text(json.dumps(self._meta), encoding="utf-8")

    def add_submission(
        self,
        course_id: str,
        submission_id: str,
        student_id: str,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_embeddings = self._embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
        new_meta = [
            {
                "course_id": course_id,
                "submission_id": submission_id,
                "student_id": student_id,
                "chunk_index": c.index,
                "text": c.text,
            }
            for c in chunks
        ]

        with self._lock:
            # Upsert semantics: drop any existing rows for this submission
            # first (handles re-analysis of an already-indexed submission).
            keep = [i for i, m in enumerate(self._meta) if m["submission_id"] != submission_id]
            self._embeddings = (
                self._embeddings[keep] if keep else np.zeros((0, new_embeddings.shape[1]), dtype=np.float32)
            )
            self._meta = [self._meta[i] for i in keep]

            self._embeddings = np.vstack([self._embeddings, new_embeddings])
            self._meta.extend(new_meta)
            self._save()

    def find_similar(
        self,
        chunks: list[Chunk],
        course_id: str,
        exclude_submission_id: str,
        top_k: int = 3,
        similarity_threshold: float = 0.80,
    ) -> dict[int, list[SimilarityMatch]]:
        """For each query chunk, find the most similar chunks from OTHER
        submissions in the same course, above `similarity_threshold`.
        """
        if not chunks:
            return {}

        with self._lock:
            course_mask = np.array(
                [m["course_id"] == course_id and m["submission_id"] != exclude_submission_id for m in self._meta],
                dtype=bool,
            )

        if not course_mask.any():
            return {}

        candidate_embeddings = self._embeddings[course_mask]
        candidate_meta = [m for m, keep in zip(self._meta, course_mask, strict=True) if keep]

        texts = [c.text for c in chunks]
        query_embeddings = self._embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

        # (n_query, dim) @ (dim, n_candidates) -> (n_query, n_candidates) cosine
        # similarities in one shot, since every vector is L2-normalized.
        similarities = query_embeddings @ candidate_embeddings.T

        matches_by_chunk: dict[int, list[SimilarityMatch]] = {}
        for qi, chunk in enumerate(chunks):
            sims = similarities[qi]
            order = np.argsort(-sims)

            found: list[SimilarityMatch] = []
            for idx in order:
                sim = float(sims[idx])
                if sim < similarity_threshold:
                    break  # sorted descending, nothing further will clear the bar
                meta = candidate_meta[idx]
                found.append(
                    SimilarityMatch(
                        query_chunk_index=chunk.index,
                        query_text=chunk.text,
                        matched_submission_id=meta["submission_id"],
                        matched_student_id=meta["student_id"],
                        matched_chunk_index=meta["chunk_index"],
                        matched_text=meta["text"],
                        similarity=sim,
                    )
                )
                if len(found) >= top_k:
                    break

            if found:
                matches_by_chunk[chunk.index] = found

        return matches_by_chunk
