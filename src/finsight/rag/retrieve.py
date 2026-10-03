"""Hybrid retrieval: dense (vector) + sparse (BM25) fused via reciprocal-rank fusion."""

from __future__ import annotations

import json
from typing import Any

import chromadb
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from finsight.config import Settings
from finsight.llm import build_embeddings
from finsight.rag.index import (
    IndexIntegrityError,
    IndexMissingError,
    chunks_path,
    read_manifest,
    validate_manifest,
)
from finsight.rag.ingest import chunk_from_dict, tokenize
from finsight.rag.models import Chunk, RetrievedChunk

_RRF_K = 60


class HybridRetriever:
    """Dense + sparse retrieval, fused with reciprocal-rank fusion (RRF).

    RRF is a rank-based fusion that requires no score normalisation across the two
    retrievers — a common choice over weighted averaging. The underlying
    similarity scores are retained on each result for explainability even though
    they do not affect ranking.

    The index is validated against its manifest at construction, so a stale or
    inconsistent index fails loudly and early rather than at query time.
    """

    _settings: Settings
    _chunks: list[Chunk]
    _bm25: BM25Okapi | None
    _collection: Any
    _embeddings: Embeddings

    def __init__(
        self,
        settings: Settings,
        *,
        collection: Any | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings

        self._manifest = read_manifest(settings.index_dir)
        registry = chunks_path(settings.index_dir, self._manifest.generation)
        if not registry.exists():
            raise IndexMissingError(
                f"No chunk registry at {registry}. Build the index first: finsight ingest"
            )

        self._chunks = [
            chunk_from_dict(d) for d in json.loads(registry.read_text(encoding="utf-8"))
        ]
        if not self._chunks:
            raise IndexIntegrityError("Chunk registry is empty. Re-run: finsight ingest")

        # Validate the vector store against the registry before trusting either.
        validate_manifest(
            self._manifest,
            self._chunks,
            collection=settings.collection_name,
            embed_provider=settings.embed_provider,
            embed_model=settings.embed_model,
        )

        # BM25 cannot initialize an empty vocabulary. Such corpora are still
        # useful to the embedding model, so retain dense retrieval on its own.
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None

        self._collection = collection if collection is not None else self._load_chroma()
        self._embeddings = embeddings if embeddings is not None else build_embeddings(settings)
        self._validate_collection_ids()

    def _load_chroma(self) -> Any:
        client = chromadb.PersistentClient(path=str(self._settings.chroma_dir))
        try:
            return client.get_collection(name=self._manifest.vector_collection)
        except Exception as exc:  # chroma raises its own NotFound types
            raise IndexIntegrityError(
                f"Vector collection {self._manifest.vector_collection!r} is missing from "
                f"{self._settings.chroma_dir}. Re-run: finsight ingest"
            ) from exc

    def _validate_collection_ids(self) -> None:
        """Fail fast if the vector store and chunk registry disagree."""
        data = self._collection.get()
        stored = set(data["ids"])
        expected = {c.chunk_id for c in self._chunks}
        if stored == expected:
            documents = dict(zip(data["ids"], data.get("documents") or [], strict=True))
            if documents != {c.chunk_id: c.text for c in self._chunks}:
                raise IndexIntegrityError(
                    "Vector documents disagree with registry content. Re-run: finsight ingest"
                )
            return
        missing = sorted(expected - stored)[:5]
        extra = sorted(stored - expected)[:5]
        detail = []
        if missing:
            detail.append(
                f"{len(expected - stored)} registry chunk(s) absent from the "
                f"vector store (e.g. {missing})"
            )
        if extra:
            detail.append(
                f"{len(stored - expected)} vector id(s) absent from the registry (e.g. {extra})"
            )
        raise IndexIntegrityError(
            "Vector store and chunk registry disagree: " + "; ".join(detail) + ". "
            "Rebuild the index with: finsight ingest"
        )

    @property
    def generation(self) -> str:
        return self._manifest.generation

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for ``query`` via hybrid RRF fusion."""
        k = top_k or self._settings.retrieval_top_k
        cand = self._settings.retrieval_candidates
        vector_hits = self._vector_search(query, cand)
        bm25_hits = self._bm25_search(query, cand)
        return self._rrf_fuse(vector_hits, bm25_hits, k)

    def _vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        qvec = self._embeddings.embed_query(query)
        res = self._collection.query(query_embeddings=[qvec], n_results=k)
        ids = list(res.get("ids", [[]])[0])
        distances = res.get("distances")
        if distances is None:
            raise IndexIntegrityError(
                "Vector store returned no distances; cannot rank results. Re-run: finsight ingest"
            )
        # cosine distance -> similarity
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances[0], strict=True)]

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        terms = set(tokenize(query))
        scores = [float(s) for s in self._bm25.get_scores(list(terms))]
        # BM25 can assign zero or negative scores to actual matches in small
        # corpora; test lexical overlap, not score positivity.
        ranked = sorted(
            (
                (c.chunk_id, score)
                for c, score, frequencies in zip(
                    self._chunks, scores, self._bm25.doc_freqs, strict=True
                )
                if terms.intersection(frequencies)
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:k]

    def _rrf_fuse(
        self,
        vector_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}

        for name, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, (cid, score) in enumerate(hits, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
                components.setdefault(cid, {})[name] = round(score, 4)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        by_id = {c.chunk_id: c for c in self._chunks}
        missing = [cid for cid, _ in ranked if cid not in by_id]
        if missing:
            # Previously an unguarded by_id[cid] raised a bare KeyError here.
            raise IndexIntegrityError(
                f"Retrieved {len(missing)} unknown chunk id(s): {missing[:5]}. "
                "The vector store and chunk registry are out of sync; "
                "rebuild with: finsight ingest"
            )

        return [
            RetrievedChunk(
                chunk=by_id[cid],
                score=round(score, 4),
                rank=rank,
                method="rrf",
                component_scores=components.get(cid),
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]
