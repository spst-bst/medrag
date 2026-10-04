from __future__ import annotations

import json
from typing import Optional

import numpy as np

from medrag.config import EVIDENCE_DEPRIORITIZED_TYPES, EVIDENCE_PREFERRED_TYPES, RRF_K
from medrag.embeddings import Embedder, cosine_similarity_matrix
from medrag.models import RetrievedChunk
from medrag.store import Store

# Rough calibration against nomic-embed-text on a small medical-literature corpus: a visibly
# scattered question ("Why heart failure?", mixing pediatrics/epidemiology/monitoring subtopics)
# measured ~0.77 avg pairwise similarity; a narrow, single-subtopic question measured ~0.90. This
# threshold sits between them. It is NOT validated against a labeled set of good/bad questions —
# treat it as a starting point, and re-calibrate (e.g. against eval/questions.json) if it fires
# too often or not enough for your corpus/embedding model, which can shift this baseline a lot.
TOPIC_SPREAD_WARNING_THRESHOLD = 0.82


def _rank_map(ordered_ids: list[int]) -> dict[int, int]:
    return {cid: rank for rank, cid in enumerate(ordered_ids)}


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def evidence_boost_multiplier(pub_types: list[str]) -> float:
    types = set(pub_types)
    if types & EVIDENCE_PREFERRED_TYPES:
        return 1.1
    if types & EVIDENCE_DEPRIORITIZED_TYPES and not (types & EVIDENCE_PREFERRED_TYPES):
        return 0.9
    return 1.0


def retrieve(
    store: Store,
    embedder: Embedder,
    query: str,
    top_k: int = 6,
    bm25_limit: int = 50,
    cosine_limit: int = 50,
    embed_model: str = "nomic-embed-text",
    use_evidence_boost: bool = True,
) -> list[RetrievedChunk]:
    bm25_results = store.fts_search(query, bm25_limit)
    bm25_ranking = [cid for cid, _ in bm25_results]

    cosine_ranking: list[int] = []
    chunk_ids, matrix = store.get_embedding_matrix(embed_model)
    if chunk_ids:
        query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
        sims = cosine_similarity_matrix(query_vec, matrix)
        order = np.argsort(-sims)[:cosine_limit]
        cosine_ranking = [chunk_ids[i] for i in order]

    fused = reciprocal_rank_fusion([r for r in (bm25_ranking, cosine_ranking) if r])

    bm25_set = set(bm25_ranking)
    cosine_set = set(cosine_ranking)

    results: list[RetrievedChunk] = []
    for chunk_id, score in fused.items():
        row = store.get_chunk(chunk_id)
        if row is None:
            continue
        pub_types = json.loads(row["rec_pub_types"] or "[]")
        final_score = score
        if use_evidence_boost:
            final_score *= evidence_boost_multiplier(pub_types)
        rank_sources = []
        if chunk_id in bm25_set:
            rank_sources.append("bm25")
        if chunk_id in cosine_set:
            rank_sources.append("cosine")
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                pmid=row["pmid"],
                header=row["header"],
                text=row["text"],
                section=row["section"],
                title=row["rec_title"],
                year=row["rec_year"],
                journal=row["rec_journal"],
                pub_types=pub_types,
                score=final_score,
                rank_sources=rank_sources,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def topic_spread_warning(
    store: Store,
    embed_model: str,
    chunks: list[RetrievedChunk],
    threshold: float = TOPIC_SPREAD_WARNING_THRESHOLD,
) -> Optional[str]:
    """Heuristic: low average pairwise cosine similarity among the given chunks' embeddings
    suggests they span unrelated topics — often a sign the question was too broad or ambiguous
    for this corpus, rather than a retrieval bug. Returns a warning string, or None if there's
    not enough signal to judge (fewer than 2 chunks have an embedding for this model) or the
    spread looks fine.
    """
    vectors_by_id = store.get_embeddings_for_chunk_ids([c.chunk_id for c in chunks], embed_model)
    vectors = list(vectors_by_id.values())
    if len(vectors) < 2:
        return None

    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = matrix / norms
    sims = normalized @ normalized.T

    n = len(vectors)
    avg_similarity = (sims.sum() - np.trace(sims)) / (n * (n - 1))

    if avg_similarity < threshold:
        return (
            f"Retrieved chunks look topically scattered (avg pairwise similarity="
            f"{avg_similarity:.2f}, threshold={threshold}). This heuristic often fires on "
            "broad/ambiguous questions — if the answer reads like unrelated facts stitched "
            "together, try narrowing the question."
        )
    return None
