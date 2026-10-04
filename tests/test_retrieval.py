from __future__ import annotations

from medrag.indexer import index_records
from medrag.models import RetrievedChunk
from medrag.retrieval import (
    evidence_boost_multiplier,
    reciprocal_rank_fusion,
    retrieve,
    topic_spread_warning,
)
from tests.fakes import FakeEmbedder


def _chunk(chunk_id: int, pmid: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        pmid=pmid,
        header="h",
        text="t",
        section=None,
        title="t",
        year=2020,
        journal=None,
        pub_types=[],
        score=1.0,
    )


def test_reciprocal_rank_fusion_combines_rankings():
    bm25_ranking = [10, 20, 30]
    cosine_ranking = [10, 30, 40]
    fused = reciprocal_rank_fusion([bm25_ranking, cosine_ranking], k=60)
    # chunk 10 ranks first in both lists -> highest combined score
    assert fused[10] > fused[20]
    assert fused[10] > fused[30]
    assert 40 in fused


def test_evidence_boost_preferred_types():
    assert evidence_boost_multiplier(["Randomized Controlled Trial"]) == 1.1
    assert evidence_boost_multiplier(["Meta-Analysis"]) == 1.1
    assert evidence_boost_multiplier(["Systematic Review"]) == 1.1


def test_evidence_boost_deprioritized_types():
    assert evidence_boost_multiplier(["Case Reports"]) == 0.9


def test_evidence_boost_neutral_types():
    assert evidence_boost_multiplier(["Journal Article"]) == 1.0
    assert evidence_boost_multiplier([]) == 1.0


def test_evidence_boost_preferred_wins_over_deprioritized():
    assert evidence_boost_multiplier(["Case Reports", "Randomized Controlled Trial"]) == 1.1


def test_retrieve_ranks_relevant_chunk_first(tmp_store, sample_records):
    embedder = FakeEmbedder()
    index_records(tmp_store, embedder, sample_records, model="fake-model-v1")

    results = retrieve(
        tmp_store, embedder, "dapagliflozin heart failure hospitalization",
        top_k=3, embed_model="fake-model-v1",
    )
    assert len(results) > 0
    assert results[0].pmid == "10000001"


def test_retrieve_respects_top_k(tmp_store, sample_records):
    embedder = FakeEmbedder()
    index_records(tmp_store, embedder, sample_records, model="fake-model-v1")

    results = retrieve(tmp_store, embedder, "diabetes", top_k=2, embed_model="fake-model-v1")
    assert len(results) <= 2


def test_retrieve_boost_can_be_disabled(tmp_store, sample_records):
    embedder = FakeEmbedder()
    index_records(tmp_store, embedder, sample_records, model="fake-model-v1")

    boosted = retrieve(
        tmp_store, embedder, "SGLT2 inhibitor adverse event case",
        top_k=5, embed_model="fake-model-v1", use_evidence_boost=True,
    )
    unboosted = retrieve(
        tmp_store, embedder, "SGLT2 inhibitor adverse event case",
        top_k=5, embed_model="fake-model-v1", use_evidence_boost=False,
    )
    assert isinstance(boosted, list) and isinstance(unboosted, list)


def test_retrieve_marks_which_ranking_each_chunk_came_from(tmp_store, sample_records):
    embedder = FakeEmbedder()
    index_records(tmp_store, embedder, sample_records, model="fake-model-v1")

    results = retrieve(
        tmp_store, embedder, "dapagliflozin heart failure hospitalization",
        top_k=5, embed_model="fake-model-v1",
    )
    assert len(results) > 0
    for r in results:
        assert r.rank_sources, "every fused chunk must come from at least one ranking"
        assert set(r.rank_sources) <= {"bm25", "cosine"}


def test_topic_spread_warning_fires_for_scattered_embeddings(tmp_store):
    model = "fake-model-v1"
    tmp_store.insert_embedding(1, model, [1.0, 0.0, 0.0])
    tmp_store.insert_embedding(2, model, [0.0, 1.0, 0.0])
    tmp_store.insert_embedding(3, model, [0.0, 0.0, 1.0])
    chunks = [_chunk(1, "p1"), _chunk(2, "p2"), _chunk(3, "p3")]

    warning = topic_spread_warning(tmp_store, model, chunks)
    assert warning is not None
    assert "scattered" in warning


def test_topic_spread_warning_quiet_for_similar_embeddings(tmp_store):
    model = "fake-model-v1"
    tmp_store.insert_embedding(1, model, [1.0, 0.01, 0.0])
    tmp_store.insert_embedding(2, model, [0.99, 0.0, 0.01])
    chunks = [_chunk(1, "p1"), _chunk(2, "p2")]

    assert topic_spread_warning(tmp_store, model, chunks) is None


def test_topic_spread_warning_none_with_fewer_than_two_embeddings(tmp_store):
    chunks = [_chunk(1, "p1")]
    assert topic_spread_warning(tmp_store, "fake-model-v1", chunks) is None
