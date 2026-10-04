from __future__ import annotations

import numpy as np

from medrag.indexer import index_records
from tests.fakes import FakeEmbedder


def test_initial_indexing_creates_chunks_and_embeddings(tmp_store, sample_records):
    embedder = FakeEmbedder()
    stats = index_records(tmp_store, embedder, sample_records, model="fake-model-v1")

    assert stats["records_indexed"] == 4
    assert stats["chunks_created"] == 4 + 1 + 2 + 1  # sections: 4,1,2,1 for the 4 records with abstracts
    assert stats["embeddings_created"] == stats["chunks_created"]

    chunk_ids, matrix = tmp_store.get_embedding_matrix("fake-model-v1")
    assert len(chunk_ids) == stats["chunks_created"]
    assert matrix.shape[0] == len(chunk_ids)


def test_incremental_reindex_skips_existing(tmp_store, sample_records):
    embedder = FakeEmbedder()
    first = index_records(tmp_store, embedder, sample_records, model="fake-model-v1")
    assert first["chunks_created"] > 0

    second = index_records(tmp_store, embedder, sample_records, model="fake-model-v1")
    assert second["records_indexed"] == 0
    assert second["chunks_created"] == 0
    assert second["embeddings_created"] == 0


def test_embedding_model_change_rebuilds_embeddings(tmp_store, sample_records):
    embedder = FakeEmbedder()
    first = index_records(tmp_store, embedder, sample_records, model="fake-model-v1")
    total_chunks = first["chunks_created"]

    second = index_records(tmp_store, embedder, sample_records, model="fake-model-v2")
    # chunks already exist (not re-chunked), but embeddings must be rebuilt under the new model
    assert second["records_indexed"] == 0
    assert second["chunks_created"] == 0
    assert second["embeddings_created"] == total_chunks

    old_ids, old_matrix = tmp_store.get_embedding_matrix("fake-model-v1")
    assert old_ids == []
    new_ids, new_matrix = tmp_store.get_embedding_matrix("fake-model-v2")
    assert len(new_ids) == total_chunks


def test_get_embeddings_for_chunk_ids_filters_by_model_and_id(tmp_store):
    tmp_store.insert_embedding(1, "m1", [1.0, 2.0, 3.0])
    tmp_store.insert_embedding(2, "m1", [4.0, 5.0, 6.0])
    tmp_store.insert_embedding(3, "m2", [7.0, 8.0, 9.0])

    result = tmp_store.get_embeddings_for_chunk_ids([1, 2, 3], "m1")

    assert set(result.keys()) == {1, 2}
    assert np.allclose(result[1], [1.0, 2.0, 3.0])
    assert np.allclose(result[2], [4.0, 5.0, 6.0])


def test_get_embeddings_for_chunk_ids_empty_input(tmp_store):
    assert tmp_store.get_embeddings_for_chunk_ids([], "m1") == {}
