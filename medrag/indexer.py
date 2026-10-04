from __future__ import annotations

from medrag.chunking import chunk_record
from medrag.embeddings import Embedder
from medrag.models import Record
from medrag.store import Store


def index_records(store: Store, embedder: Embedder, records: list[Record], model: str) -> dict:
    """Chunk new records, then embed any chunks missing an embedding for `model`.

    Rebuilds (clears) embeddings if the stored embedding model differs from `model`.
    Returns stats: {"records_indexed", "chunks_created", "embeddings_created"}.
    """
    stored_model = store.get_meta("embedding_model")
    if stored_model is not None and stored_model != model:
        store.clear_embeddings()
    store.set_meta("embedding_model", model)

    records_indexed = 0
    chunks_created = 0
    for record in records:
        store.upsert_record(record)
        if store.record_has_chunks(record.pmid):
            continue
        for chunk in chunk_record(record):
            store.insert_chunk(chunk)
            chunks_created += 1
        records_indexed += 1

    missing_ids = store.chunk_ids_missing_embedding(model)
    embeddings_created = 0
    if missing_ids:
        texts = [store.get_chunk(cid)["text"] for cid in missing_ids]
        vectors = embedder.embed_documents_batched(texts)
        for chunk_id, vector in zip(missing_ids, vectors):
            store.insert_embedding(chunk_id, model, vector)
            embeddings_created += 1

    return {
        "records_indexed": records_indexed,
        "chunks_created": chunks_created,
        "embeddings_created": embeddings_created,
    }
