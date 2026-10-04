# 3. Indexing — `medrag index`

## Why it matters

This module decides what a "unit of retrieval" is. Get the chunk boundaries wrong and no amount
of clever ranking in module 4 will save you — you can't retrieve a good answer out of a bad
chunk.

Once you've read the mental model below, [03a-indexing-workflow.md](03a-indexing-workflow.md) is
a literal, step-numbered trace of the same process with exact file:line references — useful when
you're debugging a specific `medrag index` run rather than building first-time intuition.

## Read this, in order

1. `medrag/chunking.py` — `chunk_record`. The whole file is 34 lines; read it completely.
2. `medrag/embeddings.py` — `OllamaEmbedder`, and the `vector_to_blob` / `blob_to_vector` /
   `cosine_similarity_matrix` helpers at the bottom.
3. `medrag/store.py` — the `SCHEMA` string first (lines 13–44), then `upsert_record`,
   `insert_chunk`, `chunk_ids_missing_embedding`, `insert_embedding`.
4. `medrag/indexer.py` — `index_records`. Only 44 lines; this is the orchestrator that ties the
   above three together.

## Mental model

**Chunking branches on whether the abstract has labeled sections** (`chunking.py:18`):
- Labeled (`Background`/`Methods`/`Results`/...): one chunk per section, however long.
- Unlabeled: one blob of text split every `WORDS_PER_CHUNK = 250` words.

Either way, every chunk's stored `text` is `header + "\n" + body` — the header
(`PMID | Title | Year | PubType [| Section]`) is baked into the text that gets embedded and
full-text-indexed, not stored separately. This means a chunk's embedding is influenced by its own
metadata string, and FTS5 keyword search can match on title words too. That's intentional, not an
accident — keep it in mind if a retrieval result ever surprises you.

**Indexing is idempotent at the record level, not the corpus level.**
`store.record_has_chunks(pmid)` (`store.py:83`) skips re-chunking a PMID that's already been
chunked — so re-running `medrag fetch` + `medrag index` on overlapping queries never duplicates
chunks. But there's no chunk-level dedup within a single `chunk_record` call, and no update path:
if you want to re-chunk a record (e.g. after a chunking logic change), you'd need to delete its
rows manually — there's no "reindex" command.

**Embeddings are keyed by model name, and stale ones are nuked on a model switch.**
`index_records` (`indexer.py:15-18`) compares the `model` argument against the
`embedding_model` value stored in the `meta` table. If they differ, it calls
`store.clear_embeddings()` — wiping *all* embeddings for *all* chunks — before indexing
continues. Switching `MEDRAG_EMBED_MODEL` and re-running `index` is how you'd migrate to a new
embedding model, but it means a full re-embed, not an incremental one.

**Vectors are raw bytes, not a vector extension.** `vector_to_blob`/`blob_to_vector`
(`embeddings.py:53-58`) just reinterpret a `float32` NumPy array as bytes and back. There's no
ANN index (no FAISS/HNSW) — `get_embedding_matrix` (`store.py:138`) pulls every vector for the
given model into one NumPy matrix, and cosine similarity (module 4) is a single matrix multiply
over all of them. Fine today; the first thing to revisit if the corpus grows past a few tens of
thousands of chunks.

**FTS5 is kept in sync with `chunks` manually.** Look at `insert_chunk` (`store.py:88-99`): it
inserts into `chunks` and then explicitly inserts the same rowid/text into `chunks_fts`. There's
no trigger — if you ever write a second code path that inserts into `chunks` directly, you'll
silently break keyword search for those rows until you add the matching `chunks_fts` insert.

## Exercise

Run `medrag index` on your module-1 data, then:
```bash
sqlite3 data/index.db "select pmid, section, length(text) from chunks limit 10;"
```
Find one labeled-section record and one unlabeled record in the output. For the unlabeled one,
confirm with `select count(*) from chunks where pmid='<that pmid>'` that longer abstracts produced
more chunks, consistent with `WORDS_PER_CHUNK`.

Then: temporarily change `WORDS_PER_CHUNK` to `50`, delete `data/index.db`, re-run `fetch`+`index`
(fetch will hit the cache, so it's fast), and observe the chunk count change. Revert the change
afterward — this was for observation, not a real tuning decision.

## Checkpoint

- What's baked into every chunk's `text`, and what two retrieval mechanisms does that affect?
- If you change `MEDRAG_EMBED_MODEL` and re-run `medrag index`, what happens to existing
  embeddings? What *doesn't* happen to existing chunks?
- Why would inserting into `chunks` without also inserting into `chunks_fts` be a silent bug
  rather than a loud one?
