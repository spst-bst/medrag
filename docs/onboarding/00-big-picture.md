# 0. Big picture

## Why it matters

Every module after this one is "zoom into one box." If the box diagram isn't in your head first,
the individual files will feel like trivia.

## The six commands

```
medrag fetch "<query>"   NCBI PubMed  --->  data/records.jsonl
medrag index              records.jsonl --->  data/index.db (chunks + embeddings)
medrag ask "<question>"   index.db  --->  retrieved chunks  --->  LLM  --->  cited answer
medrag eval questions.json  runs `ask`'s pipeline over a batch, scores it
medrag doctor              environment/connectivity checks
medrag ui                  FastAPI wrapper around the same `ask` pipeline
```

Everything lives under `medrag/` as plain modules (no app framework, no plugin system). `cli.py`
and `webapp.py` are both thin callers into the same core — neither contains pipeline logic.

## Data flow for `medrag ask`

```
question
   |
   v
[retrieval.py: retrieve()]
   - BM25 full-text search (store.fts_search)       -----> ranking A
   - cosine similarity over embeddings (store.get_embedding_matrix) -----> ranking B
   - reciprocal rank fusion merges A + B
   - evidence_boost_multiplier reweights by pub type (RCT/meta-analysis up, case report down)
   - top_k RetrievedChunk objects
   v
[providers/*.py: generate()]
   - OllamaProvider (local, default) or ClaudeProvider (cloud, opt-in twice)
   - same SYSTEM_PROMPT (providers/base.py): answer only from excerpts, cite every claim,
     say "Not found in the retrieved literature." when the excerpts don't answer it
   v
[citations.py: enforce_citation_rules()]
   - runs on EVERY provider's output, regardless of what the model claims
   - flags citations to PMIDs that were never retrieved
   - flags answers with no citations at all
   v
Answer (text, status, citations, warnings) -> printed/returned, logged if cloud
```

The single most important design fact in this codebase: **the LLM is never trusted to self-police
citations.** `citations.py` re-checks every citation against the actual retrieved set after the
fact. Keep that in mind — it explains why `enforce_citation_rules` is called identically from
both `ollama_provider.py` and `claude_provider.py` instead of being "the provider's problem."

## Storage

One SQLite file, `data/index.db` (`medrag/store.py`), holds three things:
- `records` — one row per PubMed article (title, year, journal, MeSH terms...)
- `chunks` — pieces of an abstract, plus an FTS5 virtual table (`chunks_fts`) for keyword search
- `embeddings` — one vector per chunk per embedding model, stored as a raw `float32` BLOB

There's no vector database. Cosine similarity is computed in Python/NumPy over whatever fits in
the `embeddings` table (`embeddings.py: cosine_similarity_matrix`). That's a deliberate scale
trade-off, not an oversight — fine for a few thousand abstracts, not for millions.

## Exercise

Without opening any other file, sketch the diagram above from memory, labeling which file owns
each box. Then open [module 1](01-setup.md) and get it running.

## Checkpoint

- Name the six `medrag` commands and, for each, the one file that contains its actual logic
  (not just the CLI wiring).

  <details><summary>Answer</summary>

  - `fetch` → `medrag/ncbi.py` (`NCBIClient.fetch_all`, plus `pubmed_xml.py` for parsing)
  - `index` → `medrag/indexer.py` (`index_records`)
  - `ask` → split across `medrag/retrieval.py` (`retrieve`) and `medrag/providers/*.py`
    (`generate`) — there's no single "ask logic" file, which is itself worth noticing
  - `eval` → `medrag/eval_runner.py` (`run_eval`)
  - `doctor` → `medrag/doctor.py` (the `check_*` functions)
  - `ui` → `medrag/webapp.py` (the FastAPI routes); `cli.py`'s `ui()` itself only calls
    `uvicorn.run(...)`
  </details>

- Why is it fair to call `cli.py` and `webapp.py` "thin," and what would it mean for that to stop
  being true?

  <details><summary>Answer</summary>

  Neither file implements an algorithm — they parse input (Typer args / a Pydantic request body),
  call into `config`, `ncbi`, `indexer`, `retrieval`, `providers/*`, and `audit`, and format the
  result (console printing vs. JSON response). It would stop being true the moment either file
  grew logic that isn't also reachable from the other caller — e.g. a retrieval filter written
  inline inside `cli.py: ask()` instead of in `retrieval.py`. That logic would then only work from
  the CLI, not the web UI or the eval harness, which is exactly the bug this "thin shell" rule is
  meant to prevent.
  </details>

- Which two ranking signals get merged by RRF, and where do they each come from?

  <details><summary>Answer</summary>

  BM25 keyword-search ranking from SQLite FTS5 (`store.fts_search`) and cosine-similarity ranking
  over embeddings (`store.get_embedding_matrix` + `cosine_similarity_matrix`). Both are computed
  inside `retrieval.py: retrieve()` and merged by `reciprocal_rank_fusion`.
  </details>

- What's the one function that both providers call before returning an `Answer`, and why does it
  exist outside the provider classes?

  <details><summary>Answer</summary>

  `enforce_citation_rules` in `citations.py`. It lives outside both provider classes so the
  verification can't silently diverge per-provider (or be skipped by a future third provider) —
  it's an independent trust boundary checking the model's claims against the actual retrieved
  set, not a detail either provider gets to own or reimplement.
  </details>

- Where are vectors stored, and what Python type are they decoded into for search?

  <details><summary>Answer</summary>

  Stored as raw `float32` BLOBs in the `embeddings` table (`medrag/store.py`). Decoded via
  `blob_to_vector` (`embeddings.py`) into a NumPy `ndarray`, then stacked into a matrix for the
  cosine-similarity matrix multiply.
  </details>

- What's the scale trade-off being made by computing cosine similarity in NumPy instead of using
  a vector database, and at roughly what point would you expect it to start mattering?

  <details><summary>Answer</summary>

  Every query does a dense, linear scan — one matrix multiply against *every* stored embedding for
  the model, with no approximate-nearest-neighbor index (no FAISS/HNSW) to skip unpromising
  vectors. That's simple and exact, but memory and latency both grow linearly with corpus size.
  The module text's own framing ("fine for a few thousand abstracts, not for millions") is the
  right order of magnitude to reason from — expect it to start being noticeable somewhere in the
  tens-of-thousands-of-chunks range, well before "millions."
  </details>
