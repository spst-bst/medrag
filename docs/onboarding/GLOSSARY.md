# Glossary

Terms used throughout this course and the codebase, in roughly the order you'll meet them.

- **PMID** — PubMed ID. The unique identifier for a PubMed article; the citation unit for every
  answer medrag gives. `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`.
- **E-utilities** — NCBI's public API for PubMed (`esearch`, `efetch`). Requires an identifying
  email per their usage policy; see `medrag/ncbi.py`.
- **MeSH terms** — Medical Subject Headings. NCBI's controlled vocabulary tags for articles;
  stored on `Record.mesh_terms` but not currently used in retrieval.
- **Record** — one parsed PubMed article (`medrag/models.py`): title, abstract sections,
  journal, year, publication types, MeSH terms, DOI.
- **Chunk** — a retrievable slice of a `Record`'s abstract (`medrag/chunking.py`): either a whole
  labeled section (Background/Methods/...) or a ~250-word slice of an unlabeled abstract.
- **Embedding** — a vector representation of a chunk's text, produced by an embedding model
  (`nomic-embed-text` via Ollama by default) for semantic (cosine) search.
- **FTS5** — SQLite's full-text search extension. Powers keyword/BM25 search over chunk text
  (`chunks_fts` table in `medrag/store.py`).
- **BM25** — a classic keyword-ranking algorithm (term frequency, inverse document frequency,
  length normalization). SQLite FTS5 computes it for you; medrag just orders by it.
- **Cosine similarity** — a measure of how similar two vectors' *directions* are, regardless of
  magnitude. Used to rank chunks by embedding similarity to the query's embedding.
- **RRF (Reciprocal Rank Fusion)** — the algorithm `medrag/retrieval.py` uses to merge the BM25
  ranking and the cosine ranking into one score, based on each chunk's *rank* in each list rather
  than its raw score. See module 4.
- **Evidence boost** — a small multiplier applied to a chunk's fused score based on its
  publication type (higher for RCTs/meta-analyses/systematic reviews, lower for case reports).
- **RetrievedChunk** — a `Chunk` plus its article metadata and final retrieval `score`, the type
  that crosses from `retrieval.py` into the providers.
- **Provider** — an implementation of the `Generator` protocol (`medrag/providers/base.py`) that
  turns a question + retrieved chunks into an `Answer`. Two exist: `OllamaProvider` (local,
  default) and `ClaudeProvider` (cloud, opt-in).
- **PHI** — Protected Health Information (patient identifiers: names, SSNs, MRNs, DOBs...).
  `medrag/privacy.py`'s heuristic scan exists to catch *some* obvious PHI patterns before a cloud
  call — it is explicitly not comprehensive.
- **Cloud gate** — the `MEDRAG_ALLOW_CLOUD=1` + `--provider anthropic` double opt-in
  (`medrag/privacy.py: cloud_gate_error`) required before any Anthropic API call. See module 6.
- **Audit log** vs. **usage log** — `data/audit.jsonl` (what was sent, to whom, when) vs.
  `data/usage.jsonl` (token counts and estimated cost only). Separate files, separate purposes.
- **Citation enforcement** — `medrag/citations.py: enforce_citation_rules`, run after every
  provider call, that flags citations to PMIDs never retrieved and answers with no citations at
  all. Independent of what the model/API itself reports.
- **`hit_at_5`** — an eval metric: did retrieval surface at least one expected PMID in the top 5
  results, independent of what the LLM did with them. See module 9.
- **"Not found in the retrieved literature."** — the exact, literal string every provider must
  return when the retrieved excerpts don't answer the question. Checked by exact string match
  (`citations.py: is_not_found_reply`), not a fuzzy heuristic.
