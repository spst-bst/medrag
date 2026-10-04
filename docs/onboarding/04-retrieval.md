# 4. Retrieval — turning a question into chunks

## Why it matters

This is the file most likely to need tuning as the corpus grows or answer quality drifts, and
it's the one place in the codebase doing something mathematically non-obvious (reciprocal rank
fusion). It's also small — 82 lines — so there's no excuse to skim it.

[04a-ask-workflow.md](04a-ask-workflow.md) traces this module's role inside the full `medrag ask`
call, step by step with exact file:line references — read it after this module and modules 5–7
if you want the literal call sequence rather than the mental model.

## Read this

`medrag/retrieval.py`, start to finish. It's one file, no sub-imports to chase except
`config.py`'s `EVIDENCE_PREFERRED_TYPES` / `EVIDENCE_DEPRIORITIZED_TYPES` / `RRF_K`.

## Mental model

**Two independent rankings, fused, not blended.** `retrieve()` builds:
- `bm25_ranking`: keyword search via SQLite FTS5 (`store.fts_search`), ordered by BM25 score.
- `cosine_ranking`: every chunk's embedding compared to the query's embedding via
  `cosine_similarity_matrix`, ordered descending.

Neither ranking's raw *score* survives into the final result — only each chunk's *position* in
each list matters. That's what makes this reciprocal **rank** fusion rather than a weighted
average of scores: BM25 scores and cosine similarities live on incomparable scales, so averaging
them directly would be meaningless. Converting both to "rank in this list" first is what makes
combining them valid.

**The RRF formula** (`reciprocal_rank_fusion`, lines 17–22):
```
score(chunk) = sum over each ranking it appears in of  1 / (k + rank + 1)
```
`k = RRF_K = 60` (`config.py`) dampens how much rank 1 dominates rank 2 — a chunk at rank 0 in one
list scores `1/61`; at rank 0 in *both* lists, `2/61`. A chunk that's merely decent in both
rankings can out-score a chunk that's #1 in only one. If a user ever asks "why wasn't the single
best keyword match the top result?", this is why — check whether it also showed up in the cosine
ranking.

**Evidence boost is a final reweighting, not a ranking signal.** `evidence_boost_multiplier`
(lines 25–31) multiplies the fused score by 1.1 for meta-analyses/systematic reviews/RCTs, 0.9 for
case reports, 1.0 otherwise — applied *after* fusion, and skippable via `use_evidence_boost=False`
(the CLI's `--no-evidence-boost` flag). Note *where* it happens in `retrieve()`: on every fused
candidate, before the final `sort()` and `[:top_k]` slice — not just on chunks that already made
the cut. A borderline chunk's boost can pull it into the top_k, or a deprioritized chunk's penalty
can push it out. This isn't purely a tie-breaker among results you'd have gotten anyway.

**No embeddings yet doesn't crash — it just skips that half.** If `get_embedding_matrix` returns
an empty matrix (`chunk_ids` empty), `cosine_ranking` stays `[]` and fusion runs on BM25 alone.
Useful to know when debugging a freshly-fetched, not-yet-indexed corpus, or when embeddings got
cleared by a model switch (module 3) mid-session.

## Exercise

Pick a question you know the answer to from your fetched corpus. Run it with `--show-evidence`
twice: once normally, once with `--no-evidence-boost`. If the source list differs, look up the
`pub_types` of whichever chunk moved and connect it to `evidence_boost_multiplier`.

Then, by hand, compute the RRF score for a chunk that's rank 2 (0-indexed) in the BM25 list and
rank 0 in the cosine list, with `k=60`. Confirm your arithmetic against what
`reciprocal_rank_fusion` would produce by adding a `print(fused)` temporarily in `retrieve()` —
remove the print afterward.

## Checkpoint

- Why does RRF use rank position instead of raw BM25/cosine scores?
- A chunk appears in the BM25 top-50 but not in the cosine top-50 (or embeddings don't exist for
  it yet). Does it still get a fused score? From which ranking(s)?
- Does the evidence boost only reorder the final top_k, or can it change *which* chunks make the
  top_k cut at all? Point to the line in `retrieve()` that settles this.
