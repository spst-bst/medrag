# 2. Ingestion — `medrag fetch`

## Why it matters

Every downstream bug — bad citations, missing results, weird chunk boundaries — can originate
here if the source data is wrong. You need to know exactly what shape of data comes out before
you can debug anything that consumes it.

## Read this, in order

1. `medrag/models.py` — the `Record`, `Chunk`, and `RetrievedChunk` dataclasses. These three
   types are the vocabulary of the entire codebase. Read all 38 lines before anything else.
2. `medrag/ncbi.py` — `NCBIClient`: `esearch` (query → PMIDs) then `efetch_batch` (PMIDs → raw
   XML), batched at `EFETCH_BATCH_SIZE = 100`.
3. `medrag/pubmed_xml.py` — `parse_pubmed_xml`: raw PubMed XML → `list[Record]`.
4. Back to `ncbi.py`: `save_records_jsonl` / `load_records_jsonl` — the on-disk format
   (`data/records.jsonl`, one JSON object per line).

## Mental model

**Two NCBI calls, cached by content hash.** `_cache_key` (`ncbi.py:19`) hashes the full request
URL + params and stores the raw response bytes under `data/cache/<hash>.bin`. Re-running `fetch`
with the same query never re-hits the network — this is why the tests and your own repeated runs
during development don't need network. If you ever need to force a refetch, delete the matching
cache file (or `data/cache/` entirely).

**Politeness is enforced, not optional.** `RateLimiter` (`ratelimit.py`) caps requests to 3/sec
before every *uncached* NCBI call. `MEDRAG_EMAIL` is a hard requirement
(`config.py: medrag_email()` raises if unset) because NCBI's usage policy requires an identifying
contact for E-utilities traffic — this isn't medrag's own invention.

**Records without abstracts are silently dropped.** `parse_pubmed_xml` (`pubmed_xml.py:87-88`)
skips any `PubmedArticle` with no `AbstractText` sections. If a user asks "why did I get fewer
records than PMIDs returned?", this is almost always the answer — check before assuming a bug.

**Abstracts can be labeled or unlabeled.** `_parse_abstract_sections` returns a list of
`(label, text)` tuples. Some journals structure abstracts (`Background`, `Methods`, `Results`,
`Conclusions` as separate labels); others give one blob with `label=None`. This single fact drives
a branch in the *next* module (`chunking.py`) — note it now so it's not a surprise there.

## Exercise

Using `tests/fixtures/sample_pubmed.xml` as a reference for the XML shape, write a short throwaway
script (keep it outside the repo, e.g. in your scratch space) that:
1. Calls `parse_pubmed_xml` on that fixture.
2. Prints each record's PMID, year, and whether its abstract has labeled sections or not.

Then delete the script — it was for your own understanding, not for the repo.

## Checkpoint

- If `medrag fetch "x" --max 200` returns 200 PMIDs from `esearch`, is it guaranteed you'll get
  200 `Record`s in `records.jsonl`? Why or why not?
- Where exactly does caching happen — before or after the rate limiter?
- What does a `Record.abstract_sections` entry with `label=None` mean, and what produces one?
