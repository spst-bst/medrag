# Ask workflow — a literal step-by-step trace

This is a companion to [04-retrieval.md](04-retrieval.md), [05-answering.md](05-answering.md),
[06-safety-and-cost.md](06-safety-and-cost.md), and [07-interfaces.md](07-interfaces.md). Those
modules explain the *mental model* behind retrieval, answering, safety gates, and the CLI/web
split. This doc is the literal call-by-call trace of what happens when you run `medrag ask` —
useful when you're debugging a specific run rather than building first-time intuition. All line
numbers refer to the current state of the files after the streaming/debug-retrieval/topic-spread
enhancements.

## Precondition

`data/index.db` must already have chunks and embeddings (written by `medrag index`). If it
doesn't, step 5 below returns an empty list and the command exits early.

## Step 1 — Resolve the provider

```python
provider_name = provider or os.environ.get("MEDRAG_PROVIDER", "ollama")
```
(`cli.py:132`) The `--provider` flag wins if passed; otherwise `MEDRAG_PROVIDER` is checked;
otherwise it defaults to `"ollama"`. Nothing else has happened yet — no store opened, no network
call made.

## Step 2 — Cloud gate + PHI scan (only if `provider_name == "anthropic"`)

```python
gate_err = cloud_gate_error()
if gate_err: ... raise typer.Exit(1)
phi_hits = scan_for_phi(question)
if phi_hits and not i_confirm_no_phi: ... raise typer.Exit(1)
```
(`cli.py:134-144`) Two independent checks, in this exact order, **before any store or client is
constructed**:
1. `cloud_gate_error()` (`privacy.py:38-50`) — requires `MEDRAG_ALLOW_CLOUD=1`. The `--provider
   anthropic` flag alone is never sufficient.
2. `scan_for_phi()` (`privacy.py:23-35`) — regex heuristic for SSNs/emails/phones/MRN-DOB labels;
   blocked unless `--i-confirm-no-phi` is also passed.

For `provider_name == "ollama"` (the default), this entire block is skipped.

## Step 3 — Open the store, construct the embedder

```python
store = Store(config.INDEX_DB_PATH)
embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)
```
(`cli.py:146-147`) Note this embedder is constructed **regardless of which answer provider you
chose** — even `--provider anthropic` still needs a local Ollama embedding model to embed the
query for the cosine-similarity half of retrieval. If Ollama isn't running, this is where an
otherwise-cloud-only `ask` call would fail.

## Step 4 — Retrieve: two rankings, fused, boosted, tagged

```python
chunks = retrieve(store, embedder, question, top_k=top_k, embed_model=..., use_evidence_boost=...)
```
(`cli.py:148-155`, logic in `retrieval.py:43-100`). Inside `retrieve()`:

1. **BM25 ranking** — `store.fts_search(query, bm25_limit=50)` (`retrieval.py:53`, SQL in
   `store.py:161-176`) hits the `chunks_fts` virtual table, ordered by BM25 score.
2. **Cosine ranking** — `embedder.embed_query(question)` (one HTTP call to Ollama's `/api/embed`
   with the `"search_query: "` prefix), then `store.get_embedding_matrix(embed_model)` pulls
   *every* stored vector for that model into a NumPy matrix, and `cosine_similarity_matrix()`
   ranks by similarity (`retrieval.py:56-62`).
3. **Fusion** — `reciprocal_rank_fusion([bm25_ranking, cosine_ranking])` (`retrieval.py:64`)
   merges the two by rank position, not raw score (see module 4 for why).
4. **Per-chunk loop** (`retrieval.py:69-97`): for each fused chunk id, `store.get_chunk()` pulls
   the row, `evidence_boost_multiplier()` reweights by pub type if enabled, and — new in the
   debug-retrieval enhancement — `rank_sources` is tagged (`["bm25"]`, `["cosine"]`, or both)
   based on membership in `bm25_set`/`cosine_set` (`retrieval.py:66-67, 78-82`).
5. **Sort + slice** to `top_k` (`retrieval.py:99-100`). This is also where the evidence boost can
   change *membership* in the final list, not just order within it — see module 4.

## Step 5 — Topic-spread diagnostic (before the store closes)

```python
spread_warning = topic_spread_warning(store, config.embed_model(), chunks) if len(chunks) >= 2 else None
store.close()
```
(`cli.py:156-159`) `topic_spread_warning()` (`retrieval.py:103-136`) looks up the actual embedding
vectors for just these retrieved chunks via `store.get_embeddings_for_chunk_ids()`
(`store.py:116-124`), computes their average pairwise cosine similarity, and returns a warning
string if that average falls below `TOPIC_SPREAD_WARNING_THRESHOLD` (currently `0.82`, calibrated
against two real examples — see the comment at `retrieval.py:13-19`). This has to run **before**
`store.close()` — it's the only reason the store stays open this long after retrieval.

If fewer than 2 chunks were retrieved, this is skipped entirely (`None`) — there's no "spread" to
measure with a single chunk.

## Step 6 — Guard: no chunks retrieved

```python
if not chunks:
    console.print("[yellow]No chunks retrieved...[/yellow]")
    raise typer.Exit(1)
```
(`cli.py:161-163`) Exits here if the index is empty or the query matched nothing in either
ranking.

## Step 7 — Construct the provider

```python
if provider_name == "anthropic":
    ... gen = ClaudeProvider()
elif provider_name == "ollama":
    gen = OllamaProvider()
```
(`cli.py:165-178`) For `anthropic`, this also prints the "sending N abstracts to Anthropic's API"
notice and the server-side-fallback notice if enabled — the last visible reminder that data is
about to leave the machine, right before it actually does.

## Step 8 — Generate: streaming (Ollama) vs. blocking (Anthropic)

This branches on provider, and it's the biggest structural difference from before the streaming
enhancement:

**Ollama** (`cli.py:180-189`):
```python
console.print(f"\n[bold]Answer[/bold] (provider=ollama model={gen.model})\n")

def _on_progress(piece: str) -> None:
    console.print(piece, end="", markup=False, highlight=False)

answer = gen.generate(question, chunks, on_progress=_on_progress)
console.print(f"\n\n[dim](status={answer.status} seconds={answer.seconds:.1f})[/dim]")
```
The header prints *before* generation starts (so you see context immediately), and
`_on_progress` is wired into `OllamaProvider.generate()`'s streaming loop
(`ollama_provider.py:55-67`) — each token prints to the terminal the moment Ollama emits it, with
`markup=False` so literal `[PMID:...]` text in the stream is never misinterpreted as a Rich style
tag.

**Anthropic** (`cli.py:190-197`): unchanged from before — a blocking call under a spinner, then
the full answer text printed at once. `ClaudeProvider.generate()` has no `on_progress` wiring, so
there's nothing to stream even if this branch tried to.

Inside *either* provider, before `generate()` returns:
- Citations get built — either regex-extracted `[PMID:...]` markers (Ollama,
  `citations.py:9-10`) or structured citation objects from the Anthropic API with verbatim-quote
  verification (`claude_provider.py:205-220`).
- `enforce_citation_rules(answer, retrieved_pmids)` (`citations.py`) runs as the **last** step
  inside both providers — flagging any cited PMID that wasn't actually retrieved, and flagging a
  non-"not found" answer with zero citations.

## Step 9 — Print warnings, then the topic-spread note

```python
if answer.warnings: ... "Warnings:" ...
if spread_warning: ... "Note:" ...
```
(`cli.py:199-205`) These are deliberately two separate, differently-colored blocks: `Warnings`
(red) come from citation enforcement — something wrong with *this specific answer*. `Note`
(yellow) comes from the topic-spread heuristic — a property of *what got retrieved*, independent
of whether the model's citations were valid. Don't conflate them when reading output.

## Step 10 — Sources table, with the Cited column

```python
cited = get_cited_pmids(answer)
_print_sources(chunks, cited)
```
(`cli.py:207-208`) `get_cited_pmids()` (`citations.py`) is the same structured-citations-else-regex
logic `enforce_citation_rules` uses internally, exposed as a shared helper. `_print_sources()`
(`cli.py:70-89`) dedupes chunks by PMID and marks each row `yes`/blank in the `Cited` column —
letting you see at a glance which retrieved sources the model actually drew from versus which
were retrieved but unused.

## Step 11 — `--show-evidence`

```python
if show_evidence:
    quoted = [c for c in answer.citations if c.quote]
    if quoted: ... print quotes ...
    elif answer.provider != "anthropic": ... print "has no effect" note ...
```
(`cli.py:210-221`) Only Anthropic's structured citations ever carry a `.quote` — Ollama's
citations are always `Citation(pmid=p, quote=None)` (`ollama_provider.py:82`). So with
`--provider ollama`, this block always falls into the explanatory note rather than silently
printing nothing.

## Step 12 — `--debug-retrieval`

```python
if debug_retrieval:
    _print_retrieval_debug(chunks)
```
(`cli.py:223-224`) `_print_retrieval_debug()` (`cli.py:92-107`) prints each chunk's id, fused
score, `rank_sources` (`"bm25"`, `"cosine"`, or `"bm25+cosine"`), and section — the same
per-chunk data the topic-spread heuristic and RRF math operate on, surfaced directly instead of
requiring a temporary `print()` in `retrieval.py`.

## Step 13 — Cost estimate, audit/usage logging, budget check (Anthropic only)

```python
if answer.provider == "anthropic":
    cost = config.estimate_cost_usd(...)
    ... print tokens/cost ...
    log_audit(...); log_usage(...)
    if cost > max_usd: raise typer.Exit(1)
```
(`cli.py:226-250`) This entire block is skipped for `ollama`. For `anthropic`, it:
1. Estimates cost from the static `PRICES_PER_MILLION_TOKENS` table (`config.py`).
2. Writes to `data/audit.jsonl` (question text + PMIDs sent + usage — the compliance trail) and
   `data/usage.jsonl` (cost only, no question text) — see module 6 for why these are separate
   files.
3. Checks `MEDRAG_MAX_SESSION_USD` **after** this call already happened — it can only stop the
   *next* call, not this one.

## The shape of a typical run

| Provider | Streams to terminal? | `--show-evidence` does anything? | Cost/audit logged? |
|---|---|---|---|
| `ollama` (default) | yes, token by token | no — prints an explanatory note instead | no |
| `anthropic` | no — blocking spinner, then full text | yes, if the model cited anything | yes, always |

## Verify it yourself

```bash
medrag ask "Why heart failure?" --show-evidence --debug-retrieval
```
On a broad/scattered question against a narrow corpus, you should see: streamed text, a `Cited`
column distinguishing used vs. unused sources, a `--show-evidence has no effect` note (Ollama),
a `Retrieval debug` table showing `bm25+cosine` for chunks that both rankings agreed on, and —
depending on how scattered the retrieved topics actually are relative to your embedding model's
baseline similarity — possibly a topically-scattered `Note`.
