# medrag

Local-first, evidence-citing literature Q&A proof of concept. Fetches PubMed abstracts, indexes
them locally, and answers research questions with a pluggable answer model (local Ollama by
default, Anthropic API opt-in), citing the PMID for every claim.

**This is research support only, never clinical advice.**

New to this codebase? Start with the [onboarding course](docs/onboarding/README.md) — a
self-paced, module-by-module tour of the architecture, retrieval, providers, and safety rails.

## Setup

```
export MEDRAG_EMAIL=you@example.com   # required by medrag fetch for NCBI E-utilities
.venv/bin/medrag doctor
```

## Commands

- `medrag fetch "<query>" --max 200`
- `medrag index`
- `medrag ask "<question>" [--top-k 6] [--provider ollama|anthropic] [--show-evidence] [--debug-retrieval]`
- `medrag eval eval/questions.json [--providers ollama,anthropic] [--save eval/compare.md]`
- `medrag doctor`
- `medrag ui [--host 127.0.0.1] [--port 8000]` — local web UI for asking questions in a browser

With `--provider ollama` (the default), `ask` streams the answer to the terminal as it's
generated rather than printing it all at once. `--show-evidence` only prints verbatim quotes with
`--provider anthropic` — Ollama citations are text markers, not structured quotes, so the CLI
tells you that instead of silently printing nothing. `--debug-retrieval` prints each retrieved
chunk's id, fused score, and whether it came from the BM25 ranking, the cosine ranking, or both —
useful when an answer looks like unrelated facts stitched together. When that happens, `ask` may
also print a `Note:` about the retrieved chunks looking topically scattered — a heuristic, not a
guarantee, and usually a sign the question was broader than the corpus can answer precisely.

## Web UI

```
.venv/bin/medrag ui
```

Starts a local server at `http://127.0.0.1:8000` (override with `--host`/`--port`). Open it in a
browser to ask questions without the CLI:

- Type a question, pick a provider (`ollama` or `anthropic`), set top-k, and hit **Ask**.
- Answers render with clickable `[PMID:...]` citations and a sources table linking to PubMed.
- Picking `anthropic` reveals the same PHI self-confirmation checkbox the CLI's
  `--i-confirm-no-phi` flag provides, and is still blocked by `MEDRAG_ALLOW_CLOUD=1` below.
- A collapsible **Setup status** panel at the bottom runs the same checks as `medrag doctor`.
- The UI only reads/writes the same local `data/` files as the CLI — no separate state.

The server binds to `127.0.0.1` by default (not exposed to your network). It has no
authentication, so don't change `--host` to `0.0.0.0` on an untrusted network.

## Using the Anthropic (cloud) provider

The cloud provider is opt-in **twice**:

1. `--provider anthropic` (or `MEDRAG_PROVIDER=anthropic`)
2. `MEDRAG_ALLOW_CLOUD=1`

Every cloud call sends your question and the retrieved public PubMed abstracts to Anthropic's
API — nothing else leaves this machine. The API key must be set in `ANTHROPIC_API_KEY` in your
shell environment; it is never printed, logged, or written to any file.

### PHI guard (heuristic only)

Before a cloud call, `medrag` scans the question for obvious identifiers (SSN-like numbers,
emails, phone numbers, MRN/DOB-style labels, dates near "born"/"DOB") and refuses unless you pass
`--i-confirm-no-phi`. **This is a heuristic and cannot detect patient names or other identifiers.**
Real patient data must never be sent to a cloud provider unless your organization's compliance
team has approved it.

### Cost

Every cloud call costs money. Token usage and an estimated cost are printed after each answer and
logged to `data/usage.jsonl`. Set `MEDRAG_MAX_SESSION_USD` to stop before exceeding a budget.
`data/audit.jsonl` logs every cloud call (timestamp, provider, model, question, PMIDs sent, usage) —
`data/` is gitignored.

## Tests

```
.venv/bin/python -m pytest -q
```

Offline tests use fakes only (no network, no Ollama, no Anthropic API calls). Real-service tests
are skipped unless `MEDRAG_LIVE=1` (Ollama/NCBI) or `MEDRAG_LIVE_ANTHROPIC=1` (cloud, costs money).
