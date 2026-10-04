# 1. Setup

## Why it matters

You can't build intuition for a retrieval pipeline by reading it — you have to watch it retrieve
something wrong, then go find out why.

## Prerequisites

- Python 3.10+ (repo has a `.venv` already — check `pyproject.toml` for the pinned deps)
- [Ollama](https://ollama.com) running locally, for the default local provider and embeddings
- A real email address for NCBI E-utilities (they ask for one so they can contact you if your
  usage is abusive — this is NCBI's policy, not medrag's)
- Optional: an `ANTHROPIC_API_KEY` if you want to exercise the cloud path later (module 6 first)

## Steps

```bash
cd /projects/local-agent/medrag
export MEDRAG_EMAIL=you@example.com

# pull the two Ollama models the default config expects (medrag/config.py)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

.venv/bin/medrag doctor
```

`doctor` (`medrag/doctor.py`, wired up in `cli.py: doctor()`) checks Ollama reachability, required
models, NCBI network access, `MEDRAG_EMAIL`, and the Anthropic opt-in state. Everything should
read `OK` except the Anthropic rows (expected — you haven't opted into cloud yet).

## Run the pipeline once, end to end

```bash
.venv/bin/medrag fetch "SGLT2 inhibitors heart failure" --max 50
.venv/bin/medrag index
.venv/bin/medrag ask "What does the evidence say about SGLT2 inhibitors in heart failure?" --show-evidence
```

Watch for:
- `fetch` prints how many records it saved — some PubMed results have no abstract and get
  dropped (`pubmed_xml.py` skips anything without `AbstractText`)
- `index` prints records/chunks/embeddings created — re-running it should create *zero* of all
  three (it's idempotent; you'll see why in module 3)
- `ask` prints an answer with `[PMID:...]` markers, a sources table, and (with `--show-evidence`)
  the literal quoted excerpts behind each citation

## Poke at the database directly

```bash
sqlite3 data/index.db ".tables"
sqlite3 data/index.db "select pmid, title from records limit 5;"
sqlite3 data/index.db "select count(*) from chunks;"
```

There's no ORM here — `store.py` is raw SQL. Seeing the actual rows now will make `store.py` read
like a tour of data you've already met, not abstract schema.

## Exercise

Run `medrag ask` with a question you're confident the fetched abstracts *don't* answer (e.g. ask
about a drug you didn't fetch anything about). Confirm you get back exactly
`"Not found in the retrieved literature."` — not a hedge, not a guess. This exact-string contract
matters; you'll rely on it again in modules 5 and 9.

## Checkpoint

- What does `medrag doctor` check that `medrag ask` doesn't check for you automatically?
- Why does re-running `medrag index` on the same data create 0 new chunks?
- What two environment variables did you set (or confirm) just to get `ask` working locally?
