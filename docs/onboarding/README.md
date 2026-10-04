# medrag onboarding course

Welcome. This is a self-paced course for getting productive on **medrag** — a local-first,
evidence-citing PubMed Q&A tool. It's a small codebase (~3,200 lines of Python, no framework
magic), so the goal isn't "cover everything," it's "read every file once, with a reason to care
about each one."

Written for a developer who knows Python but has never seen this repo, and may not have touched
RAG (retrieval-augmented generation) pipelines before. No prior medical or ML background assumed.

## How to use this course

Each module has:
- **Why it matters** — what breaks or becomes unreadable if you skip this
- **Read this** — exact files/functions, in order
- **Mental model** — the one or two ideas that make the code obvious once you have them
- **Exercise** — something to run or change, so you build muscle memory, not just familiarity
- **Checkpoint** — questions you should be able to answer before moving on

Do them in order. Later modules assume earlier ones. Budget roughly a day each for modules 2–7
(less for the rest) if you're also context-switching with other work; faster if this is your main
focus.

## Course map

| # | Module | Covers |
|---|--------|--------|
| 0 | [Big picture](00-big-picture.md) | The five-command pipeline, end to end |
| 1 | [Setup](01-setup.md) | Get it running on your machine |
| 2 | [Ingestion](02-ingestion.md) | `medrag fetch` — PubMed → local records |
| 3 | [Indexing](03-indexing.md) ([step-by-step trace](03a-indexing-workflow.md)) | `medrag index` — chunking, embeddings, SQLite |
| 4 | [Retrieval](04-retrieval.md) ([step-by-step trace](04a-ask-workflow.md)) | Hybrid search: BM25 + cosine + RRF |
| 5 | [Answering](05-answering.md) | Providers, prompts, citation enforcement |
| 6 | [Safety & cost](06-safety-and-cost.md) | The PHI heuristic, cloud gate, audit log |
| 7 | [Interfaces](07-interfaces.md) | CLI and web UI — thin shells over the same core |
| 8 | [Testing](08-testing.md) | How this repo tests without a network or a model |
| 9 | [Eval harness](09-eval.md) | Measuring answer quality, not just "it runs" |
| 10 | [Capstone](10-capstone.md) | Ship one real, small feature unsupervised |

[Glossary](GLOSSARY.md) has one-line definitions for every acronym and term of art used below
(PMID, RRF, BM25, PHI, FTS5...). Keep it open in a tab.

## Before you start

Read the project [README.md](../../README.md) once, top to bottom. It's short, and it already
tells you the shape of the tool: fetch → index → ask, two providers, and a research-support
disclaimer that shows up throughout the code, not just the docs. This course assumes you've done
that.

## Non-negotiable context

This is explicitly **not** a tool for handling real patient data against the cloud provider
without compliance sign-off, and it is **not** a source of medical advice. That constraint shapes
real design decisions you'll read in [module 6](06-safety-and-cost.md) — it isn't boilerplate.
