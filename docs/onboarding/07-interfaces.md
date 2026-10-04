# 7. Interfaces — CLI and web UI

## Why it matters

By now you've read every piece of actual pipeline logic. This module is deliberately short,
because its lesson is structural: **these two files should stay thin.** If you ever find yourself
adding retrieval or citation logic inside `cli.py` or `webapp.py` instead of in the modules you've
already read, that's a sign the change belongs somewhere else.

[04a-ask-workflow.md](04a-ask-workflow.md) is the literal step-by-step trace of everything
`cli.py: ask()` calls, in order — a good companion once you've read this module.

## Read this

1. `medrag/cli.py` — skim the whole file; it's just Typer commands calling into `config`,
   `ncbi`, `indexer`, `retrieval`, `providers/*`, `audit`, in roughly the same order every time.
2. `medrag/webapp.py` — compare `api_ask` to `cli.py: ask()`. Same gate → scan → retrieve →
   generate → log sequence, adapted to return a Pydantic response instead of printing to a
   console.
3. `medrag/static/index.html` — just enough to see it's a single static page calling
   `/api/ask` and `/api/doctor`; no build step, no framework.

## Mental model

**Both interfaces duplicate the same call sequence rather than sharing a function.** `cli.py:
ask()`, `webapp.py: api_ask()`, and `eval_runner.py: run_eval()` each independently do:
gate check → PHI scan → retrieve → generate → enforce-citations (inside generate) → log if cloud.
This is real duplication — there's no shared `answer_question()` helper. Worth knowing before you
go looking for one, and worth considering if you're the one adding a fourth caller: would
extracting a shared function reduce risk of one caller forgetting a step, or is three call sites
not yet enough to justify the abstraction? There's no single right answer here; form your own
view and see if it matches how the codebase already leans (toward leaving it duplicated, so far).

**The web UI has no separate state.** Per the README: "The UI only reads/writes the same local
`data/` files as the CLI." `webapp.py` opens the same `Store(config.INDEX_DB_PATH)` and writes to
the same audit/usage logs. There's no session concept, no user accounts, no separate database.

**No auth, bound to localhost by default, and the README says why.** `ui()` in `cli.py` defaults
`--host 127.0.0.1`. The README explicitly warns against changing that to `0.0.0.0` on an
untrusted network, because `api_ask` has no authentication at all — anyone who can reach the port
can ask questions and (if cloud is enabled on that machine) trigger billed API calls.

## Exercise

Add a `--provider` default override test: run `medrag ask "..."` with `MEDRAG_PROVIDER=anthropic`
set in the shell but *no* `--provider` flag, and confirm from `cli.py: ask()`'s first line
(`provider_name = provider or os.environ.get("MEDRAG_PROVIDER", "ollama")`) that the env var wins
when the flag is omitted — then confirm passing `--provider ollama` explicitly overrides the env
var. This two-line precedence rule is easy to miss when debugging "why did this hit the cloud
provider when I didn't pass `--provider anthropic`."

## Checkpoint

- Name the five-step sequence (gate → ... → log) that all three callers (`cli.py`, `webapp.py`,
  `eval_runner.py`) independently implement.
- What does the web UI do differently from the CLI in terms of where data is stored?
- What's the precedence between `--provider` and `MEDRAG_PROVIDER` when both are set?
