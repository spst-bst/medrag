# 10. Capstone — ship one small feature, unsupervised

## Why it matters

Reading comprehension and contribution-readiness are different skills. This module is the
transition between them: one real, small, end-to-end feature, touching the same number of layers
a real PR would, with no module guiding you through which file to open next.

## Pick one (don't do all of them — pick the one that sounds most useful to actually have)

**A. `--since-year` filter on `ask`.** Add a CLI/API option that drops chunks whose record year
is older than a cutoff, applied in `retrieval.py: retrieve()` before (or after — your call, but
be able to justify it) the `top_k` slice. Needs: a `retrieval.py` change, a `cli.py` flag, a
`webapp.py` request field, and a test in `test_retrieval.py` proving records older than the
cutoff never appear even if they'd otherwise rank #1.

**B. A `medrag stats` command.** Print corpus size (records, chunks, embeddings), a breakdown of
`pub_types` counts, and year range, reading only from `data/index.db` via `store.py`. Needs: a new
`store.py` query method (or two), a new `cli.py` command, and a test using the `tmp_store`
fixture.

**C. Per-question cost cap enforcement, not just session cost.** Currently
`MEDRAG_MAX_SESSION_USD` only stops *further* calls after the total is exceeded (module 6). Add a
`MEDRAG_MAX_QUESTION_USD` that refuses to log a successful answer as billable if a single call's
estimated cost blows the per-question budget — decide for yourself whether that should block
before or after the call (you can't know the real cost before calling the API, so think about
what "refuse" can realistically mean here). Needs: `config.py`, `cli.py`/`webapp.py`, and tests in
`test_privacy_and_cost.py`.

## What "done" looks like, for any of them

- The change respects the module-6 safety/cost gate ordering if it touches the cloud path at all.
- `enforce_citation_rules` is still called if you touch provider code — never skip it.
- New tests run offline by default (module 8) — no new live network dependency without a
  `MEDRAG_LIVE*` gate.
- `.venv/bin/python -m pytest -q` passes.
- The README's `## Commands` or `## Using the Anthropic (cloud) provider` section is updated if
  you added or changed a user-facing flag or env var — don't let the docs drift on day one.

## Checkpoint (self-review before you call it done)

- Which existing module's "mental model" section most directly constrains your design choice
  here, and did you follow it or deliberately deviate? If you deviated, could you defend that in
  a PR description?
- Did you duplicate logic across `cli.py` and `webapp.py` the way module 7 said the codebase
  already does, or did you add a shared helper? Either is defensible — which did you pick, and
  why?
- What's the one test you'd want someone to run to convince themselves your feature actually
  works, not just that it doesn't crash?
