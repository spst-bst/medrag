# 9. Eval harness — measuring answer quality, not just "it runs"

## Why it matters

Tests (module 8) prove the code behaves correctly given fixed inputs. The eval harness answers a
different question: given this corpus and these real questions, is the *pipeline* — retrieval +
prompting + a specific model — actually any good? That's a question tests can't answer, and one
you'll be asked whenever someone proposes a prompt or retrieval change.

## Read this, in order

1. `eval/questions.json` — the question set. Note the two kinds of entries: `answerable: true`
   with `expected_pmids` (auto-suggested, flagged "needs review" — treat as approximate, not
   ground truth), and `answerable: false` (questions the corpus should *not* be able to answer,
   e.g. unrelated topics like acupuncture for migraine against a SGLT2-inhibitor corpus).
2. `medrag/eval_runner.py` — `run_eval`, start to finish.
3. `eval/compare.md` — a previously generated comparison file. Open it to see the actual output
   shape before you generate your own.

## Mental model

**Four metrics, each measuring a different failure mode:**
- `hit_at_5` — of the answerable questions with known expected PMIDs, did *retrieval* (not the
  LLM) surface at least one expected PMID in the top 5? This isolates retrieval quality from
  answer quality — a provider can't fix a retrieval miss.
- `valid_citation_rate` — fraction of answers with zero `INVALID CITATION` warnings
  (module 5's `enforce_citation_rules`, reused here, not reimplemented).
- `not_found_correctness` — of the `answerable: false` questions, how often did the provider
  correctly return the exact `not_found` status instead of hallucinating an answer? This is the
  metric that would catch a model confidently answering about acupuncture from a cardiology
  corpus.
- `evidence_quote_match_rate` (Anthropic only) — of the structured citations returned, how many
  quotes verified verbatim against the source chunk (module 5 again).

**Retrieval happens once per question, shared across all requested providers.** `run_eval`
retrieves chunks once, then loops `providers.items()` to generate an answer from each with the
*same* retrieved chunks. This isolates comparisons to "same evidence, different model/prompt
handling" — if you're comparing Ollama vs. Anthropic, you're comparing their answering, not
accidentally also comparing what each one happened to retrieve.

**`--save` produces a human-readable side-by-side, not just metrics.** `_write_compare_md`
writes one markdown section per question, with each provider's full answer text and warnings
underneath. This is the artifact you'd actually read to judge "is this answer good," since the
aggregate metrics tell you *that* something changed but not *what* changed about the answer's
substance.

**Cloud eval runs cost real money and are rate-gated the same way as `ask`.** `eval_cmd`
(`cli.py:215-224`) runs the same `cloud_gate_error()` check as `ask`, and warns upfront with the
exact number of API calls about to be made (`n_questions`) before anything runs. `run_eval` also
feeds every Anthropic call through the same `SessionCostTracker`, so a long question set with a
tight `MEDRAG_MAX_SESSION_USD` can abort partway through — raising a `RuntimeError`, caught by
`eval_cmd` and turned into a clean CLI error rather than a traceback.

## Exercise

```bash
.venv/bin/medrag eval eval/questions.json --providers ollama --save eval/compare.md
```
Read the generated `eval/compare.md`. Find at least one `answerable: false` question
(acupuncture/ibuprofen-dosing/intermittent-fasting) and confirm whether Ollama correctly said
"Not found" or hallucinated something plausible-sounding. If it hallucinated, that's not a bug in
*your* setup — it's exactly the kind of result this harness exists to surface, and worth
mentioning to whoever owns prompt/retrieval tuning.

## Checkpoint

- Why is `hit_at_5` computed from retrieval alone, with no provider involved?
- If you run `eval` with `--providers ollama,anthropic`, how many times does retrieval happen per
  question — once total, or once per provider?
- Which field in each answerable question of `eval/questions.json` is flagged via `note` as
  "auto-suggested, needs my review," and what would you actually check it against to confirm or
  correct it?
