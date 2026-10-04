# 5. Answering — providers, prompts, and citation enforcement

## Why it matters

This is where "research support only" stops being a disclaimer in the README and becomes code
that actively constrains what the model is allowed to get away with. It's also the module with
the most moving parts: two providers, a shared contract, and a verification layer that doesn't
trust either of them.

See [04a-ask-workflow.md](04a-ask-workflow.md) for where this fits in the literal step-by-step
trace of a full `medrag ask` call.

## Read this, in order

1. `medrag/providers/base.py` — the `Answer`/`Citation`/`Usage` dataclasses, the `Generator`
   protocol, and `SYSTEM_PROMPT`. Read the system prompt's 5 numbered rules carefully; everything
   else in this module exists to enforce or react to them.
2. `medrag/citations.py` — all 33 lines. This is the verification layer, independent of either
   provider.
3. `medrag/providers/ollama_provider.py` — the simpler of the two providers.
4. `medrag/providers/claude_provider.py` — the more involved one. Read it after Ollama's, since
   it does the same job with more error handling and native citation support.

## Mental model

**One system prompt, two enforcement mechanisms.** Both providers get the same `SYSTEM_PROMPT`,
but they can't both rely on the model to format citations the same way:
- **Ollama** has no structured citation feature, so `CITATION_FORMAT_INSTRUCTION`
  (`ollama_provider.py:14-18`) asks the model to emit literal `[PMID:12345678]` markers in its
  text, which `citations.py: extract_pmid_markers` then regex-parses back out.
- **Claude** gets each retrieved chunk passed as a `document` content block with
  `citations: {"enabled": True}` (`claude_provider.py: _document_block`), so the API returns
  structured citation objects referencing a `document_index` — no regex needed. The provider
  re-injects `[PMID:...]` markers into the text itself (lines 224-225) so the two providers'
  *output text* looks the same to everything downstream, even though they got there differently.

**Nothing is trusted — `enforce_citation_rules` runs after both.** Both providers call
`enforce_citation_rules(answer, retrieved_pmids)` as their last step before returning. Open
`citations.py` again: it checks (1) every cited PMID was actually in the retrieved set — if not,
appends an `INVALID CITATION(S)` warning — and (2) a non-"not found" answer has *some* citation at
all. This runs regardless of what `status` the provider thinks the answer has. If you ever add a
third provider, this call is not optional boilerplate — skipping it reopens exactly the failure
mode (unverifiable medical claims) the rest of the system exists to prevent.

**Claude's citation quotes are verified character-for-character.** `claude_provider.py:213-219`
re-normalizes whitespace and checks the model's `cited_text` literally appears in the source
chunk's text. A mismatch doesn't block the answer — it adds a warning. This is what the eval
harness's `evidence_quote_match_rate` metric (module 9) is measuring.

**Status is richer than "did it work."** `Answer.status` is one of `answered`, `not_found`,
`refused`, `truncated`, `error` (`providers/base.py:8`). `claude_provider.py` derives `refused`
from the API's own `stop_reason == "refusal"` and `truncated` from `max_tokens` — these are
real, distinct outcomes a caller (CLI, eval harness) needs to branch on differently, not just
"error, go away."

**`is_not_found_reply` is an exact string match, not a fuzzy "sounds like a non-answer" check**
(`citations.py:13-14`). The model must reply with *exactly*
`"Not found in the retrieved literature."` for `status` to become `not_found`. This is strict by
design — see module 9's `not_found_correctness` eval metric, which depends on this exactness.

## Exercise

Open `tests/test_claude_provider.py` and find the test that asserts an `INVALID CITATION` warning
fires. Trace backward: what does the fake Claude response contain that triggers it, and which line
in `claude_provider.py` or `citations.py` is responsible? Then do the same for one Ollama test in
`test_ollama_provider.py`.

## Checkpoint

- Why does `claude_provider.py` re-inject `[PMID:...]` text markers even though it already has
  structured citation objects from the API?
- If a provider returns `status="answered"` with zero citations and non-"not found" text, what
  happens, and where?
- What's the difference between a `refused` and an `error` status, and which provider can produce
  `refused`?
