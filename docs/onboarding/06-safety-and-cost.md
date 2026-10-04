# 6. Safety and cost — the cloud provider's guardrails

## Why it matters

This is the module most likely to come up in a code review comment or an incident. The project's
central promise — "nothing leaves this machine unless you explicitly say so, twice" — lives
entirely in a few small files. If you ever touch the Anthropic path, you need to know exactly
where these checks are and why removing or weakening one would be a real regression, not a
cleanup.

See [04a-ask-workflow.md](04a-ask-workflow.md) for exactly where these gates run relative to
everything else in a `medrag ask` call.

## Read this, in order

1. `medrag/privacy.py` — all 50 lines.
2. `medrag/config.py`'s `allow_cloud`, `max_session_usd`, `estimate_cost_usd`, and the
   `PRICES_PER_MILLION_TOKENS` table.
3. `medrag/audit.py` — `log_audit`, `log_usage`, `SessionCostTracker`.
4. How `cli.py: ask()` wires all three together (lines 108-118, 173-197) — this is the reference
   call order; `webapp.py: api_ask` and `eval_runner.py: run_eval` each reimplement the same
   sequence for their own entry point.

## Mental model

**Two separate opt-ins, checked in a specific order, before any client is constructed.**
`cloud_gate_error()` (`privacy.py:38-50`) is the *first* check in every cloud code path — note
the docstring: "Checked BEFORE constructing any API client." It requires
`MEDRAG_ALLOW_CLOUD=1` regardless of whether `--provider anthropic` was also passed; the CLI flag
alone is not consent. Only after that gate passes does the PHI scan run. If you're adding a new
caller of the Anthropic provider, copy this order — gate first, then PHI scan, then construct the
client — don't reinvent it.

**The PHI scan is a heuristic, and the code says so loudly, repeatedly.** `scan_for_phi`
(`privacy.py:23-35`) regex-matches SSN-like numbers, emails, phone numbers, MRN/DOB-style labels,
and "born ... date" patterns. `PHI_HEURISTIC_NOTICE` is printed every time it fires. It
**cannot** detect patient names — that's explicitly called out in the notice text, the README,
and this doc, three places on purpose. `--i-confirm-no-phi` doesn't disable the scan; it's a
human override for a *specific question* after a false positive, logged nowhere differently than
normal. Don't "fix" false positives by loosening the regexes without understanding you're also
narrowing detection — a looser SSN pattern that stops flagging phone numbers is a net loss.

**Cost is estimated client-side from a hardcoded price table, not billed usage.**
`PRICES_PER_MILLION_TOKENS` (`config.py:22-25`) is a static dict with a comment: "as of 2026-09;
verify current pricing before relying on these." If you're touching pricing, update this table
*and* update that comment's date — stale pricing silently under- or over-estimates cost. If a
model name isn't in the table, `estimate_cost_usd` returns `None` and every downstream cost
display just skips that field — it doesn't guess or error.

**`MEDRAG_MAX_SESSION_USD` is checked, but only after the call already happened.** Look at
`cli.py:192-197`: the cost is estimated and compared to the budget *after* the API call returns,
and the process exits non-zero if it's over. This stops you from running a *second* over-budget
call in the same session, but it cannot prevent the one call that pushed you over — there's no
pre-flight cost estimate. `SessionCostTracker` (`audit.py:58-73`) does the equivalent running-total
check across the multiple calls in `eval_runner.py`.

**Every cloud call is logged twice, to two different files, for two different purposes:**
- `data/audit.jsonl` (`log_audit`) — compliance/traceability: timestamp, provider, model, the
  *question text*, which PMIDs were sent, and usage. This is the one you'd pull up if someone
  asked "what did we send to Anthropic and when."
- `data/usage.jsonl` (`log_usage`) — cost tracking only: no question text, just usage and
  estimated cost.

Both are gitignored (`data/` as a whole is) — they're local operational logs, not something to
commit or ship.

## Exercise

Without an `ANTHROPIC_API_KEY` or `MEDRAG_ALLOW_CLOUD` set, run:
```bash
.venv/bin/medrag ask "test question" --provider anthropic
```
Read the exact error. Then set `MEDRAG_ALLOW_CLOUD=1` only (still no API key) and run it again —
notice the error changes, because you've now passed the *first* gate and hit a different failure
later in the chain. Finally, construct a question containing a fake SSN-shaped number
(`123-45-6789`) with cloud fully enabled, and confirm it's refused without `--i-confirm-no-phi`.

## Checkpoint

- Name the exact two environment/flag conditions required before any text reaches the Anthropic
  API, and the file + function that enforces each.
- Why is `MEDRAG_MAX_SESSION_USD` described as stopping *further* calls rather than preventing
  the triggering call itself?
- What's in `data/audit.jsonl` that is deliberately *not* in `data/usage.jsonl`, and why would
  that separation matter to someone doing a compliance review?
