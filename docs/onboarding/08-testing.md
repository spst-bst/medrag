# 8. Testing — how this repo tests without a model or a network

## Why it matters

You will write tests here. The repo has a strong, consistent convention for doing it without
flaky network calls or a running Ollama instance — learn the pattern before writing a test that
breaks it.

## Read this, in order

1. `tests/conftest.py` — three fixtures: `sample_xml_bytes`, `sample_records`, `tmp_store`.
2. `tests/fakes.py` — `FakeEmbedder` (deterministic hash-based, no network) and the fake Anthropic
   SDK surface (`FakeAnthropicClient`, `FakeMessagesNamespace`, `FakeTextBlock`, `FakeCitation`,
   ...). This file is the single biggest lever for testing provider code without an API key.
3. Pick two test files and read them fully: `tests/test_retrieval.py` (small, pure-logic) and
   `tests/test_claude_provider.py` (large, shows the fakes in real use).
4. `tests/test_cli_guards.py` — tests that the safety gates from module 6 actually block what
   they claim to block.

## Mental model

**Offline by default; real services are opt-in via env vars.** The README states it plainly:
"Real-service tests are skipped unless `MEDRAG_LIVE=1` (Ollama/NCBI) or
`MEDRAG_LIVE_ANTHROPIC=1` (cloud, costs money)." `.venv/bin/python -m pytest -q` with neither set
must never touch a network or spend money. If you add a test that calls a real service, gate it
behind one of these, matching existing skip-markers in the test files — don't be the PR that makes
CI flaky or costs money on every run.

**Fakes replace the network boundary, not the logic under test.** `FakeEmbedder` doesn't mock
`OllamaEmbedder` — it's a real, different `Embedder` implementation (same protocol from
`embeddings.py`) that hashes words instead of calling Ollama. `FakeAnthropicClient` is similarly a
stand-in object matching the subset of the real `anthropic` SDK's shape that `claude_provider.py`
actually calls (`.messages.stream`, `.beta.messages.stream`), injected via
`ClaudeProvider(client_factory=...)`. This is why `claude_provider.py`'s constructor accepts a
`client_factory` parameter at all — it exists for this seam, not for production flexibility.

**Dependency injection is deliberate and minimal, not a general pattern.** Only the classes that
need a test seam have one (`ClaudeProvider.client_factory`, `httpx.Client` params on
`OllamaEmbedder`/`OllamaProvider`/`NCBIClient`, `RateLimiter`'s injectable clock/sleep in
`ratelimit.py`). Don't add a constructor parameter "for testability" unless you actually have a
test that needs it — follow the existing restraint.

**Tests assert on warnings and status strings, not exceptions, for expected failure modes.** Look
at how `test_claude_provider.py` checks for an `INVALID CITATION` warning versus how
`test_cli_guards.py` checks exit codes and printed error text. Providers are designed to *return*
an `Answer` describing a failure (`status="error"`/`"refused"`, populated `warnings`) rather than
raise — mirror that in new tests instead of wrapping calls in `pytest.raises` for cases the code
already handles as data.

## Exercise

Write a new test in `tests/test_chunking.py` for a record whose `abstract_sections` is a single
`(None, text)` tuple with exactly `WORDS_PER_CHUNK` words (currently 250) — an edge case not
obviously covered by the existing tests. Confirm your prediction (one chunk? two?) by reading
`chunk_record`'s range logic before running the test, then run it and see if you were right.

## Checkpoint

- What's the difference between `MEDRAG_LIVE` and `MEDRAG_LIVE_ANTHROPIC`, and why are they
  separate flags instead of one?
- Why does `ClaudeProvider` accept a `client_factory` argument — what test need does it serve
  that a simple mock/monkeypatch wouldn't?
- When a provider hits an error it can anticipate (bad auth, rate limit, refusal), does it raise
  or return? How do the tests reflect that choice?
