from __future__ import annotations

import json
import sys

import httpx2
import pytest

from medrag.models import RetrievedChunk
from medrag.providers import claude_provider as cp
from tests.fakes import (
    FakeAnthropicClient,
    FakeCitation,
    FakeMessage,
    FakeStopDetails,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeUsage,
)

CHUNK_A = RetrievedChunk(
    chunk_id=1,
    pmid="10000001",
    header="PMID:10000001 | Dapagliflozin trial | 2019 | Randomized Controlled Trial",
    text=(
        "PMID:10000001 | Dapagliflozin trial | 2019 | Randomized Controlled Trial\n"
        "Hospitalization for heart failure or cardiovascular death occurred in 16.3% of the "
        "dapagliflozin group versus 21.2% of the placebo group."
    ),
    section="RESULTS",
    title="Dapagliflozin trial",
    year=2019,
    journal="NEJM",
    pub_types=["Randomized Controlled Trial"],
    score=1.0,
)

CHUNK_B = RetrievedChunk(
    chunk_id=2,
    pmid="10000003",
    header="PMID:10000003 | SGLT2 review | 2021 | Review",
    text=(
        "PMID:10000003 | SGLT2 review | 2021 | Review\n"
        "SGLT2 inhibitors have emerged as a cornerstone therapy across the spectrum of heart failure."
    ),
    section=None,
    title="SGLT2 review",
    year=2021,
    journal="Rev Cardiovasc Med",
    pub_types=["Review"],
    score=0.9,
)

CHUNKS = [CHUNK_A, CHUNK_B]


def make_provider(client, **kwargs):
    return cp.ClaudeProvider(model="claude-opus-5-5", client_factory=lambda: client, **kwargs)


# --- request shape --------------------------------------------------------


def test_request_shape_documents_then_question_citations_enabled():
    client = FakeAnthropicClient(final_message=FakeMessage(content=[FakeTextBlock("ok")]))
    provider = make_provider(client)
    provider.generate("What happened?", CHUNKS)

    kwargs = client.beta.messages.last_kwargs
    assert kwargs is not None
    content = kwargs["messages"][0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "document"
    assert content[1]["type"] == "document"
    assert content[2] == {"type": "text", "text": "What happened?"}
    for doc in content[:2]:
        assert doc["citations"] == {"enabled": True}
    assert content[0]["title"].startswith("PMID:10000001")
    assert content[1]["title"].startswith("PMID:10000003")


def test_request_shape_model_and_effort():
    client = FakeAnthropicClient(final_message=FakeMessage(content=[FakeTextBlock("ok")]))
    provider = make_provider(client, effort="high")
    provider.generate("q", CHUNKS)

    kwargs = client.beta.messages.last_kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"] == {"effort": "high"}


def test_request_shape_forbidden_params_absent():
    client = FakeAnthropicClient(final_message=FakeMessage(content=[FakeTextBlock("ok")]))
    provider = make_provider(client)
    provider.generate("q", CHUNKS)

    kwargs = client.beta.messages.last_kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "thinking" not in kwargs
    assert "tool_choice" not in kwargs
    assert "tools" not in kwargs
    assert "format" not in kwargs["output_config"]
    # no assistant prefill: only a single user message
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"


# --- citations rendering ---------------------------------------------------


def test_citations_rendered_as_pmid_markers():
    cited_text = "Hospitalization for heart failure or cardiovascular death occurred in 16.3%"
    msg = FakeMessage(
        content=[
            FakeTextBlock(
                "Dapagliflozin reduced hospitalization.",
                citations=[FakeCitation(document_index=0, cited_text=cited_text)],
            )
        ]
    )
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("Does dapagliflozin help?", CHUNKS)

    assert "[PMID:10000001]" in answer.text
    assert answer.status == "answered"
    assert answer.citations[0].pmid == "10000001"
    assert answer.citations[0].quote == cited_text
    assert not any("mismatch" in w for w in answer.warnings)
    assert not any("INVALID CITATION" in w for w in answer.warnings)


def test_quote_mismatch_flagged():
    msg = FakeMessage(
        content=[
            FakeTextBlock(
                "Dapagliflozin reduced hospitalization.",
                citations=[FakeCitation(document_index=0, cited_text="this text does not appear verbatim")],
            )
        ]
    )
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert any("mismatch" in w and "10000001" in w for w in answer.warnings)


def test_out_of_range_document_index_flagged():
    msg = FakeMessage(
        content=[FakeTextBlock("text", citations=[FakeCitation(document_index=99, cited_text="x")])]
    )
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert any("out-of-range" in w for w in answer.warnings)
    assert answer.citations == []


def test_thinking_blocks_skipped_and_never_in_text():
    msg = FakeMessage(
        content=[
            FakeThinkingBlock("secret reasoning the user should never see"),
            FakeTextBlock(
                "Final answer. Research support only, not medical advice. Verify against the cited papers.",
                citations=[FakeCitation(document_index=0, cited_text="16.3%")],
            ),
        ]
    )
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert "secret reasoning" not in answer.text


# --- no-citation and not-found paths ---------------------------------------


def test_no_citation_warning_on_answered_non_notfound():
    msg = FakeMessage(content=[FakeTextBlock("Some answer with no citations at all.", citations=[])])
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "answered"
    assert any("No citations found" in w for w in answer.warnings)


def test_not_found_path():
    msg = FakeMessage(content=[FakeTextBlock("Not found in the retrieved literature.", citations=[])])
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("unrelated question", CHUNKS)

    assert answer.status == "not_found"
    assert not any("No citations found" in w for w in answer.warnings)


# --- stop_reason handling ---------------------------------------------------


def test_refusal_with_stop_details():
    msg = FakeMessage(
        content=[FakeTextBlock("")],
        stop_reason="refusal",
        stop_details=FakeStopDetails(category="dual_use_biology", explanation="sensitive topic"),
    )
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "refused"
    assert any("dual_use_biology" in w and "sensitive topic" in w for w in answer.warnings)


def test_refusal_with_none_stop_details_does_not_crash():
    msg = FakeMessage(content=[FakeTextBlock("")], stop_reason="refusal", stop_details=None)
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "refused"
    assert any("category=unknown" in w for w in answer.warnings)


def test_max_tokens_truncated():
    msg = FakeMessage(content=[FakeTextBlock("partial answer")], stop_reason="max_tokens")
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "truncated"
    assert any("max_tokens" in w for w in answer.warnings)


def test_unexpected_stop_reason_warns():
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])], stop_reason="something_new")
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert any("Unexpected stop_reason" in w for w in answer.warnings)


# --- fallback behavior --------------------------------------------------------


def test_served_by_another_model_warns():
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])], model="claude-sonnet-5-5")
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.model == "claude-sonnet-5-5"
    assert any("served by claude-sonnet-5-5 after a fallback" in w for w in answer.warnings)


def test_fallbacks_enabled_uses_beta_namespace():
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])])
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client, fallbacks_enabled=True)
    provider.generate("q", CHUNKS)

    assert client.beta.messages.call_count == 1
    assert client.messages.call_count == 0


def test_fallbacks_disabled_uses_plain_namespace():
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])])
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client, fallbacks_enabled=False)
    provider.generate("q", CHUNKS)

    assert client.messages.call_count == 1
    assert client.beta.messages.call_count == 0


def test_sdk_rejects_fallback_params_falls_back_to_plain_call():
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])])
    client = FakeAnthropicClient(final_message=msg, beta_rejects_fallback_kwargs=True)
    provider = make_provider(client, fallbacks_enabled=True)
    answer = provider.generate("q", CHUNKS)

    assert client.messages.call_count == 1
    assert any("rejected server-side-fallback parameters" in w for w in answer.warnings)
    assert answer.status == "answered"


# --- typed error handling ---------------------------------------------------


def _httpx2_response(status_code, headers=None):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status_code, request=req, headers=headers or {}, json={"error": "x"})


def test_authentication_error_handled():
    import anthropic

    resp = _httpx2_response(401)
    err = anthropic.AuthenticationError("invalid x-api-key", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("Authentication failed" in w and "ANTHROPIC_API_KEY" in w for w in answer.warnings)


def test_not_found_error_handled():
    import anthropic

    resp = _httpx2_response(404)
    err = anthropic.NotFoundError("model not found", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("Model not found" in w and "claude-opus-5-5" in w for w in answer.warnings)


def test_rate_limit_error_handled_with_retry_after():
    import anthropic

    resp = _httpx2_response(429, headers={"retry-after": "30"})
    err = anthropic.RateLimitError("rate limited", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("retry_after=30" in w for w in answer.warnings)


def test_api_status_error_server_error_handled():
    import anthropic

    resp = _httpx2_response(500)
    err = anthropic.APIStatusError("server broke", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("server error 500" in w for w in answer.warnings)


def test_api_status_error_client_error_handled():
    import anthropic

    resp = _httpx2_response(400)
    err = anthropic.APIStatusError("bad request", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("API error 400" in w for w in answer.warnings)


def test_api_connection_error_handled():
    import anthropic

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(message="connection failed", request=req)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any("Connection error" in w for w in answer.warnings)


def test_missing_anthropic_package_gives_install_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = cp.ClaudeProvider(model="claude-opus-5-5", client_factory=lambda: FakeAnthropicClient())
    answer = provider.generate("q", CHUNKS)

    assert answer.status == "error"
    assert any(".venv/bin/python -m pip install anthropic" in w for w in answer.warnings)


# --- usage ---------------------------------------------------------------


def test_usage_fields_populated():
    usage = FakeUsage(input_tokens=1234, output_tokens=56, cache_creation_input_tokens=7, cache_read_input_tokens=8)
    msg = FakeMessage(content=[FakeTextBlock("answer", citations=[])], usage=usage)
    client = FakeAnthropicClient(final_message=msg)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    assert answer.usage.input_tokens == 1234
    assert answer.usage.output_tokens == 56
    assert answer.usage.cache_creation_input_tokens == 7
    assert answer.usage.cache_read_input_tokens == 8


def test_api_key_never_appears_in_output(monkeypatch, capsys):
    import anthropic

    fake_key_value = "sk-ant-TESTKEY-should-never-appear-0000000000"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key_value)

    resp = _httpx2_response(401)
    err = anthropic.AuthenticationError("invalid x-api-key", response=resp, body=None)
    client = FakeAnthropicClient(raise_exc=err)
    provider = make_provider(client)
    answer = provider.generate("q", CHUNKS)

    all_text = answer.text + " ".join(answer.warnings) + json.dumps(
        [c.__dict__ for c in answer.citations]
    )
    assert fake_key_value not in all_text
    captured = capsys.readouterr()
    assert fake_key_value not in captured.out
    assert fake_key_value not in captured.err
