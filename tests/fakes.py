from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


class FakeEmbedder:
    """Deterministic hash-based (feature hashing) embedder. No network, no Ollama."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_documents_batched(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


def fake_ollama_chat_client(ndjson_lines: list[str]) -> httpx.Client:
    """An httpx.Client whose POST /api/chat returns a canned NDJSON streaming body."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n".join(ndjson_lines) + "\n"
        return httpx.Response(200, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def fake_ollama_error_client(status_code: int = 500) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="internal error")

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- Fake Anthropic SDK surface ------------------------------------------------


class FakeCitation:
    def __init__(self, document_index, cited_text, document_title=None):
        self.type = "char_location"
        self.document_index = document_index
        self.cited_text = cited_text
        self.document_title = document_title
        self.start_char_index = None
        self.end_char_index = None


class FakeTextBlock:
    def __init__(self, text, citations=None):
        self.type = "text"
        self.text = text
        self.citations = citations or []


class FakeThinkingBlock:
    def __init__(self, thinking="internal reasoning"):
        self.type = "thinking"
        self.thinking = thinking


class FakeStopDetails:
    def __init__(self, category=None, explanation=None):
        self.category = category
        self.explanation = explanation


class FakeUsage:
    def __init__(
        self,
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class FakeMessage:
    def __init__(self, content, stop_reason="end_turn", stop_details=None, usage=None, model="claude-opus-5-5"):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = usage if usage is not None else FakeUsage()
        self.model = model


class FakeStreamContextManager:
    def __init__(self, final_message=None, raise_exc=None):
        self._final_message = final_message
        self._raise_exc = raise_exc

    def __enter__(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_final_message(self):
        return self._final_message


class FakeMessagesNamespace:
    def __init__(self, final_message=None, raise_exc=None, reject_kwargs=None):
        self._final_message = final_message
        self._raise_exc = raise_exc
        self._reject_kwargs = reject_kwargs or []
        self.call_count = 0
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        for bad_kwarg in self._reject_kwargs:
            if bad_kwarg in kwargs:
                raise TypeError(f"stream() got an unexpected keyword argument '{bad_kwarg}'")
        self.call_count += 1
        return FakeStreamContextManager(self._final_message, self._raise_exc)


class FakeBetaNamespace:
    def __init__(self, messages_namespace: FakeMessagesNamespace):
        self.messages = messages_namespace


class FakeAnthropicClient:
    def __init__(self, final_message=None, raise_exc=None, beta_rejects_fallback_kwargs=False):
        self.messages = FakeMessagesNamespace(final_message=final_message, raise_exc=raise_exc)
        if beta_rejects_fallback_kwargs:
            beta_messages = FakeMessagesNamespace(
                final_message=final_message, reject_kwargs=["betas", "fallbacks"]
            )
        else:
            beta_messages = FakeMessagesNamespace(final_message=final_message, raise_exc=raise_exc)
        self.beta = FakeBetaNamespace(beta_messages)
