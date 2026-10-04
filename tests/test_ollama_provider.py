from __future__ import annotations

import json

from medrag.models import RetrievedChunk
from medrag.providers.ollama_provider import OllamaProvider
from tests.fakes import fake_ollama_chat_client, fake_ollama_error_client

CHUNK = RetrievedChunk(
    chunk_id=1,
    pmid="10000001",
    header="PMID:10000001 | Title | 2019 | Randomized Controlled Trial",
    text="PMID:10000001 | Title | 2019 | Randomized Controlled Trial\nDapagliflozin reduced hospitalization.",
    section="RESULTS",
    title="Title",
    year=2019,
    journal="NEJM",
    pub_types=["Randomized Controlled Trial"],
    score=1.0,
)


def _ndjson(*contents_and_done):
    lines = []
    for content, done in contents_and_done:
        lines.append(json.dumps({"message": {"role": "assistant", "content": content}, "done": done}))
    return lines


def test_generate_extracts_valid_citation():
    client = fake_ollama_chat_client(_ndjson(("Dapagliflozin helped. [PMID:10000001]", False), ("", True)))
    provider = OllamaProvider(model="qwen2.5:7b", client=client)
    answer = provider.generate("Does dapagliflozin help?", [CHUNK])

    assert answer.status == "answered"
    assert [c.pmid for c in answer.citations] == ["10000001"]
    assert answer.warnings == []


def test_generate_flags_invalid_citation():
    client = fake_ollama_chat_client(_ndjson(("Benefit shown. [PMID:99999999]", False), ("", True)))
    provider = OllamaProvider(model="qwen2.5:7b", client=client)
    answer = provider.generate("question", [CHUNK])

    assert any("INVALID CITATION" in w for w in answer.warnings)


def test_generate_not_found_path():
    client = fake_ollama_chat_client(
        _ndjson(("Not found in the retrieved literature.", False), ("", True))
    )
    provider = OllamaProvider(model="qwen2.5:7b", client=client)
    answer = provider.generate("unrelated question", [CHUNK])

    assert answer.status == "not_found"
    assert answer.warnings == []


def test_generate_http_error_returns_error_status():
    client = fake_ollama_error_client(500)
    provider = OllamaProvider(model="qwen2.5:7b", client=client)
    answer = provider.generate("question", [CHUNK])

    assert answer.status == "error"
    assert answer.warnings
