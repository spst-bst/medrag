from __future__ import annotations

from typer.testing import CliRunner

from medrag import cli as cli_module
from medrag import config
from medrag.models import RetrievedChunk
from medrag.providers.base import Answer, Citation, Usage

runner = CliRunner()

FAKE_CHUNK = RetrievedChunk(
    chunk_id=1,
    pmid="10000001",
    header="PMID:10000001 | Title | 2019 | Randomized Controlled Trial",
    text="PMID:10000001 | Title | 2019 | Randomized Controlled Trial\nSome excerpt text.",
    section="RESULTS",
    title="Title",
    year=2019,
    journal="NEJM",
    pub_types=["Randomized Controlled Trial"],
    score=1.0,
)


class FakeStore:
    def __init__(self, *a, **kw):
        pass

    def close(self):
        pass


class FakeClaudeProvider:
    instantiations: list[bool] = []

    def __init__(self, *a, **kw):
        FakeClaudeProvider.instantiations.append(True)

    def generate(self, question, chunks, on_progress=None):
        return Answer(
            text="Dapagliflozin helps. [PMID:10000001] Research support only, not medical advice. "
            "Verify against the cited papers.",
            status="answered",
            citations=[Citation(pmid="10000001", quote="Some excerpt text.")],
            model="claude-opus-5-5",
            provider="anthropic",
            seconds=0.2,
            usage=Usage(input_tokens=500, output_tokens=100),
            warnings=[],
        )


def _patch_common(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "Store", FakeStore)
    monkeypatch.setattr(cli_module, "retrieve", lambda *a, **kw: [FAKE_CHUNK])
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(config, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    FakeClaudeProvider.instantiations = []
    monkeypatch.setattr(cli_module, "ClaudeProvider", FakeClaudeProvider)


def test_privacy_gate_blocks_without_allow_cloud(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?", "--provider", "anthropic"])

    assert result.exit_code == 1
    assert "MEDRAG_ALLOW_CLOUD" in result.output
    assert FakeClaudeProvider.instantiations == []


def test_phi_guard_blocks_question_with_email(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(
        cli_module.app,
        ["ask", "contact patient at john@example.com about trial results", "--provider", "anthropic"],
    )

    assert result.exit_code == 1
    assert "identifiers" in result.output.lower()
    assert FakeClaudeProvider.instantiations == []


def test_phi_guard_override_flag_proceeds(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(
        cli_module.app,
        [
            "ask",
            "contact patient at john@example.com about trial results",
            "--provider",
            "anthropic",
            "--i-confirm-no-phi",
        ],
    )

    assert result.exit_code == 0
    assert FakeClaudeProvider.instantiations == [True]


def test_audit_log_written_on_cloud_call(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?", "--provider", "anthropic"])

    assert result.exit_code == 0
    audit_path = tmp_path / "audit.jsonl"
    usage_path = tmp_path / "usage.jsonl"
    assert audit_path.exists()
    assert usage_path.exists()
    assert "10000001" in audit_path.read_text()


def test_cost_cap_stops_when_exceeded(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    monkeypatch.setenv("MEDRAG_MAX_SESSION_USD", "0.00001")
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?", "--provider", "anthropic"])

    assert result.exit_code == 1
    assert "exceeds MEDRAG_MAX_SESSION_USD" in result.output


FAKE_CHUNK_2 = RetrievedChunk(
    chunk_id=2,
    pmid="10000002",
    header="PMID:10000002 | Title 2 | 2020 | Case Reports",
    text="PMID:10000002 | Title 2 | 2020 | Case Reports\nUnrelated excerpt text.",
    section=None,
    title="Title 2",
    year=2020,
    journal="JAMA",
    pub_types=["Case Reports"],
    score=0.5,
)


class FakeOllamaProvider:
    def __init__(self, *a, **kw):
        self.model = "fake-ollama-model"

    def generate(self, question, chunks, on_progress=None):
        if on_progress:
            on_progress("Partial answer text ")
            on_progress("[PMID:10000001].")
        return Answer(
            text="Partial answer text [PMID:10000001].",
            status="answered",
            citations=[],
            model=self.model,
            provider="ollama",
            seconds=0.05,
            warnings=[],
        )


def test_ollama_streams_answer_and_marks_cited_source(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "OllamaProvider", FakeOllamaProvider)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?"])

    assert result.exit_code == 0
    assert "Partial answer text [PMID:10000001]." in result.output
    assert "fake-ollama-model" in result.output
    assert "10000001" in result.output


def test_show_evidence_notes_ollama_has_no_quotes(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "OllamaProvider", FakeOllamaProvider)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?", "--show-evidence"])

    assert result.exit_code == 0
    assert "--show-evidence has no effect" in result.output


def test_show_evidence_prints_quotes_for_anthropic(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    _patch_common(monkeypatch, tmp_path)

    result = runner.invoke(
        cli_module.app,
        ["ask", "What does the evidence say?", "--provider", "anthropic", "--show-evidence"],
    )

    assert result.exit_code == 0
    assert "Evidence quotes" in result.output
    assert "Some excerpt text." in result.output
    assert "--show-evidence has no effect" not in result.output


def test_debug_retrieval_prints_table(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "OllamaProvider", FakeOllamaProvider)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?", "--debug-retrieval"])

    assert result.exit_code == 0
    assert "Retrieval debug" in result.output


class FakeStoreWithScatteredEmbeddings(FakeStore):
    def get_embeddings_for_chunk_ids(self, chunk_ids, model):
        vectors = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]}
        return {cid: vectors[cid] for cid in chunk_ids if cid in vectors}


def test_topic_spread_note_shown_for_scattered_chunks(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "Store", FakeStoreWithScatteredEmbeddings)
    monkeypatch.setattr(cli_module, "retrieve", lambda *a, **kw: [FAKE_CHUNK, FAKE_CHUNK_2])
    monkeypatch.setattr(cli_module, "OllamaProvider", FakeOllamaProvider)

    result = runner.invoke(cli_module.app, ["ask", "What does the evidence say?"])

    assert result.exit_code == 0
    assert "topically scattered" in result.output
