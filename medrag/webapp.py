from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from medrag import config
from medrag import doctor as doctor_checks
from medrag.audit import log_audit, log_usage
from medrag.citations import get_cited_pmids
from medrag.embeddings import OllamaEmbedder
from medrag.privacy import PHI_HEURISTIC_NOTICE, cloud_gate_error, scan_for_phi
from medrag.providers.claude_provider import ClaudeProvider
from medrag.providers.ollama_provider import OllamaProvider
from medrag.retrieval import retrieve, topic_spread_warning
from medrag.store import Store

app = FastAPI(title="medrag", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    top_k: int = 6
    provider: str = "ollama"
    show_evidence: bool = False
    i_confirm_no_phi: bool = False
    no_evidence_boost: bool = False


class SourceOut(BaseModel):
    pmid: str
    title: str
    year: Optional[int] = None
    pub_types: list[str] = []
    url: str
    cited: bool = False


class EvidenceOut(BaseModel):
    pmid: str
    quote: str


class AskResponse(BaseModel):
    text: str = ""
    status: str
    provider: str
    model: str = ""
    seconds: float = 0.0
    warnings: list[str] = []
    sources: list[SourceOut] = []
    evidence: list[EvidenceOut] = []
    usage: Optional[dict] = None
    estimated_cost_usd: Optional[float] = None
    topic_spread_warning: Optional[str] = None
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/doctor")
def api_doctor() -> dict:
    checks = [
        ("Ollama reachable", doctor_checks.check_ollama()),
        ("Ollama required models", doctor_checks.check_ollama_models([config.chat_model(), config.embed_model()])),
        ("NCBI network", doctor_checks.check_network()),
        ("MEDRAG_EMAIL", doctor_checks.check_email_env()),
        ("anthropic package", doctor_checks.check_anthropic_package()),
        ("ANTHROPIC_API_KEY", doctor_checks.check_anthropic_key()),
        ("Cloud consent (MEDRAG_ALLOW_CLOUD)", doctor_checks.check_cloud_consent()),
    ]
    return {"checks": [{"name": name, "ok": ok, "detail": detail} for name, (ok, detail) in checks]}


@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    provider_name = req.provider or "ollama"

    if provider_name == "anthropic":
        gate_err = cloud_gate_error()
        if gate_err:
            return AskResponse(status="refused", provider=provider_name, error=gate_err)
        phi_hits = scan_for_phi(req.question)
        if phi_hits and not req.i_confirm_no_phi:
            return AskResponse(
                status="refused",
                provider=provider_name,
                error=f"Question may contain identifiers: {phi_hits}. {PHI_HEURISTIC_NOTICE}",
            )
    elif provider_name != "ollama":
        return AskResponse(status="error", provider=provider_name, error=f"Unknown provider: {provider_name}")

    store = Store(config.INDEX_DB_PATH)
    embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)
    chunks = retrieve(
        store,
        embedder,
        req.question,
        top_k=req.top_k,
        embed_model=config.embed_model(),
        use_evidence_boost=not req.no_evidence_boost,
    )
    spread_warning = topic_spread_warning(store, config.embed_model(), chunks) if len(chunks) >= 2 else None
    store.close()

    if not chunks:
        return AskResponse(
            status="error",
            provider=provider_name,
            error="No chunks retrieved from the index. Run 'medrag index' first.",
        )

    gen = ClaudeProvider() if provider_name == "anthropic" else OllamaProvider()
    answer = gen.generate(req.question, chunks)
    cited = get_cited_pmids(answer)

    sources: list[SourceOut] = []
    seen: set[str] = set()
    for c in chunks:
        if c.pmid in seen:
            continue
        seen.add(c.pmid)
        sources.append(
            SourceOut(
                pmid=c.pmid,
                title=c.title,
                year=c.year,
                pub_types=c.pub_types,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/",
                cited=c.pmid in cited,
            )
        )

    evidence: list[EvidenceOut] = []
    if req.show_evidence:
        for c in answer.citations:
            if c.quote:
                evidence.append(EvidenceOut(pmid=c.pmid, quote=c.quote))

    warnings_out = list(answer.warnings)
    if req.show_evidence and not evidence and answer.provider != "anthropic":
        warnings_out.append(
            f"--show-evidence has no effect with provider={answer.provider}: it never attaches "
            "verbatim quotes to citations. Use provider=anthropic for quoted evidence."
        )

    usage_out = None
    cost = None
    if answer.provider == "anthropic":
        usage = answer.usage
        cost = config.estimate_cost_usd(answer.model, usage.input_tokens, usage.output_tokens) if usage else None
        if usage:
            usage_out = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
            }
        pmids_sent = sorted({c.pmid for c in chunks})
        log_audit(config.AUDIT_LOG_PATH, answer.provider, answer.model, req.question, pmids_sent, usage)
        log_usage(config.USAGE_LOG_PATH, answer.provider, answer.model, usage, cost)

    return AskResponse(
        text=answer.text,
        status=answer.status,
        provider=answer.provider,
        model=answer.model,
        seconds=answer.seconds,
        warnings=warnings_out,
        sources=sources,
        evidence=evidence,
        usage=usage_out,
        estimated_cost_usd=cost,
        topic_spread_warning=spread_warning,
    )
