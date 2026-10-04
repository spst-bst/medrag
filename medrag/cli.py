from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from medrag import config
from medrag import doctor as doctor_checks
from medrag.audit import log_audit, log_usage
from medrag.citations import get_cited_pmids
from medrag.embeddings import OllamaEmbedder
from medrag.eval_runner import run_eval
from medrag.indexer import index_records
from medrag.ncbi import NCBIClient, load_records_jsonl, save_records_jsonl
from medrag.privacy import PHI_HEURISTIC_NOTICE, cloud_gate_error, scan_for_phi
from medrag.providers.claude_provider import ClaudeProvider
from medrag.providers.ollama_provider import OllamaProvider
from medrag.retrieval import retrieve, topic_spread_warning
from medrag.store import Store

app = typer.Typer(
    add_completion=False,
    help="medrag: local-first, evidence-citing PubMed literature Q&A. Research support only, not medical advice.",
)
console = Console()


@app.command()
def fetch(
    query: str = typer.Argument(..., help="PubMed search query"),
    max: int = typer.Option(200, "--max", help="Maximum number of records to fetch"),
):
    """Download PubMed records via NCBI E-utilities into data/records.jsonl."""
    email = config.medrag_email()
    client = NCBIClient(email=email, cache_dir=config.CACHE_DIR)
    console.print(f"Fetching up to {max} PubMed records for query: {query!r} (email={email})")
    records = client.fetch_all(query, max)
    save_records_jsonl(records, config.RECORDS_PATH)
    console.print(f"Saved {len(records)} records with abstracts to {config.RECORDS_PATH}")


@app.command(name="index")
def index_cmd():
    """Chunk, embed (always locally) and store records into data/index.db."""
    records = load_records_jsonl(config.RECORDS_PATH)
    if not records:
        console.print(f"[red]No records found at {config.RECORDS_PATH}. Run 'medrag fetch' first.[/red]")
        raise typer.Exit(1)

    store = Store(config.INDEX_DB_PATH)
    embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)

    import time

    start = time.monotonic()
    stats = index_records(store, embedder, records, config.embed_model())
    elapsed = time.monotonic() - start
    store.close()

    console.print(
        f"Indexed {stats['records_indexed']} new records, {stats['chunks_created']} new chunks, "
        f"{stats['embeddings_created']} new embeddings in {elapsed:.1f}s"
    )


def _print_sources(chunks, cited: set[str]) -> None:
    table = Table(title="Sources")
    table.add_column("PMID")
    table.add_column("Title")
    table.add_column("Year")
    table.add_column("Type")
    table.add_column("Cited")
    table.add_column("URL")
    seen = set()
    for c in chunks:
        if c.pmid in seen:
            continue
        seen.add(c.pmid)
        table.add_row(
            c.pmid,
            c.title,
            str(c.year or "n.d."),
            ", ".join(c.pub_types) or "Unknown",
            "yes" if c.pmid in cited else "",
            f"https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/",
        )
    console.print(table)


def _print_retrieval_debug(chunks) -> None:
    table = Table(title="Retrieval debug")
    table.add_column("chunk_id")
    table.add_column("PMID")
    table.add_column("score")
    table.add_column("ranking(s)")
    table.add_column("section")
    for c in chunks:
        table.add_row(
            str(c.chunk_id),
            c.pmid,
            f"{c.score:.4f}",
            "+".join(c.rank_sources) or "none",
            c.section or "-",
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(6, "--top-k"),
    provider: Optional[str] = typer.Option(None, "--provider", help="ollama (default) or anthropic"),
    show_evidence: bool = typer.Option(False, "--show-evidence"),
    i_confirm_no_phi: bool = typer.Option(
        False,
        "--i-confirm-no-phi",
        help="Override the PHI heuristic (which cannot detect names) after confirming this question has no patient data.",
    ),
    no_evidence_boost: bool = typer.Option(False, "--no-evidence-boost"),
    debug_retrieval: bool = typer.Option(
        False,
        "--debug-retrieval",
        help="Print each retrieved chunk's id, score, and which ranking(s) (bm25/cosine) it came from.",
    ),
):
    """Retrieve relevant excerpts and answer the question, citing PMIDs."""
    provider_name = provider or os.environ.get("MEDRAG_PROVIDER", "ollama")

    if provider_name == "anthropic":
        gate_err = cloud_gate_error()
        if gate_err:
            console.print(f"[red]{gate_err}[/red]")
            raise typer.Exit(1)
        phi_hits = scan_for_phi(question)
        if phi_hits and not i_confirm_no_phi:
            console.print(f"[red]Question may contain identifiers: {phi_hits}.[/red]")
            console.print(f"[yellow]{PHI_HEURISTIC_NOTICE}[/yellow]")
            console.print("Pass --i-confirm-no-phi if this is a false positive.")
            raise typer.Exit(1)

    store = Store(config.INDEX_DB_PATH)
    embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)
    chunks = retrieve(
        store,
        embedder,
        question,
        top_k=top_k,
        embed_model=config.embed_model(),
        use_evidence_boost=not no_evidence_boost,
    )
    spread_warning = (
        topic_spread_warning(store, config.embed_model(), chunks) if len(chunks) >= 2 else None
    )
    store.close()

    if not chunks:
        console.print("[yellow]No chunks retrieved from the index. Did you run 'medrag index'?[/yellow]")
        raise typer.Exit(1)

    if provider_name == "anthropic":
        model_name = config.anthropic_model()
        console.print(
            f"Sending your question and {len(chunks)} public PubMed abstracts to Anthropic's API "
            f"(model {model_name}). Nothing else leaves this machine."
        )
        if config.anthropic_fallbacks_enabled():
            console.print("[dim]Server-side fallbacks enabled (MEDRAG_ANTHROPIC_FALLBACKS=1).[/dim]")
        gen = ClaudeProvider()
    elif provider_name == "ollama":
        gen = OllamaProvider()
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        raise typer.Exit(1)

    if provider_name == "ollama":
        # OllamaProvider streams tokens via on_progress; print them as they arrive instead of
        # sitting silently behind a spinner for the whole call.
        console.print(f"\n[bold]Answer[/bold] (provider=ollama model={gen.model})\n")

        def _on_progress(piece: str) -> None:
            console.print(piece, end="", markup=False, highlight=False)

        answer = gen.generate(question, chunks, on_progress=_on_progress)
        console.print(f"\n\n[dim](status={answer.status} seconds={answer.seconds:.1f})[/dim]")
    else:
        with console.status(f"Asking {provider_name}..."):
            answer = gen.generate(question, chunks)
        console.print(
            f"\n[bold]Answer[/bold] (provider={answer.provider} model={answer.model} "
            f"status={answer.status} seconds={answer.seconds:.1f})\n"
        )
        console.print(answer.text)

    if answer.warnings:
        console.print("\n[bold red]Warnings:[/bold red]")
        for w in answer.warnings:
            console.print(f"  - {w}")

    if spread_warning:
        console.print(f"\n[yellow]Note:[/yellow] {spread_warning}")

    cited = get_cited_pmids(answer)
    _print_sources(chunks, cited)

    if show_evidence:
        quoted = [c for c in answer.citations if c.quote]
        if quoted:
            console.print("\n[bold]Evidence quotes[/bold]")
            for c in quoted:
                console.print(f'  PMID:{c.pmid} — "{c.quote}"')
        elif answer.provider != "anthropic":
            console.print(
                "\n[dim]--show-evidence has no effect with provider="
                f"{answer.provider}: it never attaches verbatim quotes to citations. "
                "Use --provider anthropic for quoted evidence.[/dim]"
            )

    if debug_retrieval:
        _print_retrieval_debug(chunks)

    if answer.provider == "anthropic":
        usage = answer.usage
        cost = (
            config.estimate_cost_usd(answer.model, usage.input_tokens, usage.output_tokens)
            if usage
            else None
        )
        if usage:
            console.print(
                f"\nTokens: input={usage.input_tokens} output={usage.output_tokens} "
                f"cache_creation={usage.cache_creation_input_tokens} cache_read={usage.cache_read_input_tokens}"
            )
        if cost is not None:
            console.print(f"Estimated cost: ${cost:.4f} (as of 2026-09; verify current pricing)")

        pmids_sent = sorted({c.pmid for c in chunks})
        log_audit(config.AUDIT_LOG_PATH, answer.provider, answer.model, question, pmids_sent, usage)
        log_usage(config.USAGE_LOG_PATH, answer.provider, answer.model, usage, cost)

        max_usd = config.max_session_usd()
        if max_usd is not None and cost is not None and cost > max_usd:
            console.print(
                f"[red]Estimated cost ${cost:.4f} exceeds MEDRAG_MAX_SESSION_USD=${max_usd:.4f}[/red]"
            )
            raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    questions_file: Path = typer.Argument(...),
    provider: Optional[str] = typer.Option(None, "--provider"),
    providers: Optional[str] = typer.Option(None, "--providers"),
    save: Optional[Path] = typer.Option(None, "--save"),
):
    """Run a batch of questions against one or more providers, retrieving once per question."""
    if providers:
        provider_names = [p.strip() for p in providers.split(",")]
    elif provider:
        provider_names = [provider]
    else:
        provider_names = ["ollama"]

    if "anthropic" in provider_names:
        gate_err = cloud_gate_error()
        if gate_err:
            console.print(f"[red]{gate_err}[/red]")
            raise typer.Exit(1)
        n_questions = len(json.loads(questions_file.read_text()))
        console.print(
            f"[yellow]This will make up to {n_questions} Anthropic API calls (one per question). "
            "Each call costs money. Make sure you have approved this step.[/yellow]"
        )

    try:
        metrics = run_eval(questions_file, provider_names, console, save_path=save)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    hits, total = metrics["hit_at_5"]
    if total:
        console.print(f"\nRetrieval hit@5: {hits}/{total}")

    for pname, m in metrics["providers"].items():
        vc_count, vc_total = m["valid_citation_rate"]
        nf_count, nf_total = m["not_found_correctness"]
        console.print(f"\n[bold]{pname}[/bold]")
        console.print(f"  valid-citation rate: {vc_count}/{vc_total}")
        console.print(f"  'Not found' correctness: {nf_count}/{nf_total if nf_total else 'n/a'}")
        console.print(f"  avg seconds/answer: {m['avg_seconds']:.1f}")
        if m["evidence_quote_match_rate"] is not None:
            qc, qt = m["evidence_quote_match_rate"]
            console.print(f"  evidence-quote match rate: {qc}/{qt if qt else 'n/a'}")
        if m["estimated_cost_usd"] is not None:
            console.print(
                f"  tokens: input={m['total_input_tokens']} output={m['total_output_tokens']} "
                f"estimated cost: ${m['estimated_cost_usd']:.4f}"
            )

    if save:
        console.print(f"\nSide-by-side comparison written to {save}")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Launch a local web UI for asking questions (http://127.0.0.1:8000 by default)."""
    import uvicorn

    console.print(f"Starting medrag UI at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("medrag.webapp:app", host=host, port=port)


@app.command()
def doctor():
    """Check Ollama, models, network, email env var, and the Anthropic setup."""
    checks = [
        ("Ollama reachable", doctor_checks.check_ollama()),
        ("Ollama required models", doctor_checks.check_ollama_models([config.chat_model(), config.embed_model()])),
        ("NCBI network", doctor_checks.check_network()),
        ("MEDRAG_EMAIL", doctor_checks.check_email_env()),
        ("anthropic package", doctor_checks.check_anthropic_package()),
        ("ANTHROPIC_API_KEY", doctor_checks.check_anthropic_key()),
        ("Cloud consent (MEDRAG_ALLOW_CLOUD)", doctor_checks.check_cloud_consent()),
    ]
    table = Table(title="medrag doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, (ok, detail) in checks:
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    console.print(
        "\n[dim]Live Anthropic model lookup (client.models.retrieve) is a real API call and is "
        "skipped unless you explicitly re-run with approval for that specific step.[/dim]"
    )


if __name__ == "__main__":
    app()
