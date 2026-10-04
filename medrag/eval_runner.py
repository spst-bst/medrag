from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from medrag import config
from medrag.audit import SessionCostTracker, log_audit, log_usage
from medrag.embeddings import OllamaEmbedder
from medrag.providers.base import Answer
from medrag.providers.claude_provider import ClaudeProvider
from medrag.providers.ollama_provider import OllamaProvider
from medrag.retrieval import retrieve
from medrag.store import Store


def _make_provider(name: str):
    if name == "ollama":
        return OllamaProvider()
    if name == "anthropic":
        return ClaudeProvider()
    raise ValueError(f"Unknown provider: {name}")


def run_eval(
    questions_path: Path,
    provider_names: list[str],
    console,
    save_path: Optional[Path] = None,
) -> dict:
    questions = json.loads(questions_path.read_text())
    store = Store(config.INDEX_DB_PATH)
    embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)

    providers = {name: _make_provider(name) for name in provider_names}
    cost_tracker = SessionCostTracker(max_usd=config.max_session_usd())

    results = []
    hit_at_5_hits = 0
    hit_at_5_total = 0

    for q in questions:
        chunks = retrieve(
            store, embedder, q["question"], top_k=6, embed_model=config.embed_model()
        )
        top5_pmids = {c.pmid for c in chunks[:5]}
        expected = set(q.get("expected_pmids", []))
        if q.get("answerable", True) and expected:
            hit_at_5_total += 1
            if top5_pmids & expected:
                hit_at_5_hits += 1

        answers: dict[str, Answer] = {}
        for pname, gen in providers.items():
            if pname == "anthropic":
                console.print(
                    f"Sending question {q.get('id', '?')} and {len(chunks)} public PubMed abstracts "
                    f"to Anthropic's API (model {config.anthropic_model()}). Nothing else leaves this machine."
                )
            answer = gen.generate(q["question"], chunks)
            answers[pname] = answer
            if pname == "anthropic":
                usage = answer.usage
                cost = (
                    config.estimate_cost_usd(answer.model, usage.input_tokens, usage.output_tokens)
                    if usage
                    else None
                )
                cost_tracker.add(cost)
                log_audit(
                    config.AUDIT_LOG_PATH,
                    answer.provider,
                    answer.model,
                    q["question"],
                    sorted({c.pmid for c in chunks}),
                    usage,
                )
                log_usage(config.USAGE_LOG_PATH, answer.provider, answer.model, usage, cost)
                exceeded = cost_tracker.check_exceeded()
                if exceeded:
                    store.close()
                    raise RuntimeError(exceeded)

        results.append({"question": q, "chunks": chunks, "answers": answers})

    store.close()

    metrics: dict = {"hit_at_5": (hit_at_5_hits, hit_at_5_total), "providers": {}}

    for pname in provider_names:
        total = len(results)
        valid_citation_count = 0
        notfound_correct = 0
        notfound_total = 0
        quote_match_count = 0
        quote_total = 0
        seconds_sum = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for r in results:
            ans = r["answers"][pname]
            seconds_sum += ans.seconds
            has_invalid = any("INVALID CITATION" in w for w in ans.warnings)
            if not has_invalid:
                valid_citation_count += 1
            if not r["question"].get("answerable", True):
                notfound_total += 1
                if ans.status == "not_found":
                    notfound_correct += 1
            if pname == "anthropic":
                for c in ans.citations:
                    quote_total += 1
                    mismatch = any(
                        f"PMID:{c.pmid}" in w and "mismatch" in w for w in ans.warnings
                    )
                    if c.quote and not mismatch:
                        quote_match_count += 1
                if ans.usage:
                    total_input_tokens += ans.usage.input_tokens or 0
                    total_output_tokens += ans.usage.output_tokens or 0
                    cost = config.estimate_cost_usd(
                        ans.model, ans.usage.input_tokens, ans.usage.output_tokens
                    )
                    if cost:
                        total_cost += cost

        metrics["providers"][pname] = {
            "valid_citation_rate": (valid_citation_count, total),
            "not_found_correctness": (notfound_correct, notfound_total),
            "avg_seconds": seconds_sum / total if total else 0.0,
            "evidence_quote_match_rate": (quote_match_count, quote_total) if pname == "anthropic" else None,
            "total_input_tokens": total_input_tokens if pname == "anthropic" else None,
            "total_output_tokens": total_output_tokens if pname == "anthropic" else None,
            "estimated_cost_usd": total_cost if pname == "anthropic" else None,
        }

    if save_path:
        _write_compare_md(save_path, results, provider_names)

    return metrics


def _write_compare_md(path: Path, results: list[dict], provider_names: list[str]) -> None:
    lines = ["# medrag eval comparison\n"]
    for r in results:
        q = r["question"]
        lines.append(f"## {q.get('id', '')}: {q['question']}\n")
        if q.get("expected_pmids"):
            lines.append(f"*Expert-expected PMIDs (auto-suggested, needs review): {', '.join(q['expected_pmids'])}*\n")
        for pname in provider_names:
            ans = r["answers"][pname]
            lines.append(f"### {pname} ({ans.model}, status={ans.status}, {ans.seconds:.1f}s)\n")
            lines.append(ans.text + "\n")
            if ans.warnings:
                lines.append("**Warnings:** " + "; ".join(ans.warnings) + "\n")
        lines.append("---\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
