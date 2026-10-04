from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from medrag.providers.base import Usage


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_audit(
    path: Path,
    provider: str,
    model: str,
    question: str,
    pmids: list[str],
    usage: Optional[Usage],
) -> None:
    _append_jsonl(
        path,
        {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "question": question,
            "pmids": pmids,
            "usage": asdict(usage) if usage else None,
        },
    )


def log_usage(
    path: Path,
    provider: str,
    model: str,
    usage: Optional[Usage],
    estimated_cost_usd: Optional[float],
) -> None:
    _append_jsonl(
        path,
        {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "usage": asdict(usage) if usage else None,
            "estimated_cost_usd": estimated_cost_usd,
        },
    )


class SessionCostTracker:
    def __init__(self, max_usd: Optional[float] = None):
        self.max_usd = max_usd
        self.total_usd = 0.0

    def add(self, cost_usd: Optional[float]) -> None:
        if cost_usd:
            self.total_usd += cost_usd

    def check_exceeded(self) -> Optional[str]:
        if self.max_usd is not None and self.total_usd > self.max_usd:
            return (
                f"Session cost estimate ${self.total_usd:.4f} exceeds MEDRAG_MAX_SESSION_USD="
                f"${self.max_usd:.4f}. Stopping further cloud calls."
            )
        return None
