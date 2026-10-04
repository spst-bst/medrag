from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.environ.get("MEDRAG_DATA_DIR", "data"))
RECORDS_PATH = DATA_DIR / "records.jsonl"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DB_PATH = DATA_DIR / "index.db"
USAGE_LOG_PATH = DATA_DIR / "usage.jsonl"
AUDIT_LOG_PATH = DATA_DIR / "audit.jsonl"

OLLAMA_BASE_URL = os.environ.get("MEDRAG_OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL_DEFAULT = "qwen2.5:7b"
OLLAMA_EMBED_MODEL_DEFAULT = "nomic-embed-text"

ANTHROPIC_MODEL_DEFAULT = "claude-opus-5-5"
ANTHROPIC_EFFORT_DEFAULT = "medium"

# USD per million tokens, as of 2026-09; verify current pricing before relying on these.
PRICES_PER_MILLION_TOKENS = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
}

EVIDENCE_PREFERRED_TYPES = {
    "Meta-Analysis",
    "Systematic Review",
    "Randomized Controlled Trial",
}
EVIDENCE_DEPRIORITIZED_TYPES = {
    "Case Reports",
}

RRF_K = 60


def chat_model() -> str:
    return os.environ.get("MEDRAG_MODEL", OLLAMA_CHAT_MODEL_DEFAULT)


def embed_model() -> str:
    return os.environ.get("MEDRAG_EMBED_MODEL", OLLAMA_EMBED_MODEL_DEFAULT)


def anthropic_model() -> str:
    return os.environ.get("MEDRAG_ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT)


def anthropic_effort() -> str:
    return os.environ.get("MEDRAG_ANTHROPIC_EFFORT", ANTHROPIC_EFFORT_DEFAULT)


def anthropic_fallbacks_enabled() -> bool:
    return os.environ.get("MEDRAG_ANTHROPIC_FALLBACKS", "1") != "0"


def medrag_email() -> str:
    email = os.environ.get("MEDRAG_EMAIL")
    if not email:
        raise RuntimeError(
            "MEDRAG_EMAIL is not set. NCBI E-utilities requires an identifying email "
            "for polite API usage. Set it with: export MEDRAG_EMAIL=you@example.com"
        )
    return email


def allow_cloud() -> bool:
    return os.environ.get("MEDRAG_ALLOW_CLOUD") == "1"


def max_session_usd() -> Optional[float]:
    val = os.environ.get("MEDRAG_MAX_SESSION_USD")
    if val is None:
        return None
    return float(val)


def estimate_cost_usd(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    prices = PRICES_PER_MILLION_TOKENS.get(model)
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
