from __future__ import annotations

import os

import httpx

from medrag.config import OLLAMA_BASE_URL, anthropic_model


def check_ollama() -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        resp.raise_for_status()
        return True, f"Ollama reachable: {resp.json()}"
    except httpx.HTTPError as e:
        return False, f"Ollama NOT reachable at {OLLAMA_BASE_URL}: {e}"


def check_ollama_models(required: list[str]) -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}
    except httpx.HTTPError as e:
        return False, f"Could not list Ollama models: {e}"
    missing = [m for m in required if not any(n == m or n.startswith(m + ":") for n in names)]
    if missing:
        return False, f"Missing Ollama models: {missing}. Found: {sorted(names)}"
    return True, f"Required Ollama models present: {required}"


def check_network() -> tuple[bool, str]:
    try:
        resp = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed", timeout=10
        )
        ok = resp.status_code == 200
        return ok, f"NCBI E-utilities HTTP {resp.status_code}"
    except httpx.HTTPError as e:
        return False, f"NCBI E-utilities unreachable: {e}"


def check_email_env() -> tuple[bool, str]:
    email = os.environ.get("MEDRAG_EMAIL")
    if not email:
        return False, "MEDRAG_EMAIL is not set (required for NCBI fetches)."
    return True, f"MEDRAG_EMAIL is set ({email})"


def check_anthropic_package() -> tuple[bool, str]:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "anthropic package NOT installed (pip install anthropic to enable --provider anthropic)"
    return True, f"anthropic package importable (version {getattr(anthropic, '__version__', 'unknown')})"


def check_anthropic_key() -> tuple[bool, str]:
    is_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return is_set, f"ANTHROPIC_API_KEY is {'set' if is_set else 'NOT set'} (value never shown)"


def check_cloud_consent() -> tuple[bool, str]:
    is_set = os.environ.get("MEDRAG_ALLOW_CLOUD") == "1"
    return is_set, f"MEDRAG_ALLOW_CLOUD is {'1 (cloud calls permitted)' if is_set else 'not set (cloud calls blocked)'}"


def check_anthropic_model_live(client) -> tuple[bool, str]:
    """Only call when the user has approved a live check. Costs no tokens but is a real API call."""
    model = anthropic_model()
    try:
        client.models.retrieve(model)
        return True, f"Model '{model}' confirmed to exist via live lookup."
    except Exception as e:
        return False, f"Live model lookup for '{model}' failed: {e}"
