from __future__ import annotations

import re

from medrag.config import allow_cloud

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
MRN_DOB_LABEL_RE = re.compile(r"\b(MRN|DOB)\b\s*[:#]?\s*\S", re.IGNORECASE)
BORN_DATE_RE = re.compile(
    r"\b(born|date of birth|DOB)\b.{0,20}\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE
)

PHI_HEURISTIC_NOTICE = (
    "This PHI check is a heuristic: it looks for SSN-like numbers, emails, phone numbers, "
    "and MRN/DOB-style labels. It CANNOT detect patient names or other identifiers. "
    "Real patient data must never be sent to a cloud provider unless your organization's "
    "compliance team has approved it."
)


def scan_for_phi(text: str) -> list[str]:
    hits = []
    if SSN_RE.search(text):
        hits.append("SSN-like pattern")
    if EMAIL_RE.search(text):
        hits.append("email address")
    if PHONE_RE.search(text):
        hits.append("phone number")
    if MRN_DOB_LABEL_RE.search(text):
        hits.append("MRN/DOB-style label")
    if BORN_DATE_RE.search(text):
        hits.append("date near 'born'/'DOB'")
    return hits


def cloud_gate_error() -> str | None:
    """Returns an error message if cloud use is not permitted, else None.

    Checked BEFORE constructing any API client.
    """
    if not allow_cloud():
        return (
            "Cloud provider requires MEDRAG_ALLOW_CLOUD=1 in addition to --provider anthropic. "
            "This double opt-in exists because a cloud call sends your question and the retrieved "
            "public PubMed abstracts to Anthropic's API. Set MEDRAG_ALLOW_CLOUD=1 to proceed, or "
            "use --provider ollama to stay fully local."
        )
    return None
