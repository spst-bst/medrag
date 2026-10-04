from __future__ import annotations

import re

NOT_FOUND_TEXT = "Not found in the retrieved literature."
PMID_MARKER_RE = re.compile(r"\[PMID:(\d+)\]")


def extract_pmid_markers(text: str) -> list[str]:
    return PMID_MARKER_RE.findall(text)


def is_not_found_reply(text: str) -> bool:
    return text.strip() == NOT_FOUND_TEXT


def get_cited_pmids(answer) -> set[str]:
    """PMIDs the answer actually cites: structured citations if present, else [PMID:...] markers."""
    if answer.citations:
        return {c.pmid for c in answer.citations}
    return set(extract_pmid_markers(answer.text))


def enforce_citation_rules(answer, retrieved_pmids: set[str]) -> None:
    """Shared enforcement, applied after every provider's generate(): never trust the model.

    Mutates answer.warnings in place.
    """
    cited = get_cited_pmids(answer)

    invalid = sorted(p for p in cited if p not in retrieved_pmids)
    if invalid:
        answer.warnings.append(
            f"INVALID CITATION(S): PMID(s) {', '.join(invalid)} are not among the retrieved chunks."
        )

    if not cited and not is_not_found_reply(answer.text) and answer.status == "answered":
        answer.warnings.append(
            "No citations found in an answer that is not the exact 'Not found in the retrieved literature.' reply."
        )
