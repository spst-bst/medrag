from __future__ import annotations

from medrag.citations import enforce_citation_rules, extract_pmid_markers, get_cited_pmids, is_not_found_reply
from medrag.providers.base import Answer, Citation


def _make_answer(text, citations, status="answered"):
    return Answer(text=text, status=status, citations=citations, model="m", provider="p", seconds=0.1, warnings=[])


def test_extract_pmid_markers():
    text = "Benefit was shown [PMID:123] and confirmed [PMID:456] and again [PMID:123]."
    assert extract_pmid_markers(text) == ["123", "456", "123"]


def test_is_not_found_reply():
    assert is_not_found_reply("Not found in the retrieved literature.")
    assert is_not_found_reply("  Not found in the retrieved literature.  \n")
    assert not is_not_found_reply("Not found in the literature.")


def test_valid_citation_no_warning():
    ans = _make_answer("Some finding [PMID:111].", [Citation(pmid="111")])
    enforce_citation_rules(ans, {"111", "222"})
    assert ans.warnings == []


def test_invalid_citation_flagged():
    ans = _make_answer("Some finding [PMID:999].", [Citation(pmid="999")])
    enforce_citation_rules(ans, {"111", "222"})
    assert len(ans.warnings) == 1
    assert "INVALID CITATION" in ans.warnings[0]
    assert "999" in ans.warnings[0]


def test_missing_citation_on_answered_flagged():
    ans = _make_answer("Some finding with no citation markers at all.", [])
    enforce_citation_rules(ans, {"111"})
    assert any("No citations found" in w for w in ans.warnings)


def test_not_found_with_no_citations_is_fine():
    ans = _make_answer("Not found in the retrieved literature.", [], status="not_found")
    enforce_citation_rules(ans, {"111"})
    assert ans.warnings == []


def test_citations_fallback_to_text_markers_when_list_empty():
    ans = _make_answer("Some finding [PMID:111].", [])
    enforce_citation_rules(ans, {"111"})
    assert ans.warnings == []


def test_get_cited_pmids_prefers_structured_citations_over_text():
    ans = _make_answer("text mentions [PMID:999] too", [Citation(pmid="111")])
    assert get_cited_pmids(ans) == {"111"}


def test_get_cited_pmids_falls_back_to_markers_when_no_structured_citations():
    ans = _make_answer("Some finding [PMID:111] and [PMID:222].", [])
    assert get_cited_pmids(ans) == {"111", "222"}
