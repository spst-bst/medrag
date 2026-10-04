from __future__ import annotations


def test_skips_records_without_abstract(sample_records):
    pmids = {r.pmid for r in sample_records}
    assert "10000002" not in pmids
    assert len(sample_records) == 4


def test_structured_abstract_sections(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000001")
    assert rec.title.startswith("Dapagliflozin and outcomes")
    assert rec.journal == "New England Journal of Medicine"
    assert rec.year == 2019
    assert "Randomized Controlled Trial" in rec.pub_types
    assert rec.doi == "10.1056/NEJMoa1911303"
    assert "Heart Failure" in rec.mesh_terms
    labels = [label for label, _ in rec.abstract_sections]
    assert labels == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]


def test_unstructured_abstract_has_no_label(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000003")
    assert len(rec.abstract_sections) == 1
    label, text = rec.abstract_sections[0]
    assert label is None
    assert "SGLT2 inhibitors" in text


def test_unicode_preserved(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000004")
    assert "β-Zell-Funktion" in rec.title
    assert "café-au-lait" in rec.abstract_sections[0][1]
    assert "Δ = 12" in rec.abstract_sections[1][1]


def test_medline_date_year_fallback(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000005")
    assert rec.year == 2019
    assert rec.pub_types == ["Case Reports"]
