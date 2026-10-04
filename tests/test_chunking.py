from __future__ import annotations

from medrag.chunking import chunk_record
from medrag.models import Record


def test_structured_abstract_one_chunk_per_section(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000001")
    chunks = chunk_record(rec)
    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.pmid == "10000001"
        assert chunk.text.startswith(
            "PMID:10000001 | Dapagliflozin and outcomes in heart failure with reduced ejection fraction: "
            "a randomized trial | 2019 | Randomized Controlled Trial, Journal Article"
        )
    assert chunks[0].section == "BACKGROUND"
    assert "cardiovascular death" in chunks[0].text


def test_unstructured_abstract_single_chunk_under_limit(sample_records):
    rec = next(r for r in sample_records if r.pmid == "10000003")
    chunks = chunk_record(rec)
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert "PMID:10000003" in chunks[0].header


def test_long_unstructured_abstract_splits_by_words():
    long_text = " ".join(f"word{i}" for i in range(600))
    rec = Record(
        pmid="99999999",
        title="Synthetic long record",
        abstract_sections=[(None, long_text)],
        journal="Test Journal",
        year=2020,
        pub_types=["Review"],
        mesh_terms=[],
        doi=None,
    )
    chunks = chunk_record(rec)
    assert len(chunks) == 3  # 600 words / 250 per chunk -> 3 chunks
    assert chunks[0].text.count("word0 ") or "word0" in chunks[0].text
    assert "word599" in chunks[-1].text


def test_no_abstract_produces_no_chunks():
    rec = Record(
        pmid="88888888",
        title="No abstract",
        abstract_sections=[],
        journal=None,
        year=None,
        pub_types=[],
        mesh_terms=[],
        doi=None,
    )
    assert chunk_record(rec) == []
