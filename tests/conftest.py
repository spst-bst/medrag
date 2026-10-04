from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_xml_bytes() -> bytes:
    return (FIXTURES_DIR / "sample_pubmed.xml").read_bytes()


@pytest.fixture
def sample_records(sample_xml_bytes):
    from medrag.pubmed_xml import parse_pubmed_xml

    return parse_pubmed_xml(sample_xml_bytes)


@pytest.fixture
def tmp_store(tmp_path):
    from medrag.store import Store

    store = Store(tmp_path / "index.db")
    yield store
    store.close()
