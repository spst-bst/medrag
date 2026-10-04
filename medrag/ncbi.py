from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx

from medrag.models import Record
from medrag.pubmed_xml import parse_pubmed_xml
from medrag.ratelimit import RateLimiter

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_BATCH_SIZE = 100


def _cache_key(url: str, params: dict) -> str:
    raw = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NCBIClient:
    def __init__(
        self,
        email: str,
        tool: str = "medrag",
        client: Optional[httpx.Client] = None,
        cache_dir: Path = Path("data/cache"),
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.email = email
        self.tool = tool
        self.client = client or httpx.Client(timeout=60)
        self.cache_dir = cache_dir
        self.rate_limiter = rate_limiter or RateLimiter(max_per_second=3.0)

    def _get_cached_or_fetch(self, path: str, params: dict) -> bytes:
        url = f"{BASE_URL}/{path}"
        full_params = {**params, "tool": self.tool, "email": self.email}
        key = _cache_key(url, full_params)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{key}.bin"
        if cache_file.exists():
            return cache_file.read_bytes()

        self.rate_limiter.wait()
        resp = self.client.get(url, params=full_params)
        resp.raise_for_status()
        cache_file.write_bytes(resp.content)
        return resp.content

    def esearch(self, query: str, max_results: int) -> list[str]:
        params = {"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "json"}
        raw = self._get_cached_or_fetch("esearch.fcgi", params)
        data = json.loads(raw)
        return data.get("esearchresult", {}).get("idlist", [])

    def efetch_batch(self, pmids: list[str]) -> bytes:
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        return self._get_cached_or_fetch("efetch.fcgi", params)

    def fetch_all(self, query: str, max_results: int) -> list[Record]:
        pmids = self.esearch(query, max_results)
        records: list[Record] = []
        for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = pmids[i : i + EFETCH_BATCH_SIZE]
            xml_bytes = self.efetch_batch(batch)
            records.extend(parse_pubmed_xml(xml_bytes))
        return records


def save_records_jsonl(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "pmid": r.pmid,
                        "title": r.title,
                        "abstract_sections": r.abstract_sections,
                        "journal": r.journal,
                        "year": r.year,
                        "pub_types": r.pub_types,
                        "mesh_terms": r.mesh_terms,
                        "doi": r.doi,
                    }
                )
                + "\n"
            )


def load_records_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(
                Record(
                    pmid=d["pmid"],
                    title=d["title"],
                    abstract_sections=[tuple(s) for s in d["abstract_sections"]],
                    journal=d.get("journal"),
                    year=d.get("year"),
                    pub_types=d.get("pub_types", []),
                    mesh_terms=d.get("mesh_terms", []),
                    doi=d.get("doi"),
                )
            )
    return records
