from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Record:
    pmid: str
    title: str
    abstract_sections: list[tuple[Optional[str], str]]
    journal: Optional[str]
    year: Optional[int]
    pub_types: list[str]
    mesh_terms: list[str]
    doi: Optional[str]


@dataclass
class Chunk:
    pmid: str
    header: str
    text: str
    section: Optional[str]


@dataclass
class RetrievedChunk:
    chunk_id: int
    pmid: str
    header: str
    text: str
    section: Optional[str]
    title: str
    year: Optional[int]
    journal: Optional[str]
    pub_types: list[str]
    score: float = 0.0
    rank_sources: list[str] = field(default_factory=list)
