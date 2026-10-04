from __future__ import annotations

from medrag.models import Chunk, Record

WORDS_PER_CHUNK = 250


def _header_for(record: Record, section: str | None = None) -> str:
    pub_type = ", ".join(record.pub_types) if record.pub_types else "Unknown"
    base = f"PMID:{record.pmid} | {record.title} | {record.year or 'n.d.'} | {pub_type}"
    if section:
        base += f" | {section}"
    return base


def chunk_record(record: Record) -> list[Chunk]:
    chunks: list[Chunk] = []
    has_labeled_sections = any(section for section, _ in record.abstract_sections)

    if has_labeled_sections:
        for section, text in record.abstract_sections:
            header = _header_for(record, section)
            chunks.append(Chunk(pmid=record.pmid, header=header, text=f"{header}\n{text}", section=section))
    else:
        full_text = " ".join(text for _, text in record.abstract_sections)
        words = full_text.split()
        header = _header_for(record)
        if not words:
            return chunks
        for i in range(0, len(words), WORDS_PER_CHUNK):
            part = " ".join(words[i : i + WORDS_PER_CHUNK])
            chunks.append(Chunk(pmid=record.pmid, header=header, text=f"{header}\n{part}", section=None))

    return chunks
