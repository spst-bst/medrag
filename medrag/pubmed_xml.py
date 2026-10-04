from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from medrag.models import Record


def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None or elem.text is None:
        return None
    return elem.text.strip() or None


def _parse_year(article_elem: ET.Element) -> Optional[int]:
    pub_date = article_elem.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year_elem = pub_date.find("Year")
        if year_elem is not None and year_elem.text and year_elem.text.strip().isdigit():
            return int(year_elem.text.strip())
        medline_date = pub_date.find("MedlineDate")
        if medline_date is not None and medline_date.text:
            digits = "".join(ch for ch in medline_date.text[:4] if ch.isdigit())
            if len(digits) == 4:
                return int(digits)
    return None


def _parse_abstract_sections(article_elem: ET.Element) -> list[tuple[Optional[str], str]]:
    sections: list[tuple[Optional[str], str]] = []
    abstract = article_elem.find(".//Abstract")
    if abstract is None:
        return sections
    for abstract_text in abstract.findall("AbstractText"):
        label = abstract_text.get("Label")
        text = "".join(abstract_text.itertext()).strip()
        if not text:
            continue
        sections.append((label, text))
    return sections


def _parse_pub_types(article_elem: ET.Element) -> list[str]:
    return [
        t.text.strip()
        for t in article_elem.findall(".//PublicationTypeList/PublicationType")
        if t.text and t.text.strip()
    ]


def _parse_mesh_terms(medline_citation: ET.Element) -> list[str]:
    return [
        t.text.strip()
        for t in medline_citation.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if t.text and t.text.strip()
    ]


def _parse_doi(pubmed_data: Optional[ET.Element]) -> Optional[str]:
    if pubmed_data is None:
        return None
    for article_id in pubmed_data.findall(".//ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "doi" and article_id.text:
            return article_id.text.strip()
    return None


def parse_pubmed_xml(xml_bytes: bytes) -> list[Record]:
    """Parse a PubmedArticleSet XML document into Records. Skips records with no abstract."""
    root = ET.fromstring(xml_bytes)
    records: list[Record] = []

    for pubmed_article in root.findall(".//PubmedArticle"):
        medline_citation = pubmed_article.find("MedlineCitation")
        if medline_citation is None:
            continue
        pmid_elem = medline_citation.find("PMID")
        pmid = _text_or_none(pmid_elem)
        if not pmid:
            continue

        article_elem = medline_citation.find("Article")
        if article_elem is None:
            continue

        abstract_sections = _parse_abstract_sections(article_elem)
        if not abstract_sections:
            continue

        title = _text_or_none(article_elem.find("ArticleTitle")) or "(no title)"
        journal = _text_or_none(article_elem.find("Journal/Title"))
        year = _parse_year(article_elem)
        pub_types = _parse_pub_types(article_elem)
        mesh_terms = _parse_mesh_terms(medline_citation)
        pubmed_data = pubmed_article.find("PubmedData")
        doi = _parse_doi(pubmed_data)

        records.append(
            Record(
                pmid=pmid,
                title=title,
                abstract_sections=abstract_sections,
                journal=journal,
                year=year,
                pub_types=pub_types,
                mesh_terms=mesh_terms,
                doi=doi,
            )
        )

    return records
