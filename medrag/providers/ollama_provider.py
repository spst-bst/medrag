from __future__ import annotations

import json
import time
from typing import Callable, Optional

import httpx

from medrag.citations import enforce_citation_rules, extract_pmid_markers, is_not_found_reply
from medrag.config import OLLAMA_BASE_URL, chat_model
from medrag.models import RetrievedChunk
from medrag.providers.base import Answer, Citation, SYSTEM_PROMPT

CITATION_FORMAT_INSTRUCTION = (
    "\n\nCitation format: after every factual sentence, cite its source excerpt using exactly "
    "this marker syntax: [PMID:12345678] (the PMID digits of the excerpt, no other text inside "
    "the brackets). Use this exact bracket format every time, not \"Reference:\" or parentheses."
)


class OllamaProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = OLLAMA_BASE_URL,
        client: Optional[httpx.Client] = None,
    ):
        self.model = model or chat_model()
        self.base_url = base_url
        self.client = client or httpx.Client(timeout=600)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Answer:
        start = time.monotonic()
        retrieved_pmids = {c.pmid for c in chunks}

        context = "\n\n---\n\n".join(c.text for c in chunks)
        user_content = f"Retrieved excerpts:\n\n{context}\n\nQuestion: {question}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + CITATION_FORMAT_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
            "options": {"num_ctx": 8192, "temperature": 0.1},
        }

        text_parts: list[str] = []
        try:
            with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    message = obj.get("message", {})
                    content = message.get("content")
                    if content:
                        text_parts.append(content)
                        if on_progress:
                            on_progress(content)
                    if obj.get("done"):
                        break
        except httpx.HTTPError as e:
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="ollama",
                seconds=time.monotonic() - start,
                warnings=[f"Ollama request failed: {e}"],
            )

        full_text = "".join(text_parts)
        pmids = list(dict.fromkeys(extract_pmid_markers(full_text)))
        citations = [Citation(pmid=p, quote=None) for p in pmids]
        status = "not_found" if is_not_found_reply(full_text) else "answered"

        answer = Answer(
            text=full_text,
            status=status,
            citations=citations,
            model=self.model,
            provider="ollama",
            seconds=time.monotonic() - start,
            usage=None,
            warnings=[],
        )
        enforce_citation_rules(answer, retrieved_pmids)
        return answer
