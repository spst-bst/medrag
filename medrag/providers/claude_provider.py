from __future__ import annotations

import time
from typing import Callable, Optional

from medrag.citations import enforce_citation_rules, is_not_found_reply
from medrag.config import anthropic_effort, anthropic_fallbacks_enabled, anthropic_model
from medrag.models import RetrievedChunk
from medrag.providers.base import Answer, Citation, SYSTEM_PROMPT, Usage

INSTALL_MESSAGE = (
    "The 'anthropic' package is not installed. Install it with: "
    ".venv/bin/python -m pip install anthropic"
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _import_anthropic():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(INSTALL_MESSAGE) from e
    return anthropic


def _document_block(chunk: RetrievedChunk) -> dict:
    pub_type = ", ".join(chunk.pub_types) if chunk.pub_types else "Unknown"
    title = f"PMID:{chunk.pmid} | {chunk.title} | {chunk.year or 'n.d.'} | {pub_type}"
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
        "title": title,
        "citations": {"enabled": True},
    }


class ClaudeProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        fallbacks_enabled: Optional[bool] = None,
        client_factory: Optional[Callable[[], object]] = None,
    ):
        self.model = model or anthropic_model()
        self.effort = effort or anthropic_effort()
        self.fallbacks_enabled = (
            fallbacks_enabled if fallbacks_enabled is not None else anthropic_fallbacks_enabled()
        )
        self._client_factory = client_factory

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Answer:
        start = time.monotonic()
        retrieved_pmids = {c.pmid for c in chunks}

        try:
            anthropic = _import_anthropic()
        except RuntimeError as e:
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[str(e)],
            )

        client = self._client_factory() if self._client_factory is not None else anthropic.Anthropic()

        documents = [_document_block(c) for c in chunks]
        content = documents + [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]

        request_kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": self.effort},
        )

        warnings: list[str] = []
        msg = None
        try:
            if self.fallbacks_enabled:
                try:
                    with client.beta.messages.stream(
                        **request_kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                    ) as stream:
                        msg = stream.get_final_message()
                except TypeError as e:
                    warnings.append(
                        f"Installed SDK rejected server-side-fallback parameters ({e}); "
                        "falling back to the plain (non-beta) call."
                    )
            if msg is None:
                with client.messages.stream(**request_kwargs) as stream:
                    msg = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[
                    "Authentication failed. Set ANTHROPIC_API_KEY in your shell environment "
                    f"(never pass it on the command line). request_id={getattr(e, 'request_id', None)}"
                ],
            )
        except anthropic.NotFoundError as e:
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[f"Model not found: '{self.model}'. {e}"],
            )
        except anthropic.RateLimitError as e:
            retry_after = None
            resp = getattr(e, "response", None)
            if resp is not None:
                retry_after = resp.headers.get("retry-after")
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[
                    f"Rate limited by Anthropic API. retry_after={retry_after}. "
                    f"request_id={getattr(e, 'request_id', None)}"
                ],
            )
        except anthropic.APIStatusError as e:
            kind = "server error" if e.status_code >= 500 else "API error"
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[
                    f"Anthropic {kind} {e.status_code}: {getattr(e, 'message', str(e))}. "
                    f"request_id={getattr(e, 'request_id', None)}"
                ],
            )
        except anthropic.APIConnectionError as e:
            return Answer(
                text="",
                status="error",
                citations=[],
                model=self.model,
                provider="anthropic",
                seconds=time.monotonic() - start,
                warnings=[f"Connection error reaching Anthropic API: {e}"],
            )

        served_model = getattr(msg, "model", self.model)
        if served_model != self.model:
            warnings.append(f"served by {served_model} after a fallback")

        stop_reason = getattr(msg, "stop_reason", None)
        status = "answered"
        if stop_reason == "refusal":
            status = "refused"
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            explanation = getattr(details, "explanation", None) if details is not None else None
            warnings.append(
                f"Refused by model. category={category or 'unknown'} "
                f"explanation={explanation or 'none provided'}. "
                "Try rephrasing the question, or use --provider ollama for this one."
            )
        elif stop_reason == "max_tokens":
            status = "truncated"
            warnings.append("Response was truncated at max_tokens.")
        elif stop_reason not in ("end_turn", None):
            warnings.append(f"Unexpected stop_reason: {stop_reason}")

        text_parts: list[str] = []
        citations: list[Citation] = []
        for block in getattr(msg, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                continue
            if btype != "text":
                continue
            block_text = getattr(block, "text", "") or ""
            text_parts.append(block_text)

            block_pmids_in_order: list[str] = []
            for cit in getattr(block, "citations", None) or []:
                doc_idx = getattr(cit, "document_index", None)
                cited_text = getattr(cit, "cited_text", None)
                if doc_idx is None or doc_idx < 0 or doc_idx >= len(chunks):
                    warnings.append(f"Citation referenced out-of-range document_index={doc_idx}")
                    continue
                source_chunk = chunks[doc_idx]
                pmid = source_chunk.pmid
                if cited_text is not None:
                    normalized_quote = " ".join(cited_text.split())
                    normalized_chunk_text = " ".join(source_chunk.text.split())
                    if normalized_quote not in normalized_chunk_text:
                        warnings.append(
                            f"Citation quote mismatch for PMID:{pmid}: quote not found verbatim in source chunk"
                        )
                citations.append(Citation(pmid=pmid, quote=cited_text))
                if pmid not in block_pmids_in_order:
                    block_pmids_in_order.append(pmid)

            for pmid in block_pmids_in_order:
                text_parts.append(f" [PMID:{pmid}]")

        full_text = "".join(text_parts)
        if status == "answered" and is_not_found_reply(full_text):
            status = "not_found"

        usage_obj = getattr(msg, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = Usage(
                input_tokens=getattr(usage_obj, "input_tokens", None),
                output_tokens=getattr(usage_obj, "output_tokens", None),
                cache_creation_input_tokens=getattr(usage_obj, "cache_creation_input_tokens", None),
                cache_read_input_tokens=getattr(usage_obj, "cache_read_input_tokens", None),
            )

        answer = Answer(
            text=full_text,
            status=status,
            citations=citations,
            model=served_model,
            provider="anthropic",
            seconds=time.monotonic() - start,
            usage=usage,
            warnings=warnings,
        )
        enforce_citation_rules(answer, retrieved_pmids)
        return answer
