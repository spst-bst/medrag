from __future__ import annotations

from typing import Optional, Protocol

import httpx
import numpy as np

EMBED_BATCH_SIZE = 16
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder(Protocol):
    def embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        client: Optional[httpx.Client] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.client = client or httpx.Client(timeout=600)

    def embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        prefixed = [f"{prefix}{t}" for t in texts]
        resp = self.client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": prefixed},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def embed_documents_batched(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
        return self._embed_batched(texts, DOCUMENT_PREFIX, batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], QUERY_PREFIX)[0]

    def _embed_batched(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out.extend(self.embed(batch, prefix))
        return out


def vector_to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1e-10
    return (matrix @ query_vec) / denom
