"""Local embedding providers selected through environment configuration."""

from __future__ import annotations

import importlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Common interface for local embedding backends."""

    name: str
    model: str
    device: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this provider."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed one text value."""

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple values, with a provider-specific override when available."""
        return [self.embed(text) for text in texts]

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "device": self.device,
            "dimension": self.dimension,
        }


class OllamaEmbeddingProvider(EmbeddingProvider):
    """The existing Ollama embedding behavior, exposed through the provider interface."""

    name = "ollama"
    device = "ollama-managed"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        dimension: int = 768,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self._dimension = dimension
        self._client = client

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = importlib.import_module("ollama")
            except ImportError as exc:
                raise RuntimeError(
                    "The Ollama provider requires the 'ollama' package from requirements.txt."
                ) from exc
        return self._client

    def embed(self, text: str) -> List[float]:
        response = self._get_client().embeddings(model=self.model, prompt=text)
        vector = response.get("embedding", [])
        if not vector:
            raise RuntimeError(f"Ollama model {self.model!r} returned no embedding")
        if len(vector) != self.dimension:
            raise RuntimeError(
                f"Ollama model {self.model!r} returned {len(vector)} dimensions; "
                f"set EMBEDDING_DIM={len(vector)} and use a matching Elasticsearch index"
            )
        return [float(value) for value in vector]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """CPU-friendly in-process embeddings using sentence-transformers."""

    name = "sentence-transformers"

    def __init__(
        self,
        model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        device: str = "cpu",
        encoder: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.device = device
        if encoder is None:
            try:
                sentence_transformers = importlib.import_module("sentence_transformers")
            except ImportError as exc:
                raise RuntimeError(
                    "The sentence-transformers provider is optional. "
                    "Install requirements-cpu.txt before selecting it."
                ) from exc
            encoder = sentence_transformers.SentenceTransformer(model, device=device)
        self._encoder = encoder
        self._dimension = int(encoder.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._encoder.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


def create_embedding_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    dimension: Optional[int] = None,
) -> EmbeddingProvider:
    """Create an embedding provider from arguments or local environment variables."""
    provider_name = (provider_name or os.getenv("EMBEDDING_PROVIDER", "ollama")).strip().lower()
    aliases = {
        "sentence_transformer": "sentence-transformers",
        "sentence_transformer_cpu": "sentence-transformers",
        "sentence-transformer": "sentence-transformers",
    }
    provider_name = aliases.get(provider_name, provider_name)

    if provider_name == "ollama":
        configured_model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_OLLAMA_MODEL
        configured_dimension = dimension or int(os.getenv("EMBEDDING_DIM", "768"))
        return OllamaEmbeddingProvider(configured_model, configured_dimension)
    if provider_name == "sentence-transformers":
        configured_model = (
            model or os.getenv("EMBEDDING_MODEL") or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        )
        configured_device = device or os.getenv("EMBEDDING_DEVICE") or "cpu"
        return SentenceTransformerEmbeddingProvider(configured_model, configured_device)
    raise ValueError(
        f"Unsupported embedding provider {provider_name!r}; "
        "choose 'ollama' or 'sentence-transformers'."
    )


_default_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = create_embedding_provider()
        logger.info("Embedding configuration: %s", _default_provider.describe())
    return _default_provider


def get_embedding(text: str) -> List[float]:
    """Compatibility wrapper used by the current storage layer."""
    try:
        return get_embedding_provider().embed(text)
    except Exception as exc:
        logger.error("Error getting embedding: %s", exc)
        return []


def get_embedding_dimension() -> int:
    return get_embedding_provider().dimension


__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "create_embedding_provider",
    "get_embedding",
    "get_embedding_dimension",
    "get_embedding_provider",
]
