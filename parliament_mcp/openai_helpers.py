import logging
from itertools import batched

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from parliament_mcp.settings import ParliamentMCPSettings

logger = logging.getLogger(__name__)

EmbeddingClient = AsyncOpenAI | AsyncAzureOpenAI

# Retry only transient failures (429s, 5xx, timeouts). Config errors fail fast.
_retry_transient = retry(
    retry=retry_if_exception_type((RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)),
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=30),
    reraise=True,
)


def get_openai_client(settings: ParliamentMCPSettings) -> EmbeddingClient:
    """Get an async embeddings client.

    EMBEDDING_PROVIDER=azure  -> Azure OpenAI (upstream behaviour)
    EMBEDDING_PROVIDER=openai -> any OpenAI-compatible endpoint
                                 (OpenRouter, OpenAI, Ollama /v1, LiteLLM)
    """
    http_client = httpx.AsyncClient(timeout=60.0)

    if settings.EMBEDDING_PROVIDER == "azure":
        return AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            http_client=http_client,
        )

    if settings.EMBEDDING_PROVIDER == "openai":
        if not settings.EMBEDDING_BASE_URL:
            msg = "EMBEDDING_BASE_URL must be set when EMBEDDING_PROVIDER=openai"
            raise ValueError(msg)
        return AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY or "not-needed",  # Ollama ignores the key
            base_url=settings.EMBEDDING_BASE_URL,
            default_headers={"X-Title": settings.EMBEDDING_APP_TITLE},  # shows in OpenRouter activity log
            http_client=http_client,
        )

    msg = f"Unknown EMBEDDING_PROVIDER {settings.EMBEDDING_PROVIDER!r} (expected 'azure' or 'openai')"
    raise ValueError(msg)


def _check_dimensions(vectors: list[list[float]], expected: int) -> None:
    """Fail loudly rather than let Qdrant reject an upsert with an opaque error."""
    for vector in vectors:
        if len(vector) != expected:
            msg = (
                f"Embedding model returned {len(vector)} dimensions but EMBEDDING_DIMENSIONS={expected}. "
                "Change EMBEDDING_DIMENSIONS and re-run init-qdrant (existing collections must be recreated)."
            )
            raise ValueError(msg)


async def _create_embeddings(
    client: EmbeddingClient,
    texts: str | list[str],
    model: str,
    dimensions: int,
    *,
    send_dimensions: bool,
) -> list[list[float]]:
    kwargs: dict = {"input": texts, "model": model, "encoding_format": "float"}
    if send_dimensions:
        # Only Matryoshka-capable models (e.g. text-embedding-3-*) honour this.
        kwargs["dimensions"] = dimensions
    response = await client.embeddings.create(**kwargs)
    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    _check_dimensions(vectors, dimensions)
    return vectors


@_retry_transient
async def embed_single(
    client: EmbeddingClient,
    text: str,
    model: str,
    dimensions: int = 1024,
    *,
    send_dimensions: bool = True,
) -> list[float]:
    """Generate a single embedding for a text."""
    vectors = await _create_embeddings(client, text, model, dimensions, send_dimensions=send_dimensions)
    return vectors[0]


async def embed_batch(
    client: EmbeddingClient,
    texts: list[str],
    model: str,
    dimensions: int = 1024,
    batch_size: int = 100,
    *,
    send_dimensions: bool = True,
) -> list[list[float]]:
    """Generate embeddings for a list of texts, retrying each batch independently.

    Args:
        client: OpenAI-compatible async client
        texts: List of texts to embed
        model: Model name (Azure deployment name, or provider model slug)
        dimensions: Expected vector size (must match the Qdrant collection)
        batch_size: Number of texts to process in each API call
        send_dimensions: Whether to pass `dimensions` to the API

    Returns:
        List of embedding vectors
    """

    @_retry_transient
    async def _embed(batch: list[str]) -> list[list[float]]:
        return await _create_embeddings(client, batch, model, dimensions, send_dimensions=send_dimensions)

    all_embeddings = []
    for i, batch in enumerate(batched(texts, batch_size)):
        try:
            all_embeddings.extend(await _embed(list(batch)))
        except Exception:
            logger.exception("Error generating embeddings for batch %d", i + 1)
            raise

    return all_embeddings
