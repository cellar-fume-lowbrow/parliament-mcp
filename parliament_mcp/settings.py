import logging
import os
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


@lru_cache
def get_ssm_parameter(parameter_name: str, region: str = "eu-west-2") -> str:
    """Fetch a parameter from AWS Systems Manager Parameter Store."""
    try:
        ssm = boto3.client("ssm", region_name=region)
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not fetch SSM parameter %s: %s", parameter_name, e)
        return ""


def get_environment_or_ssm(env_var_name: str, ssm_path: str | None = None, default: str = "") -> str:
    """Get value from environment variable or fall back to SSM parameter."""
    env_value = os.environ.get(env_var_name)
    if env_value:
        return env_value

    # Only use SSM if not in local environment
    environment = os.environ.get("ENVIRONMENT", "local")
    if ssm_path and os.environ.get("AWS_REGION") and environment != "local":
        return get_ssm_parameter(ssm_path, os.environ.get("AWS_REGION"))

    return default


class ParliamentMCPSettings(BaseSettings):
    """Configuration settings for Parliament MCP application with environment-based loading."""

    AWS_ACCOUNT_ID: str | None = None
    AWS_REGION: str = "eu-west-2"
    ENVIRONMENT: str = "local"

    # Use SSM for sensitive parameters in AWS environments
    @property
    def SENTRY_DSN(self) -> str | None:
        return get_environment_or_ssm("SENTRY_DSN", f"/{self._get_project_name()}/env_secrets/SENTRY_DSN")

    @property
    def AZURE_OPENAI_API_KEY(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_API_KEY",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_API_KEY",
        )

    @property
    def AZURE_OPENAI_ENDPOINT(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_ENDPOINT",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_ENDPOINT",
        )

    @property
    def AZURE_OPENAI_EMBEDDING_MODEL(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_EMBEDDING_MODEL",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_EMBEDDING_MODEL",
        )

    @property
    def AZURE_OPENAI_API_VERSION(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_API_VERSION",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_API_VERSION",
            "preview",
        )

    # Embedding provider selection.
    #   azure  -> uses the AZURE_OPENAI_* settings above (upstream default)
    #   openai -> any OpenAI-compatible endpoint: OpenRouter, OpenAI, Ollama (/v1), LiteLLM
    EMBEDDING_PROVIDER: str = "azure"
    EMBEDDING_BASE_URL: str | None = None
    EMBEDDING_APP_TITLE: str = "parliament-mcp"
    # Only send `dimensions` for models that support truncation (text-embedding-3-*).
    # Set false for fixed-size models such as bge-m3.
    EMBEDDING_SEND_DIMENSIONS: bool = True

    @property
    def EMBEDDING_API_KEY(self) -> str:
        return get_environment_or_ssm(
            "EMBEDDING_API_KEY",
            f"/{self._get_project_name()}/env_secrets/EMBEDDING_API_KEY",
        )

    # Template applied to search queries only (never to ingested documents), for
    # instruction-aware / asymmetric models. Must contain {query}; a literal \n in
    # the value is converted to a newline. Empty = embed queries as-is.
    # qwen3-embedding: "Instruct: <task>\nQuery:{query}"
    EMBEDDING_QUERY_TEMPLATE: str = ""

    def format_query_for_embedding(self, query: str) -> str:
        template = self.EMBEDDING_QUERY_TEMPLATE.replace("\\n", "\n")  # literal backslash-n -> newline
        if not template:
            return query
        if "{query}" not in template:
            msg = "EMBEDDING_QUERY_TEMPLATE must contain {query}"
            raise ValueError(msg)
        return template.replace("{query}", query)  # not str.format: user queries may contain braces

    # Model slug for EMBEDDING_PROVIDER=openai, e.g. baai/bge-m3 (OpenRouter) or bge-m3 (Ollama)
    EMBEDDING_MODEL: str | None = None

    @property
    def embedding_model(self) -> str:
        """Model name for the active provider (falls back to the Azure deployment name)."""
        return self.EMBEDDING_MODEL or self.AZURE_OPENAI_EMBEDDING_MODEL

    # Qdrant connection settings
    @property
    def QDRANT_URL(self) -> str | None:
        return get_environment_or_ssm("QDRANT_URL", f"/{self._get_project_name()}/env_secrets/QDRANT_URL")

    @property
    def QDRANT_API_KEY(self) -> str | None:
        return get_environment_or_ssm("QDRANT_API_KEY", f"/{self._get_project_name()}/env_secrets/QDRANT_API_KEY")

    AUTH_PROVIDER_PUBLIC_KEY: str | None = None
    DISABLE_AUTH_SIGNATURE_VERIFICATION: bool = ENVIRONMENT == "local"

    def _get_project_name(self) -> str:
        """Get the project name from environment or use default."""
        return os.environ.get("PROJECT_NAME", "i-dot-ai-dev-parliament-mcp")

    # Qdrant collection names
    QDRANT_COLLECTION_PREFIX: str = "parliament_mcp_"

    EMBEDDING_DIMENSIONS: int = 1024

    # Sparse text embedding model
    SPARSE_TEXT_EMBEDDING_MODEL: str = "Qdrant/bm25"

    # Chunking settings
    # See https://www.elastic.co/search-labs/blog/elasticsearch-chunking-inference-api-endpoints
    CHUNK_SIZE: int = 300
    SENTENCE_OVERLAP: int = 1
    CHUNK_STRATEGY: str = "sentence"

    PARLIAMENTARY_QUESTIONS_COLLECTION: str = "parliament_mcp_parliamentary_questions"
    HANSARD_CONTRIBUTIONS_COLLECTION: str = "parliament_mcp_hansard_contributions"

    # MCP settings
    MCP_HOST: str = "0.0.0.0"  # nosec B104 - Binding to all interfaces is intentional for containerized deployment
    MCP_PORT: int = 8080

    # The MCP server can be accessed at /{MCP_ROOT_PATH}/mcp
    MCP_ROOT_PATH: str = "/"

    # Allowed hosts for MCP transport security (comma-separated)
    # Used to prevent DNS rebinding attacks
    MCP_ALLOWED_HOSTS: str = "localhost,127.0.0.1"

    # Rate limiting settings for parliament.uk API.
    HTTP_MAX_RATE_PER_SECOND: float = 10
    # Retries for transient parliament.uk API failures (429, 5xx, timeouts, dropped connections)
    HTTP_MAX_ATTEMPTS: int = 5

    # Load environment variables from .env file in local environment
    # from pydantic_settings import SettingsConfigDict
    if ENVIRONMENT == "local":
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = ParliamentMCPSettings()
