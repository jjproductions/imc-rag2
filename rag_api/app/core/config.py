from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    API_KEY: str = "local-key"

    APP_NAME: str = "Local IMC Chatbot"

    QDRANT_URL: str = "http://localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "imc_corpus_hybrid"
    QDRANT_TIMEOUT: float = 30.0

    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # LLM general settings
    LLM_PROVIDER: str = "azure_openai"  # "azure_openai" or "ollama"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024

    # Ollama-specific (Left for legacy fallback if ever needed)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:14b"

    # Azure OpenAI-specific
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o-mini"

    @property
    def ACTIVE_LLM_MODEL(self) -> str:
        if self.LLM_PROVIDER == "azure_openai":
            return self.AZURE_OPENAI_DEPLOYMENT_NAME
        return self.OLLAMA_MODEL

    VECTOR_SIZE: int = 1024
    LOG_LEVEL: str = "INFO"

    TOP_K: int = 5
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    TRANSFORMERS_OFFLINE: int = int(os.getenv("TRANSFORMERS_OFFLINE", "0"))
    HF_HOME: str = os.getenv("HF_HOME", "/models")
    FASTEMBED_CACHE_PATH: str = os.getenv("FASTEMBED_CACHE_PATH", "/models/fastembed")
    
    DOC_BASE_URL: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()