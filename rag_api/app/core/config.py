from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    API_KEY: str = "local-key"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "imc_corpus2"

    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:14b"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024

    TOP_K: int = 5
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    TRANSFORMERS_OFFLINE: int = int(os.getenv("TRANSFORMERS_OFFLINE", "0"))
    HF_HOME: str = os.getenv("HF_HOME", "/models")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()