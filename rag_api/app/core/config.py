from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    API_KEY: str = "local-key"

    APP_NAME: str = "Local IMC Chatbot"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "imc_corpus2"

    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:14b"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024

    VECTOR_SIZE: int = 1024
    LOG_LEVEL: str = "INFO"

    TOP_K: int = 5
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    TRANSFORMERS_OFFLINE: int = int(os.getenv("TRANSFORMERS_OFFLINE", "0"))
    HF_HOME: str = os.getenv("HF_HOME", "/models")
    
    DOC_BASE_URL: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()