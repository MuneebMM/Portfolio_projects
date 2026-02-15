"""Centralized configuration loaded from .env file."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # Cohere
    cohere_api_key: str = Field(..., alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(
        "rerank-english-v3.0", alias="COHERE_RERANK_MODEL"
    )

    # ChromaDB
    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("hybrid_rag", alias="CHROMA_COLLECTION_NAME")

    # Retrieval
    dense_top_k: int = Field(10, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(10, alias="SPARSE_TOP_K")
    rerank_top_k: int = Field(5, alias="RERANK_TOP_K")

    # Chunking
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(50, alias="CHUNK_OVERLAP")

    # App
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    data_dir: str = Field("./data", alias="DATA_DIR")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
