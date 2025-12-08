from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebCfg(BaseModel):
    enabled: bool = False
    seeds: List[str] = ["https://constructor.university/"]
    allowed_domain: str = "constructor.university"
    max_pages: int = 150
    rate_limit_seconds: float = 0.8


class YamlCfg(BaseModel):
    data: Dict[str, Any]
    indexing: Dict[str, Any]
    retrieval: Dict[str, Any]
    models: Dict[str, Any]
    web: Dict[str, Any] = {}


class Settings(BaseSettings):
    """
    Secrets come from .env, non-secrets from config.yaml.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Secrets / env-driven
    openai_api_key: str
    qdrant_url: str
    qdrant_collection: str = "constructor_kb"
    qdrant_api_key: Optional[str] = None
    embedding_dim: Optional[int] = None
    
    # Non-secrets / yaml-driven
    bachelors_excel: Path = Path("Data/Bachelors_programs.xlsx")
    masters_excel: Path = Path("Data/Masters_programs.xlsx")
    pdf_bachelor_dir: Path = Path("Data/PDF/Bachelor")
    pdf_masters_dir: Path = Path("Data/PDF/Masters")
    web_cache_dir: Path = Path("Data/web_cache")
    artifacts_dir: Path = Path("Data/artifacts")

    chunk_chars: int = 2800
    chunk_overlap: int = 350
    max_chunks_per_doc: int = 400

    top_k: int = 10
    min_score: float = 0.15

    embed_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1-mini"

    web: WebCfg = WebCfg()

    @classmethod
    def load(cls, config_path: Path = Path("config.yaml")) -> "Settings":
        if not config_path.exists():
            return cls()

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        cfg = YamlCfg(**raw)

        data = cfg.data
        indexing = cfg.indexing
        retrieval = cfg.retrieval
        models = cfg.models
        web = cfg.web or {}

        return cls(
            bachelors_excel=data["bachelors_excel"],
            masters_excel=data["masters_excel"],
            pdf_bachelor_dir=data["pdf_bachelor_dir"],
            pdf_masters_dir=data["pdf_masters_dir"],
            web_cache_dir=data["web_cache_dir"],
            artifacts_dir=data["artifacts_dir"],
            chunk_chars=indexing["chunk_chars"],
            chunk_overlap=indexing["chunk_overlap"],
            max_chunks_per_doc=indexing["max_chunks_per_doc"],
            top_k=retrieval["top_k"],
            min_score=retrieval["min_score"],
            embed_model=models["embed_model"],
            chat_model=models["chat_model"],
            web=web,
        )
