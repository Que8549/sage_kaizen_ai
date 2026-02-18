
from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

load_dotenv()

def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


@dataclass(frozen=True)
class ServerConfig:
    # Endpoints
    q5_base_url: str = _env("SAGE_Q5_BASE_URL", "http://127.0.0.1:8011")
    q6_base_url: str = _env("SAGE_Q6_BASE_URL", "http://127.0.0.1:8012")
    embed_base_url: str = _env("SAGE_EMBED_BASE_URL", "http://127.0.0.1:8020")
    

    # Model ids (optional; client can discover via /v1/models)
    q5_model_id: str = _env("SAGE_Q5_MODEL_ID", "Q5")
    q6_model_id: str = _env("SAGE_Q6_MODEL_ID", "Q6")
    embedded_model_id: str = _env("SAGE_EMBED_MODEL_ID", "EMBED")

    # Prompt + history
    system_prompt_path: Path = Path(_env("SAGE_SYSTEM_PROMPT_PATH", "./sage_kaizen_system_prompt.txt"))
    max_history_messages: int = int(_env("SAGE_MAX_HISTORY_MESSAGES", "20"))

    # Networking
    connect_timeout_s: float = float(_env("SAGE_CONNECT_TIMEOUT_S", "3"))
    read_timeout_s: float = float(_env("SAGE_READ_TIMEOUT_S", "900"))  # up to 15 minutes
    # Streaming chunk handling
    stream_keepalive_s: float = float(_env("SAGE_STREAM_KEEPALIVE_S", "30"))


CONFIG = ServerConfig()
