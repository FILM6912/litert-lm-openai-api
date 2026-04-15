from functools import lru_cache
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    litert_model_path: str = "models/gemma-4-E2B-it.litertlm"
    openai_model_id: str = "gemma-litert"
    litert_model_download_dir: str = os.getcwd()
    litert_admin_token: str = ""
    litert_model_scan_max_depth: int = 2
    litert_model_catalog_scan_timeout_sec: float = 5.0
    litert_model_scan_subdir: str = "models"
    litert_model_scan_extra_skip: str = ""
    litert_model_catalog_ttl_sec: float = 3.0

    class Config:
        env_prefix = ""
        case_sensitive = False
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

DEFAULT_MODEL_PATH = settings.litert_model_path
DEFAULT_MODEL_ID = settings.openai_model_id
MODEL_DOWNLOAD_ROOT = os.path.realpath(settings.litert_model_download_dir)
ADMIN_TOKEN = settings.litert_admin_token.strip()
LITERT_SCAN_MAX_DEPTH = max(0, settings.litert_model_scan_max_depth)
LITERT_CATALOG_SCAN_TIMEOUT = float(settings.litert_model_catalog_scan_timeout_sec)
LITERT_SCAN_SUBDIR = settings.litert_model_scan_subdir.strip()
LITERT_CATALOG_TTL = float(settings.litert_model_catalog_ttl_sec)

_SKIP_BASE = frozenset(
    ".git .svn .hg node_modules .cursor __pycache__ .venv .uv venv env .env "
    ".mypy_cache .pytest_cache .tox .nox dist build .idea .vscode "
    "anaconda3 miniconda3 conda pkgs site-packages .eggs "
    "htmlcov .hypothesis .ruff_cache .gradle target".split()
)
_EXTRA_SKIP = frozenset(
    x.strip() for x in settings.litert_model_scan_extra_skip.split(",") if x.strip()
)
LITERT_SCAN_SKIP: frozenset[str] = _SKIP_BASE | _EXTRA_SKIP
