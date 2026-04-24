"""
config.py
=========
Módulo central de configuração do projeto.

Responsabilidades:
    - Carregar variáveis do arquivo .env.
    - Validar presença das variáveis obrigatórias.
    - Expor constantes tipadas para o resto da aplicação.

Uso:
    from config import settings
    print(settings.GEMINI_MODEL)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env da raiz do projeto. Se não existir, `load_dotenv` apenas não faz nada —
# permitindo também que as variáveis venham do ambiente do container Docker.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """Lê uma variável de ambiente com tratamento de erro e default."""
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == "" or value == "cole_sua_chave_aqui"):
        print(f"[ERRO] Variável obrigatória '{name}' não configurada no .env", file=sys.stderr)
        sys.exit(1)
    return value if value is not None else ""


def _get_int(name: str, default: int) -> int:
    """Lê variável de ambiente como inteiro, com fallback seguro."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[AVISO] '{name}' inválido ('{raw}'), usando default {default}", file=sys.stderr)
        return default


@dataclass(frozen=True)
class Settings:
    """Estrutura imutável com todas as configurações do projeto."""

    # Credenciais e modelo
    GEMINI_API_KEY: str
    GEMINI_MODEL: str

    # Caminhos (absolutos, resolvidos a partir da raiz do projeto)
    PROMPT_PATH: Path
    PDF_ROOT_DIR: Path
    CSV_OUTPUT_DIR: Path
    LOGS_DIR: Path

    # Retry / rate limit
    MAX_RETRIES: int
    RETRY_INITIAL_WAIT: int
    RETRY_MAX_WAIT: int
    DELAY_BETWEEN_CALLS: int

    # Timeouts da File API
    FILE_ACTIVE_TIMEOUT: int
    FILE_POLL_INTERVAL: int


def _resolve(path_str: str) -> Path:
    """Resolve caminho relativo à raiz do projeto (absoluto continua absoluto)."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


settings = Settings(
    GEMINI_API_KEY=_get_env("GEMINI_API_KEY", required=True),
    GEMINI_MODEL=_get_env("GEMINI_MODEL", default="gemini-3.1-pro-preview"),
    PROMPT_PATH=_resolve(_get_env("PROMPT_PATH", default="prompts/extracao_cadoc9010.txt")),
    PDF_ROOT_DIR=_resolve(_get_env("PDF_ROOT_DIR", default="PDF")),
    CSV_OUTPUT_DIR=_resolve(_get_env("CSV_OUTPUT_DIR", default="CSV")),
    LOGS_DIR=_resolve(_get_env("LOGS_DIR", default="logs")),
    MAX_RETRIES=_get_int("MAX_RETRIES", 5),
    RETRY_INITIAL_WAIT=_get_int("RETRY_INITIAL_WAIT", 4),
    RETRY_MAX_WAIT=_get_int("RETRY_MAX_WAIT", 60),
    DELAY_BETWEEN_CALLS=_get_int("DELAY_BETWEEN_CALLS", 2),
    FILE_ACTIVE_TIMEOUT=_get_int("FILE_ACTIVE_TIMEOUT", 300),
    FILE_POLL_INTERVAL=_get_int("FILE_POLL_INTERVAL", 3),
)


# Garante que diretórios de saída existam (os de entrada devem já existir no uso real)
settings.CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
