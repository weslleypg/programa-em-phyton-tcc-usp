"""
predictor.py
============
Entry point do pipeline de predição multi-modelo.

Uso (fora do Docker):
    python predictor.py                                     # perguntará a pasta
    python predictor.py --pasta 22042026_090050
    python predictor.py --pasta 22042026_090050 --force
    python predictor.py --pasta 22042026_090050 --apenas-consolidar

Uso (Docker):
    docker compose run --rm preditor
    docker compose run --rm preditor --pasta 22042026_090050
    docker compose run --rm preditor --pasta 22042026_090050 --apenas-consolidar

Modos:
    padrão               → chama os 4 LLMs (pula JSONs já existentes)
    --force              → chama os 4 LLMs (reprocessa tudo, gasta API)
    --apenas-consolidar  → pula chamadas LLM, só regenera o CSV consolidado
                          a partir dos JSONs já existentes
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config import settings as pipeline_settings
from src.predictor import (
    PredictorSettings,
    run_prediction_pipeline,
)
from src.utils import setup_logging

console = Console()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        console.print(f"[red]Variável obrigatória '{name}' não configurada no .env[/red]")
        sys.exit(1)
    return value if value is not None else ""


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_predictor_settings() -> PredictorSettings:
    """Lê todas as variáveis necessárias do .env."""
    return PredictorSettings(
        # Credenciais
        openai_api_key=_get_env("OPENAI_API_KEY", required=True),
        anthropic_api_key=_get_env("ANTHROPIC_API_KEY", required=True),
        gemini_api_key=_get_env("GEMINI_API_KEY", required=True),
        groq_api_key=_get_env("GROQ_API_KEY", required=True),
        # Modelos (defaults conservadores; usuário sobrescreve no .env)
        openai_model=_get_env("OPENAI_MODEL", default="gpt-5.4-pro"),
        anthropic_model=_get_env("ANTHROPIC_MODEL", default="claude-opus-4-7"),
        gemini_model=_get_env("GEMINI_MODEL", default="gemini-3.1-pro-preview"),
        groq_model=_get_env("GROQ_MODEL", default="meta-llama/llama-4-scout-17b-16e-instruct"),
        # Comportamento
        max_retries=_get_int("PREDICTOR_MAX_RETRIES", 5),
        retry_initial_wait=_get_int("PREDICTOR_RETRY_INITIAL_WAIT", 4),
        retry_max_wait=_get_int("PREDICTOR_RETRY_MAX_WAIT", 60),
        groq_delay_between_calls=_get_int("GROQ_DELAY_BETWEEN_CALLS", 10),
        request_timeout=_get_int("PREDICTOR_REQUEST_TIMEOUT", 180),
        # OpenAI — roteamento híbrido entre Chat Completions e Responses API
        openai_api_flavor=_get_env("OPENAI_API_FLAVOR", default="auto"),
        openai_reasoning_effort=_get_env("OPENAI_REASONING_EFFORT", default="medium"),
    )


def load_prompt() -> str:
    """Lê prompts/prompt_preditivo.txt."""
    prompt_path = pipeline_settings.PROJECT_ROOT / "prompts" / "prompt_preditivo.txt" \
        if hasattr(pipeline_settings, "PROJECT_ROOT") \
        else Path("prompts/prompt_preditivo.txt")

    # Fallback: relativo à raiz do projeto pelo próprio __file__
    if not prompt_path.is_file():
        prompt_path = Path(__file__).resolve().parent / "prompts" / "prompt_preditivo.txt"

    if not prompt_path.is_file():
        console.print(f"[red]prompt_preditivo.txt não encontrado em {prompt_path}[/red]")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8").strip()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline de Predição Multi-Modelo (OpenAI + Anthropic + Gemini + Groq).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pasta",
        metavar="DDMMAAAA_HHMMSS",
        help="Nome da pasta dentro de CSV/anonimizados/. Se omitido, será perguntado.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa todas as chamadas mesmo que o JSON já exista em disco.",
    )
    p.add_argument(
        "--apenas-consolidar",
        action="store_true",
        help="Não chama nenhum LLM; apenas reconstrói o consolidado a partir dos JSONs.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log (default: INFO).",
    )
    return p


def perguntar_pasta(base: Path) -> str:
    """Lista pastas disponíveis e pergunta ao usuário qual usar."""
    if not base.is_dir():
        console.print(f"[red]Pasta base não existe: {base}[/red]")
        sys.exit(2)

    disponiveis = sorted([d.name for d in base.iterdir() if d.is_dir()])
    if not disponiveis:
        console.print(f"[red]Nenhuma pasta em {base}. Rode o pipeline de anonimização antes.[/red]")
        sys.exit(2)

    console.print(f"\n[bold cyan]Pastas disponíveis em {base}:[/bold cyan]")
    for idx, nome in enumerate(disponiveis, start=1):
        console.print(f"  [{idx}] {nome}")

    while True:
        raw = input("\nEscolha pelo número ou digite o nome: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(disponiveis):
                return disponiveis[idx - 1]
            console.print("[red]Índice fora da faixa.[/red]")
            continue
        if raw in disponiveis:
            return raw
        console.print(f"[red]'{raw}' não existe. Tente de novo.[/red]")


def print_header(pasta: str, settings: PredictorSettings, modo: str) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]TCC — Pipeline de Predição Multi-Modelo[/bold cyan]\n"
            f"[dim]Predição cruzada com 4 LLMs[/dim]\n\n"
            f"Pasta de trabalho:  [yellow]{pasta}[/yellow]\n"
            f"Modo:               [yellow]{modo}[/yellow]\n\n"
            f"Modelos configurados:\n"
            f"  OpenAI:     [yellow]{settings.openai_model}[/yellow]\n"
            f"  Anthropic:  [yellow]{settings.anthropic_model}[/yellow]\n"
            f"  Gemini:     [yellow]{settings.gemini_model}[/yellow]\n"
            f"  Groq:       [yellow]{settings.groq_model}[/yellow]\n\n"
            f"Groq delay:         [yellow]{settings.groq_delay_between_calls}s[/yellow] entre chamadas\n"
            f"Max retries:        [yellow]{settings.max_retries}[/yellow] por chamada",
            border_style="cyan",
        )
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(pipeline_settings.LOGS_DIR, level=log_level)

    # Carrega configs do .env
    try:
        settings = load_predictor_settings()
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Erro na configuração:[/red] {e}")
        return 2

    # Carrega o prompt preditivo
    system_prompt = load_prompt()

    # Resolve pasta de trabalho
    base_anonimizados = pipeline_settings.CSV_OUTPUT_DIR / "anonimizados"
    if args.pasta:
        pasta_nome = args.pasta
    else:
        pasta_nome = perguntar_pasta(base_anonimizados)

    pasta_trabalho = base_anonimizados / pasta_nome
    if not pasta_trabalho.is_dir():
        console.print(f"[red]Pasta não existe: {pasta_trabalho}[/red]")
        return 2

    modo = (
        "apenas-consolidar" if args.apenas_consolidar
        else ("force (reprocessa tudo)" if args.force else "padrão (usa cache de JSONs)")
    )
    print_header(pasta_nome, settings, modo)

    try:
        output = run_prediction_pipeline(
            pasta_trabalho=pasta_trabalho,
            system_prompt=system_prompt,
            settings=settings,
            force=args.force,
            apenas_consolidar=args.apenas_consolidar,
        )
    except FileNotFoundError as e:
        console.print(f"\n[red]Erro:[/red] {e}")
        return 2
    except RuntimeError as e:
        console.print(f"\n[red]Erro:[/red] {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrompido pelo usuário.[/yellow]")
        return 130

    console.print(
        Panel.fit(
            f"[bold green]✓ Pipeline preditor concluído.[/bold green]\n\n"
            f"Consolidado:  [cyan]{output}[/cyan]\n"
            f"JSONs:        [cyan]{pasta_trabalho / 'json'}[/cyan]",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
