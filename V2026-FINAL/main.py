"""
main.py
=======
Ponto de entrada (CLI) do projeto.

Uso básico (depois de configurar o .env):
    python main.py                      # processa tudo, reaproveita cache
    python main.py --force              # reprocessa TUDO (ignora JSONs existentes)
    python main.py --only 13140088      # processa apenas uma instituição
    python main.py --dry-run            # só lista o que seria processado

No Docker:
    docker compose run --rm app                     # equivalente ao primeiro
    docker compose run --rm app --force             # reprocessa tudo
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel

from config import settings
from src.gemini_client import GeminiClient
from src.institution_processor import (
    discover_pdfs_grouped_by_cnpj,
    run_pipeline,
)
from src.utils import setup_logging

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Processa PDFs CADOC 9010 de Instituições de Pagamento via Gemini API "
            "e gera CSVs consolidados por instituição."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa todos os PDFs, mesmo os que já têm JSON salvo em disco.",
    )
    p.add_argument(
        "--only",
        metavar="CNPJ8",
        help="Processa apenas uma instituição (informe os 8 primeiros dígitos do CNPJ).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Não chama a API; apenas lista o que seria processado.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log (default: INFO).",
    )
    return p


def load_prompt() -> str:
    """Lê o prompt do arquivo de texto configurado."""
    if not settings.PROMPT_PATH.is_file():
        console.print(f"[red]Prompt não encontrado em {settings.PROMPT_PATH}[/red]")
        sys.exit(1)
    return settings.PROMPT_PATH.read_text(encoding="utf-8").strip()


def print_header() -> None:
    """Mostra o cabeçalho da execução para o usuário."""
    console.print(
        Panel.fit(
            "[bold cyan]TCC — Análise de Instituições de Pagamento[/bold cyan]\n"
            "[dim]Pipeline de extração via Gemini API[/dim]\n\n"
            f"Modelo:       [yellow]{settings.GEMINI_MODEL}[/yellow]\n"
            f"PDFs:         [yellow]{settings.PDF_ROOT_DIR}[/yellow]\n"
            f"Saída CSV:    [yellow]{settings.CSV_OUTPUT_DIR}[/yellow]\n"
            f"Prompt:       [yellow]{settings.PROMPT_PATH}[/yellow]\n"
            f"Max retries:  [yellow]{settings.MAX_RETRIES}[/yellow]",
            border_style="cyan",
        )
    )


def do_dry_run() -> int:
    """Lista os PDFs que seriam processados, sem chamar a API."""
    groups = discover_pdfs_grouped_by_cnpj(settings.PDF_ROOT_DIR)
    console.print(f"\n[bold]Encontradas {len(groups)} instituições:[/bold]\n")
    for cnpj8 in sorted(groups):
        pdfs = sorted(groups[cnpj8], key=lambda it: it[1].sort_key)
        console.print(f"  [cyan]{cnpj8}[/cyan] — {len(pdfs)} arquivo(s):")
        for _pdf_path, meta in pdfs:
            console.print(f"    • {meta.semestre_label}  {meta.filename}")
    console.print()
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()

    log_level = getattr(logging, args.log_level)
    setup_logging(settings.LOGS_DIR, level=log_level)
    log = logging.getLogger(__name__)

    print_header()

    if args.dry_run:
        console.print("[yellow]Modo dry-run ativo — nenhuma chamada será feita à API.[/yellow]\n")
        return do_dry_run()

    prompt_text = load_prompt()

    client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
        prompt=prompt_text,
        max_retries=settings.MAX_RETRIES,
        retry_initial_wait=settings.RETRY_INITIAL_WAIT,
        retry_max_wait=settings.RETRY_MAX_WAIT,
        file_active_timeout=settings.FILE_ACTIVE_TIMEOUT,
        file_poll_interval=settings.FILE_POLL_INTERVAL,
    )

    start = time.monotonic()
    try:
        stats = run_pipeline(
            pdf_root=settings.PDF_ROOT_DIR,
            csv_output_dir=settings.CSV_OUTPUT_DIR,
            client=client,
            force=args.force,
            delay_between_calls=settings.DELAY_BETWEEN_CALLS,
            only_cnpj8=args.only,
        )
    except FileNotFoundError as e:
        log.error(str(e))
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Execução interrompida pelo usuário.[/yellow]")
        return 130

    elapsed = time.monotonic() - start
    console.print(
        Panel.fit(
            f"[bold green]Concluído em {elapsed:.1f}s[/bold green]\n\n{stats.summary()}",
            border_style="green",
        )
    )

    # Exit code diferente de zero se alguma coisa falhou — útil em CI/automação
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
