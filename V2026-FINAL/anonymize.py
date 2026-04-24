"""
anonymize.py
============
Entry point do pipeline de anonimização.

Uso (fora do Docker):
    python anonymize.py
    python anonymize.py --corte 2024-S2
    python anonymize.py --corte 202412

Uso (Docker):
    docker compose run --rm anonymize
    docker compose run --rm anonymize --corte 2024-S2

A pasta de saída é CSV/anonimizados/DDMMAAAA_HHMMSS/ (timestamp por execução),
contendo:
    - dataset_anonymized_AAAASN.csv   (sem PII, pronto para ML)
    - mapping_fatores.csv             (ponte auditoria: hash ↔ CNPJ/Razão)
    - resumo_execucao.txt             (metadados da rodada)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from rich.console import Console
from rich.panel import Panel

from config import settings as pipeline_settings
from src.anonymize import (
    AnonymizationSettings,
    Corte,
    run_anonymization,
)
from src.utils import setup_logging

console = Console()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        console.print(f"[red]Variável obrigatória '{name}' não configurada no .env[/red]")
        sys.exit(1)
    return value if value is not None else ""


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        console.print(f"[yellow]'{name}' inválido ('{raw}'), usando default {default}[/yellow]")
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        console.print(f"[yellow]'{name}' inválido ('{raw}'), usando default {default}[/yellow]")
        return default


def load_anonymization_settings() -> AnonymizationSettings:
    """Lê variáveis do .env específicas para anonimização."""
    return AnonymizationSettings(
        salt=_get_env("ANON_SALT", required=True),
        seed=_get_int("ANON_SEED", 42),
        fator_min=_get_float("ANON_FATOR_MIN", 0.60),
        fator_max=_get_float("ANON_FATOR_MAX", 1.40),
        ruido_min=_get_float("ANON_RUIDO_MIN", 0.98),
        ruido_max=_get_float("ANON_RUIDO_MAX", 1.02),
        indicadores_casas_decimais=_get_int("ANON_INDICADORES_CASAS_DECIMAIS", 3),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline de Anonimização e Prevenção de Data Leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corte",
        metavar="AAAA-SN",
        help="Limite temporal inclusivo (ex.: 2024-S2, 2024S2, 202412). "
             "Se omitido, o script perguntará interativamente.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log (default: INFO).",
    )
    return p


def perguntar_corte() -> Corte:
    """Pergunta interativamente até que o usuário digite um valor válido."""
    console.print(
        "\n[bold cyan]Defina o corte temporal inclusivo.[/bold cyan]\n"
        "Tudo até esse semestre (inclusive) entra no dataset anonimizado.\n"
        "Formatos aceitos: [yellow]AAAA-SN[/yellow] (ex.: 2024-S2) ou "
        "[yellow]AAAAMM[/yellow] (ex.: 202412)."
    )
    while True:
        raw = input("Corte: ").strip()
        if not raw:
            console.print("[red]Entrada vazia. Tente de novo.[/red]")
            continue
        try:
            return Corte.parse(raw)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")


def print_header(corte: Corte, anon_settings: AnonymizationSettings) -> None:
    salt_fp = anon_settings.salt[:4] + "…" + anon_settings.salt[-2:]  # nunca mostra SALT inteiro
    console.print(
        Panel.fit(
            "[bold cyan]TCC — Pipeline de Anonimização[/bold cyan]\n"
            "[dim]Duplo-cego: hashing + escalonamento + ruído + recálculo[/dim]\n\n"
            f"Corte temporal:          [yellow]{corte.label}[/yellow] (inclusivo)\n"
            f"Entrada:                 [yellow]{pipeline_settings.CSV_OUTPUT_DIR}[/yellow]\n"
            f"Saída:                   [yellow]{pipeline_settings.CSV_OUTPUT_DIR}/anonimizados/[TS][/yellow]\n"
            f"SEED:                    [yellow]{anon_settings.seed}[/yellow]\n"
            f"SALT (amostra):          [yellow]{salt_fp}[/yellow] (total {len(anon_settings.salt)} chars)\n"
            f"Fator escalonamento:     [yellow][{anon_settings.fator_min}, {anon_settings.fator_max}][/yellow]\n"
            f"Ruído contas resultado:  [yellow][{anon_settings.ruido_min}, {anon_settings.ruido_max}][/yellow]\n"
            f"Arred. indicadores:      [yellow]{anon_settings.indicadores_casas_decimais} casas[/yellow]",
            border_style="cyan",
        )
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(pipeline_settings.LOGS_DIR, level=log_level)

    # Corte: CLI > input interativo
    if args.corte:
        try:
            corte = Corte.parse(args.corte)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 2
    else:
        corte = perguntar_corte()

    # Parâmetros da anonimização (do .env)
    try:
        anon_settings = load_anonymization_settings()
        anon_settings.validate()
    except ValueError as e:
        console.print(f"[red]Erro na configuração:[/red] {e}")
        return 2

    print_header(corte, anon_settings)

    # Raiz de saída fixa: CSV/anonimizados/
    output_root = pipeline_settings.CSV_OUTPUT_DIR / "anonimizados"

    try:
        result = run_anonymization(
            csv_root=pipeline_settings.CSV_OUTPUT_DIR,
            output_root=output_root,
            corte=corte,
            settings=anon_settings,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        console.print(f"\n[red]Erro:[/red] {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrompido pelo usuário.[/yellow]")
        return 130

    console.print(
        Panel.fit(
            f"[bold green]✓ Anonimização concluída.[/bold green]\n\n"
            f"Dataset:  [cyan]{result['dataset']}[/cyan]\n"
            f"Mapping:  [cyan]{result['mapping']}[/cyan]\n"
            f"Resumo:   [cyan]{result['resumo']}[/cyan]",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
