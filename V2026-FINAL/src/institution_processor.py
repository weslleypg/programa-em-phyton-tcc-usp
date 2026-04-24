"""
src/institution_processor.py
============================
Descobre todos os PDFs, agrupa por instituição (CNPJ8), processa cada PDF
via Gemini, salva o JSON bruto e, ao final, consolida o CSV da instituição.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .csv_builder import (
    SemestreRecord,
    build_csv_for_institution,
    load_raw_json,
    save_raw_json,
)
from .gemini_client import GeminiClient, GeminiExtractionError
from .utils import PdfMeta, parse_pdf_filename

log = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Estatísticas finais da execução."""
    total_pdfs: int = 0
    processed: int = 0
    skipped_cached: int = 0
    failed: int = 0
    institutions_ok: int = 0
    institutions_with_failures: int = 0

    def summary(self) -> str:
        return (
            f"Total PDFs: {self.total_pdfs} | "
            f"Processados: {self.processed} | "
            f"Cache: {self.skipped_cached} | "
            f"Falhas: {self.failed} | "
            f"Instituições OK: {self.institutions_ok} | "
            f"Instituições c/ falhas: {self.institutions_with_failures}"
        )


def discover_pdfs_grouped_by_cnpj(pdf_root: Path) -> dict[str, list[tuple[Path, PdfMeta]]]:
    """
    Varre recursivamente pdf_root procurando arquivos no padrão esperado e
    agrupa por CNPJ8.

    Retorna: { "13140088": [(caminho_pdf, PdfMeta), ...], ... }
    """
    if not pdf_root.is_dir():
        raise FileNotFoundError(f"Diretório de PDFs não existe: {pdf_root}")

    groups: dict[str, list[tuple[Path, PdfMeta]]] = defaultdict(list)
    total_found = 0
    total_ignored = 0

    for pdf_path in pdf_root.rglob("*.pdf"):
        meta = parse_pdf_filename(pdf_path)
        if meta is None:
            log.warning("Ignorando arquivo com nome fora do padrão: %s", pdf_path)
            total_ignored += 1
            continue
        groups[meta.cnpj8].append((pdf_path, meta))
        total_found += 1

    log.info(
        "Descoberta: %d PDFs válidos em %d instituições (%d ignorados).",
        total_found, len(groups), total_ignored,
    )
    return groups


def _process_single_pdf(
    pdf_path: Path,
    meta: PdfMeta,
    client: GeminiClient,
    csv_output_dir: Path,
    force: bool,
    delay_between_calls: int,
) -> SemestreRecord | None:
    """
    Processa um único PDF. Retorna SemestreRecord ou None em caso de falha.

    Se force=False e já existir JSON válido em disco, reutiliza (idempotência).
    """
    # --- 1) Cache hit? ---
    if not force:
        cached = load_raw_json(meta, csv_output_dir)
        if cached is not None:
            log.info("[CACHE] %s → já processado, reutilizando JSON existente.", meta.filename)
            return SemestreRecord(meta=meta, extraction=cached)

    # --- 2) Chamar Gemini ---
    try:
        extraction = client.extract_from_pdf(pdf_path)
    except GeminiExtractionError as e:
        log.error("[FALHA] %s → %s", meta.filename, e)
        return None
    except Exception as e:  # noqa: BLE001 — queremos isolar qualquer crash
        log.exception("[FALHA INESPERADA] %s → %s", meta.filename, e)
        return None

    # --- 3) Salvar JSON bruto ---
    save_raw_json(meta, extraction, csv_output_dir)

    # --- 4) Delay gentil entre chamadas (ajuda com rate limit) ---
    if delay_between_calls > 0:
        time.sleep(delay_between_calls)

    return SemestreRecord(meta=meta, extraction=extraction)


def process_institution(
    cnpj8: str,
    pdfs: list[tuple[Path, PdfMeta]],
    client: GeminiClient,
    csv_output_dir: Path,
    force: bool,
    delay_between_calls: int,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> tuple[int, int]:
    """
    Processa todos os PDFs de uma instituição e gera o CSV consolidado.

    Retorna (sucessos, falhas) dessa instituição.
    """
    log.info("==> Instituição %s (%d arquivo(s))", cnpj8, len(pdfs))

    # Ordena por ano/mês antes de processar — facilita a leitura dos logs.
    pdfs_sorted = sorted(pdfs, key=lambda item: item[1].sort_key)

    records: list[SemestreRecord] = []
    failures = 0

    for pdf_path, meta in pdfs_sorted:
        record = _process_single_pdf(
            pdf_path=pdf_path,
            meta=meta,
            client=client,
            csv_output_dir=csv_output_dir,
            force=force,
            delay_between_calls=delay_between_calls,
        )
        if record is None:
            failures += 1
        else:
            records.append(record)

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    # Gera CSV mesmo se houver falhas parciais — pelo menos o que deu certo fica salvo.
    if records:
        try:
            build_csv_for_institution(cnpj8, records, csv_output_dir)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao gerar CSV da instituição %s", cnpj8)
            failures += 1
    else:
        log.error("Instituição %s: nenhum registro válido — CSV não será gerado.", cnpj8)

    return (len(records), failures)


def run_pipeline(
    pdf_root: Path,
    csv_output_dir: Path,
    client: GeminiClient,
    force: bool,
    delay_between_calls: int,
    only_cnpj8: str | None = None,
) -> ProcessingStats:
    """
    Ponto de entrada do pipeline. Descobre PDFs, processa por instituição e
    retorna estatísticas consolidadas.
    """
    groups = discover_pdfs_grouped_by_cnpj(pdf_root)

    if only_cnpj8:
        if only_cnpj8 not in groups:
            log.error("Nenhum PDF encontrado para CNPJ8=%s", only_cnpj8)
            return ProcessingStats()
        groups = {only_cnpj8: groups[only_cnpj8]}
        log.info("Filtro ativo: processando apenas CNPJ8=%s", only_cnpj8)

    stats = ProcessingStats()
    stats.total_pdfs = sum(len(v) for v in groups.values())

    # Barra de progresso global cobrindo TODOS os PDFs — feedback visual no terminal.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processando PDFs",
            total=stats.total_pdfs,
        )

        for cnpj8, pdfs in sorted(groups.items()):
            progress.update(task, description=f"[cyan]Instituição {cnpj8}")
            ok, fail = process_institution(
                cnpj8=cnpj8,
                pdfs=pdfs,
                client=client,
                csv_output_dir=csv_output_dir,
                force=force,
                delay_between_calls=delay_between_calls,
                progress=progress,
                task_id=task,
            )
            stats.processed += ok
            stats.failed += fail
            if fail == 0:
                stats.institutions_ok += 1
            else:
                stats.institutions_with_failures += 1

    # Obs.: stats.skipped_cached é informativo — o código atual não distingue
    # cache de reprocessado no contador (os dois incrementam `processed`).
    # Se quiser métrica exata de cache, dá para propagar do _process_single_pdf.
    return stats
