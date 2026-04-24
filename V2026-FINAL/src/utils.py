"""
src/utils.py
============
Utilitários: logging estruturado e parsing dos nomes de arquivo PDF.

Padrão esperado de nome: `AAAAMM-9010-CNPJ8.pdf`
    AAAAMM : ano e mês (6 dígitos, ex.: 202006, 201912)
    9010   : código fixo do CADOC (documento regulatório do BCB)
    CNPJ8  : 8 primeiros dígitos do CNPJ da instituição
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rich.logging import RichHandler

# Regex para o padrão AAAAMM-9010-CNPJ8.pdf
# Permite variações no código fixo (9010) por segurança, mas valida os demais.
_PDF_NAME_RE = re.compile(r"^(?P<yyyymm>\d{6})-(?P<code>\d{4})-(?P<cnpj8>\d{8})\.pdf$", re.IGNORECASE)


@dataclass(frozen=True)
class PdfMeta:
    """Metadados extraídos a partir do nome do arquivo PDF."""
    yyyymm: str          # Ex.: "202006"
    ano: int             # Ex.: 2020
    mes: int             # Ex.: 6
    semestre_label: str  # Ex.: "2020-S1" ou "2020-S2"
    cnpj8: str           # Ex.: "13140088"
    codigo: str          # Ex.: "9010"
    filename: str        # Nome original do arquivo

    @property
    def sort_key(self) -> tuple[int, int]:
        """Chave para ordenação cronológica (ano, mês)."""
        return (self.ano, self.mes)


def parse_pdf_filename(path: Path) -> PdfMeta | None:
    """
    Tenta extrair metadados do nome do arquivo. Retorna None se não casar.

    Exemplos:
        >>> parse_pdf_filename(Path("202006-9010-13140088.pdf")).cnpj8
        '13140088'
        >>> parse_pdf_filename(Path("lixo.pdf")) is None
        True
    """
    m = _PDF_NAME_RE.match(path.name)
    if not m:
        return None

    yyyymm = m.group("yyyymm")
    ano = int(yyyymm[:4])
    mes = int(yyyymm[4:])

    # Semestre: meses 1–6 → S1, 7–12 → S2.
    # Como na prática o BCB divulga com referência a junho (S1) e dezembro (S2),
    # esse mapeamento cobre os casos reais do projeto.
    semestre = "S1" if mes <= 6 else "S2"

    return PdfMeta(
        yyyymm=yyyymm,
        ano=ano,
        mes=mes,
        semestre_label=f"{ano}-{semestre}",
        cnpj8=m.group("cnpj8"),
        codigo=m.group("code"),
        filename=path.name,
    )


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configura logging com dois destinos:
        - Terminal: colorido e formatado (via rich).
        - Arquivo: em logs_dir/run.log, para auditoria posterior.

    Retorna o logger raiz já configurado.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "run.log"

    # Formatação comum para arquivo (terminal usa o formato do RichHandler)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(level)

    # RichHandler já cuida do formato bonito no terminal (emoji de nível, cores, etc.)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=False,
    )
    rich_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove handlers antigos para evitar duplicação se setup_logging for chamado mais de uma vez
    root.handlers.clear()
    root.addHandler(rich_handler)
    root.addHandler(file_handler)

    # Silencia libs verbosas
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
