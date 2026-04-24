"""
src/csv_builder.py
==================
Monta o CSV consolidado por instituição a partir dos JSONs extraídos.

O JSON do prompt tem 3 seções aninhadas (Identificacao, Contas_Base,
Indicadores). Aqui fazemos o "achatamento": cada chave vira uma
coluna, prefixada pela seção para evitar colisões e deixar a origem clara.

A ordem das colunas é FIXA (definida em COLUMN_ORDER) para que todos os CSVs
— de qualquer instituição — sejam 100% comparáveis.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import PdfMeta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Definição do schema do CSV
# ---------------------------------------------------------------------------
# Colunas de contexto (adicionadas por nós, não vêm do JSON)
_CONTEXT_COLUMNS = [
    "Semestre_Ref",     # Ex.: "2020-S1"
    "Ano",              # Ex.: 2020
    "Mes",              # Ex.: 6
    "YYYYMM",           # Ex.: "202006"
    "CNPJ8",            # Ex.: "13140088"
    "Arquivo_PDF",      # Ex.: "202006-9010-13140088.pdf"
]

# Colunas vindas do JSON, prefixadas pela seção de origem
_IDENT_COLUMNS = [
    "Ident__Razao_Social",
    "Ident__CNPJ",
    "Ident__Auditor_Independente",
    "Ident__Opiniao_Auditor",
    "Ident__Enfase_Auditor",
    "Ident__Eventos_Subsequentes_Adversos",
]

_CONTAS_COLUMNS = [
    "Contas__Ativo_Total",
    "Contas__Patrimonio_Liquido",
    "Contas__Lucro_Liquido",
    "Contas__Caixa_Equivalentes",
    "Contas__Obrigacoes_Liquidacao",
    "Contas__Receita_Servicos",
    "Contas__Despesa_Processamento",
    "Contas__Resultado_Financeiro",
    "Contas__Despesa_Operacional_Total",
    "Contas__Provisao_Perdas",
    "Contas__Fluxo_Caixa_Operacional",
    "Contas__Ativo_Intangivel",
    "Contas__Ativo_Partes_Relacionadas",
    "Contas__Passivo_Partes_Relacionadas",
    "Contas__Provisao_Contingencias",
    "Contas__Ativo_Fiscal_Diferido",
]

_INDICADORES_COLUMNS = [
    "Indic__Liquidez_Imediata",
    "Indic__Indice_Capitalizacao",
    "Indic__ROE",
    "Indic__Margem_Servicos",
    "Indic__Indice_Eficiencia",
]

# Ordem final e canônica das colunas do CSV.
# Qualquer mudança no prompt deve ser refletida AQUI também.
COLUMN_ORDER: list[str] = (
    _CONTEXT_COLUMNS + _IDENT_COLUMNS + _CONTAS_COLUMNS + _INDICADORES_COLUMNS
)


# Mapeamento JSON → coluna. Usado para garantir que NENHUM campo do JSON
# seja esquecido ao virar coluna (caso contrário, teríamos perda silenciosa
# de informação, que é justamente o que o usuário quer evitar).
_JSON_TO_COLUMN = {
    ("Identificacao", "Razao_Social"): "Ident__Razao_Social",
    ("Identificacao", "CNPJ"): "Ident__CNPJ",
    ("Identificacao", "Auditor_Independente"): "Ident__Auditor_Independente",
    ("Identificacao", "Opiniao_Auditor"): "Ident__Opiniao_Auditor",
    ("Identificacao", "Enfase_Auditor"): "Ident__Enfase_Auditor",
    ("Identificacao", "Eventos_Subsequentes_Adversos"): "Ident__Eventos_Subsequentes_Adversos",

    ("Contas_Base", "Ativo_Total"): "Contas__Ativo_Total",
    ("Contas_Base", "Patrimonio_Liquido"): "Contas__Patrimonio_Liquido",
    ("Contas_Base", "Lucro_Liquido"): "Contas__Lucro_Liquido",
    ("Contas_Base", "Caixa_Equivalentes"): "Contas__Caixa_Equivalentes",
    ("Contas_Base", "Obrigacoes_Liquidacao"): "Contas__Obrigacoes_Liquidacao",
    ("Contas_Base", "Receita_Servicos"): "Contas__Receita_Servicos",
    ("Contas_Base", "Despesa_Processamento"): "Contas__Despesa_Processamento",
    ("Contas_Base", "Resultado_Financeiro"): "Contas__Resultado_Financeiro",
    ("Contas_Base", "Despesa_Operacional_Total"): "Contas__Despesa_Operacional_Total",
    ("Contas_Base", "Provisao_Perdas"): "Contas__Provisao_Perdas",
    ("Contas_Base", "Fluxo_Caixa_Operacional"): "Contas__Fluxo_Caixa_Operacional",
    ("Contas_Base", "Ativo_Intangivel"): "Contas__Ativo_Intangivel",
    ("Contas_Base", "Ativo_Partes_Relacionadas"): "Contas__Ativo_Partes_Relacionadas",
    ("Contas_Base", "Passivo_Partes_Relacionadas"): "Contas__Passivo_Partes_Relacionadas",
    ("Contas_Base", "Provisao_Contingencias"): "Contas__Provisao_Contingencias",
    ("Contas_Base", "Ativo_Fiscal_Diferido"): "Contas__Ativo_Fiscal_Diferido",

    ("Indicadores", "Liquidez_Imediata"): "Indic__Liquidez_Imediata",
    ("Indicadores", "Indice_Capitalizacao"): "Indic__Indice_Capitalizacao",
    ("Indicadores", "ROE"): "Indic__ROE",
    ("Indicadores", "Margem_Servicos"): "Indic__Margem_Servicos",
    ("Indicadores", "Indice_Eficiencia"): "Indic__Indice_Eficiencia",
}


@dataclass
class SemestreRecord:
    """Representa uma linha do CSV: um semestre da instituição."""
    meta: PdfMeta
    extraction: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        """Converte o registro em uma dict já no formato de coluna do CSV."""
        row: dict[str, Any] = {
            "Semestre_Ref": self.meta.semestre_label,
            "Ano": self.meta.ano,
            "Mes": self.meta.mes,
            "YYYYMM": self.meta.yyyymm,
            "CNPJ8": self.meta.cnpj8,
            "Arquivo_PDF": self.meta.filename,
        }

        # Normalização defensiva: o Gemini, ocasionalmente, embrulha a resposta
        # em uma lista de 1 elemento (`[{...}]`) mesmo o prompt pedindo objeto.
        # Também cobre JSONs antigos já salvos em disco nesse formato "bichado".
        extraction = self.extraction
        if isinstance(extraction, list):
            if len(extraction) == 1 and isinstance(extraction[0], dict):
                log.warning(
                    "JSON de %s veio embrulhado em lista — desembrulhando automaticamente.",
                    self.meta.filename,
                )
                extraction = extraction[0]
            else:
                log.error(
                    "JSON de %s tem estrutura inesperada (lista com %d elementos). "
                    "Linha será preenchida apenas com o contexto.",
                    self.meta.filename, len(extraction),
                )
                for _key, col in _JSON_TO_COLUMN.items():
                    row[col] = None
                return row

        if not isinstance(extraction, dict):
            log.error(
                "JSON de %s não é objeto nem lista (tipo=%s). Linha será preenchida "
                "apenas com o contexto.",
                self.meta.filename, type(extraction).__name__,
            )
            for _key, col in _JSON_TO_COLUMN.items():
                row[col] = None
            return row

        # Percorre o JSON esperado e preenche as colunas. Se algum campo faltar,
        # registra warning mas continua (célula vazia não quebra o pipeline).
        for (section, field), col in _JSON_TO_COLUMN.items():
            section_dict = extraction.get(section, {}) or {}
            if field not in section_dict:
                log.warning(
                    "Campo ausente no JSON: %s.%s (arquivo=%s)",
                    section, field, self.meta.filename,
                )
                row[col] = None
            else:
                row[col] = section_dict[field]

        # Sanidade: avisa se o JSON trouxe seções/campos não mapeados
        # (indica que o prompt evoluiu e COLUMN_ORDER precisa ser atualizado).
        _warn_unmapped_fields(extraction, self.meta.filename)

        return row


def _warn_unmapped_fields(extraction: dict[str, Any], source: str) -> None:
    """Alerta se o JSON retornou campos além dos esperados (ajuda a detectar drift)."""
    known_fields = {(s, f) for (s, f) in _JSON_TO_COLUMN.keys()}
    for section, section_content in extraction.items():
        if not isinstance(section_content, dict):
            continue
        for field in section_content.keys():
            if (section, field) not in known_fields:
                log.warning(
                    "Campo novo no JSON não mapeado para CSV: %s.%s (arquivo=%s). "
                    "Considere atualizar csv_builder.COLUMN_ORDER.",
                    section, field, source,
                )


def build_csv_for_institution(
    cnpj8: str,
    records: list[SemestreRecord],
    output_dir: Path,
) -> Path:
    """
    Grava o CSV consolidado de uma instituição.

    Args:
        cnpj8: 8 primeiros dígitos do CNPJ (nome da pasta e do arquivo).
        records: lista de semestres já extraídos (serão ordenados cronologicamente).
        output_dir: raiz do diretório CSV (o arquivo final fica em
                    output_dir/cnpj8/cnpj8.csv).

    Retorna o caminho do CSV gerado.
    """
    if not records:
        raise ValueError(f"Nenhum registro para gerar CSV do CNPJ {cnpj8}")

    # Ordenação cronológica: garantimos ordem ano → mês
    ordered = sorted(records, key=lambda r: r.meta.sort_key)

    rows = [r.to_row() for r in ordered]
    df = pd.DataFrame(rows, columns=COLUMN_ORDER)

    institution_dir = output_dir / cnpj8
    institution_dir.mkdir(parents=True, exist_ok=True)
    csv_path = institution_dir / f"{cnpj8}.csv"

    # Separador ";" é amigável para Excel em português (vírgula decimal).
    # utf-8-sig mantém compatibilidade com Excel no Windows/Mac.
    df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")

    log.info("CSV gerado: %s (%d linhas)", csv_path, len(df))
    return csv_path


def save_raw_json(
    meta: PdfMeta,
    extraction: dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Grava o JSON bruto de um arquivo na pasta da instituição.

    Ex.: CSV/13140088/202006-9010-13140088.json
    """
    institution_dir = output_dir / meta.cnpj8
    institution_dir.mkdir(parents=True, exist_ok=True)

    # Nome do JSON espelha o nome do PDF (trocando a extensão)
    json_name = Path(meta.filename).with_suffix(".json").name
    json_path = institution_dir / json_name

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(extraction, f, ensure_ascii=False, indent=2)

    log.debug("JSON bruto salvo: %s", json_path)
    return json_path


def load_raw_json(meta: PdfMeta, output_dir: Path) -> dict[str, Any] | None:
    """Tenta carregar um JSON já processado. Útil para retomar execuções sem --force."""
    json_name = Path(meta.filename).with_suffix(".json").name
    json_path = output_dir / meta.cnpj8 / json_name
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("JSON existente inválido (%s): %s. Será reprocessado.", json_path, e)
        return None
