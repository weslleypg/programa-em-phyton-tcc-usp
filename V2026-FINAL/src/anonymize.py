"""
src/anonymize.py
================
Pipeline ETL de Anonimização e Prevenção de Data Leakage (Duplo-Cego).

Fluxo (Extract → Transform → Load):
    1. LOAD         — lê todos os CSVs consolidados de CSV/<cnpj8>/<cnpj8>.csv.
    2. CORTE        — filtra linhas até o semestre-limite inclusivo.
    3. HASHING      — gera inst_hash SHA-256 a partir do CNPJ8 + SALT.
    4. CATEGORIZAÇÃO— normaliza Auditor_Independente para "Big 4" ou "Outros".
    5. ESCALONAMENTO— multiplicador fixo por instituição em todas as contas absolutas.
    6. RUÍDO        — ruído simétrico independente nas contas de resultado.
    7. RECÁLCULO    — recalcula indicadores e arredonda.
    8. AUDITORIA    — calcula variação %, gera mapping_fatores.csv.
    9. EXPORT       — salva dataset_anonymized_AAAASN.csv + mapping + resumo.

Reprodutibilidade:
    Dado o mesmo SALT, SEED, FATOR_MIN/MAX, RUIDO_MIN/MAX e os mesmos CSVs de
    entrada, as execuções produzem datasets IDÊNTICOS. Qualquer mudança em
    uma dessas variáveis altera completamente os fatores sorteados — útil
    para gerar múltiplos datasets "diferentes" sem mudar o código.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema: que colunas sofrem cada transformação
# ---------------------------------------------------------------------------

# Contas absolutas (multiplicadas pelo fator de escalonamento da instituição).
_COLUNAS_ESCALONAMENTO: list[str] = [
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

# Contas de resultado (recebem ruído simétrico adicional após o escalonamento).
# Não aplicamos ruído nas grandes rubricas de balanço (Ativo, PL, Caixa, etc.)
# para não quebrar a equação contábil Ativo ≈ Passivo + PL.
_COLUNAS_RUIDO: list[str] = [
    "Contas__Lucro_Liquido",
    "Contas__Receita_Servicos",
    "Contas__Despesa_Processamento",
    "Contas__Resultado_Financeiro",
    "Contas__Despesa_Operacional_Total",
    "Contas__Provisao_Perdas",
]

# Colunas PII/identificação removidas do dataset final (ficam apenas no mapping).
_COLUNAS_PII_REMOVIDAS: list[str] = [
    "Ident__CNPJ",
    "CNPJ8",
    "Ident__Razao_Social",
    "Arquivo_PDF",
]

# Regex de categorização de auditores. A ordem importa apenas para legibilidade.
# O `\b` é word boundary; aceita variações como "KPMG Brasil", "Ernst & Young LLP".
# Tornamos case-insensitive (re.IGNORECASE) para casar "PwC", "PWC", "pwc" etc.
_BIG4_PATTERN = re.compile(
    r"\b(PwC|PricewaterhouseCoopers|KPMG|Ernst\s*&\s*Young|EY|Deloitte|DTT)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Estruturas auxiliares
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationSettings:
    """Parâmetros do pipeline, lidos do .env. Imutável após criação."""
    salt: str
    seed: int
    fator_min: float
    fator_max: float
    ruido_min: float
    ruido_max: float
    indicadores_casas_decimais: int

    def validate(self) -> None:
        """Garante coerência dos parâmetros antes de rodar o pipeline."""
        if not self.salt or len(self.salt) < 16:
            raise ValueError(
                "ANON_SALT deve ter pelo menos 16 caracteres no .env. "
                "Um SALT curto torna o hash trivial de reverter."
            )
        if self.fator_min >= self.fator_max:
            raise ValueError(f"FATOR_MIN ({self.fator_min}) deve ser < FATOR_MAX ({self.fator_max}).")
        if self.ruido_min >= self.ruido_max:
            raise ValueError(f"RUIDO_MIN ({self.ruido_min}) deve ser < RUIDO_MAX ({self.ruido_max}).")
        if self.fator_min <= 0:
            raise ValueError("FATOR_MIN deve ser > 0 (negativo/zero zeraria ou inverteria sinais).")
        if self.ruido_min <= 0:
            raise ValueError("RUIDO_MIN deve ser > 0.")
        if self.indicadores_casas_decimais < 0 or self.indicadores_casas_decimais > 10:
            raise ValueError("INDICADORES_CASAS_DECIMAIS deve estar entre 0 e 10.")


@dataclass
class Corte:
    """Representa o limite temporal inclusivo (até AAAA-SN)."""
    ano: int
    semestre: int  # 1 ou 2

    @property
    def label(self) -> str:
        return f"{self.ano}-S{self.semestre}"

    @property
    def slug(self) -> str:
        """Formato compacto para nome de arquivo, ex: '2024S1'."""
        return f"{self.ano}S{self.semestre}"

    @property
    def ordinal(self) -> int:
        """Chave comparável: (ano * 10 + semestre). Ex.: 2024-S2 → 20242."""
        return self.ano * 10 + self.semestre

    @classmethod
    def parse(cls, raw: str) -> "Corte":
        """
        Aceita formatos "AAAA-SN", "AAAA-sN", "AAAASN" e "AAAAMM" (meses 06/12).
        """
        raw_clean = raw.strip().upper().replace("-", "")
        m_ss = re.fullmatch(r"(\d{4})S([12])", raw_clean)
        m_mm = re.fullmatch(r"(\d{4})(0[16]|12|06)", raw_clean)
        if m_ss:
            return cls(ano=int(m_ss.group(1)), semestre=int(m_ss.group(2)))
        if m_mm:
            ano = int(m_mm.group(1))
            mes = int(m_mm.group(2))
            # 06 → S1, 12 → S2
            return cls(ano=ano, semestre=1 if mes == 6 else 2)
        raise ValueError(
            f"Corte '{raw}' inválido. Use o formato AAAA-SN (ex.: 2024-S1) "
            f"ou AAAAMM (ex.: 202406, 202412)."
        )


@dataclass
class PipelineStats:
    """Métricas coletadas durante a execução, usadas no resumo."""
    instituicoes_encontradas: int = 0
    instituicoes_incluidas: int = 0
    instituicoes_sem_dados_no_corte: list[str] = field(default_factory=list)
    linhas_totais_antes_corte: int = 0
    linhas_totais_apos_corte: int = 0
    linhas_dataset_final: int = 0
    csvs_entrada_malformados: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Funções auxiliares (puras, testáveis)
# ---------------------------------------------------------------------------

def hash_cnpj8(cnpj8: str, salt: str) -> str:
    """
    Gera hash SHA-256 determinístico do CNPJ8 com o SALT.
    O CNPJ8 é zero-padded para 8 dígitos por consistência (caso venha como int).
    """
    normalized = str(cnpj8).strip().zfill(8)
    raw = f"{salt}::{normalized}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def categorizar_auditor(nome: Any) -> str:
    """Classifica auditor como 'Big 4' ou 'Outros'. Tolera nulos."""
    if not isinstance(nome, str) or not nome.strip():
        return "Outros"
    return "Big 4" if _BIG4_PATTERN.search(nome) else "Outros"


def _safe_divide(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Divisão que retorna 0 onde o denominador é 0/NaN, seguindo o enunciado."""
    num = pd.to_numeric(numerador, errors="coerce")
    den = pd.to_numeric(denominador, errors="coerce")
    # np.where garante vetorização; convertemos para float para evitar
    # warnings do pandas em FutureWarning sobre downcast.
    result = np.where((den == 0) | den.isna(), 0.0, num / den.replace(0, np.nan))
    return pd.Series(result, index=numerador.index, dtype="float64").fillna(0.0)


def _var_pct(novo: pd.Series, original: pd.Series) -> pd.Series:
    """
    Variação percentual: ((novo / original) - 1) * 100.
    Retorna 0 quando original é 0/NaN (não há base de comparação).
    """
    novo_f = pd.to_numeric(novo, errors="coerce")
    orig_f = pd.to_numeric(original, errors="coerce")
    ratio = np.where((orig_f == 0) | orig_f.isna(), 1.0, novo_f / orig_f.replace(0, np.nan))
    return pd.Series((ratio - 1.0) * 100.0, index=novo.index, dtype="float64").fillna(0.0)


# ---------------------------------------------------------------------------
# Descoberta e carga
# ---------------------------------------------------------------------------

def descobrir_csvs(csv_root: Path) -> list[Path]:
    """
    Procura arquivos no padrão CSV/<cnpj8>/<cnpj8>.csv. Ignora a pasta
    CSV/anonimizados/ (para que múltiplas execuções não comam as próprias saídas).
    """
    if not csv_root.is_dir():
        raise FileNotFoundError(f"Diretório raiz de CSVs não existe: {csv_root}")

    encontrados: list[Path] = []
    for subdir in sorted(csv_root.iterdir()):
        if not subdir.is_dir():
            continue
        # Pula pastas reservadas
        if subdir.name in {"anonimizados", "__pycache__"}:
            continue
        # A pasta deve se chamar como o CNPJ8 (8 dígitos). Defensivo:
        if not re.fullmatch(r"\d{8}", subdir.name):
            log.debug("Ignorando pasta fora do padrão: %s", subdir.name)
            continue
        csv_path = subdir / f"{subdir.name}.csv"
        if csv_path.is_file():
            encontrados.append(csv_path)
        else:
            log.warning("Pasta %s existe mas não tem %s.csv consolidado.",
                        subdir.name, subdir.name)
    return encontrados


def carregar_csv_instituicao(csv_path: Path) -> pd.DataFrame | None:
    """Lê um CSV de instituição. Retorna None se falhar ou vier vazio."""
    try:
        # O pipeline de extração gera CSVs com sep=';' e utf-8-sig.
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype={"CNPJ8": str, "YYYYMM": str})
    except Exception as e:
        log.error("Falha ao ler %s: %s", csv_path, e)
        return None
    if df.empty:
        log.warning("CSV vazio: %s", csv_path)
        return None
    return df


# ---------------------------------------------------------------------------
# Transformações (cada uma é testável isoladamente)
# ---------------------------------------------------------------------------

def aplicar_corte(df: pd.DataFrame, corte: Corte) -> pd.DataFrame:
    """
    Filtra linhas com (Ano, Mes convertido para semestre) <= corte.
    Assume que o CSV tem colunas 'Ano' e 'Mes'.
    """
    if "Ano" not in df.columns or "Mes" not in df.columns:
        raise ValueError("CSV sem colunas 'Ano'/'Mes' — cannot aplicar corte.")
    ano = pd.to_numeric(df["Ano"], errors="coerce").fillna(0).astype(int)
    mes = pd.to_numeric(df["Mes"], errors="coerce").fillna(0).astype(int)
    # Semestre derivado: mes<=6 → 1, mes>6 → 2
    semestre = np.where(mes <= 6, 1, 2)
    ordinal = ano * 10 + semestre
    return df.loc[ordinal <= corte.ordinal].copy()


def aplicar_hashing(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Cria coluna `inst_hash` a partir de CNPJ8 (sempre, conforme decisão do usuário)."""
    if "CNPJ8" not in df.columns:
        raise ValueError("CSV sem coluna CNPJ8 — impossível hashear.")
    df = df.copy()
    # Normalização consistente do CNPJ8 para string de 8 dígitos
    cnpj_str = df["CNPJ8"].astype(str).str.strip().str.zfill(8)
    df["inst_hash"] = cnpj_str.apply(lambda c: hash_cnpj8(c, salt))
    return df


def aplicar_categorizacao_auditor(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza Ident__Auditor_Independente para 'Big 4' / 'Outros'."""
    if "Ident__Auditor_Independente" not in df.columns:
        log.warning("Coluna Ident__Auditor_Independente ausente — pulando categorização.")
        return df
    df = df.copy()
    df["Ident__Auditor_Independente"] = df["Ident__Auditor_Independente"].apply(categorizar_auditor)
    return df


def gerar_fatores_por_instituicao(
    inst_hashes: pd.Series,
    settings: AnonymizationSettings,
) -> dict[str, float]:
    """
    Para cada inst_hash único, sorteia um fator fixo em [fator_min, fator_max].
    Determinístico dado SEED+SALT: ordena hashes para garantir reprodutibilidade.
    """
    unique_hashes = sorted(inst_hashes.drop_duplicates().tolist())
    rng = np.random.default_rng(settings.seed)
    fatores = rng.uniform(settings.fator_min, settings.fator_max, size=len(unique_hashes))
    return dict(zip(unique_hashes, fatores))


def aplicar_escalonamento(
    df: pd.DataFrame,
    fatores: dict[str, float],
) -> pd.DataFrame:
    """Multiplica colunas absolutas pelo fator da instituição (broadcast por linha)."""
    df = df.copy()
    serie_fator = df["inst_hash"].map(fatores).astype("float64")
    # Fator deve existir para toda linha. Se algum estiver NaN, é bug grave.
    if serie_fator.isna().any():
        hashes_sem_fator = df.loc[serie_fator.isna(), "inst_hash"].unique().tolist()
        raise RuntimeError(f"Fatores ausentes para hashes: {hashes_sem_fator}")

    colunas_presentes = [c for c in _COLUNAS_ESCALONAMENTO if c in df.columns]
    faltantes = set(_COLUNAS_ESCALONAMENTO) - set(colunas_presentes)
    if faltantes:
        log.warning("Colunas ausentes no DF (não serão escalonadas): %s", sorted(faltantes))

    for col in colunas_presentes:
        # pd.to_numeric é defensivo: alguma célula pode vir como string "0"
        valores = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col] = valores * serie_fator

    # Grava o fator na linha (será removido do dataset final, permanece no mapping)
    df["_Fator_Escalonamento"] = serie_fator
    return df


def aplicar_ruido(df: pd.DataFrame, settings: AnonymizationSettings) -> pd.DataFrame:
    """
    Aplica ruído multiplicativo ~U(ruido_min, ruido_max) em contas de resultado.
    Cada célula recebe um sorteio INDEPENDENTE (diferente de cada instituição ter
    o mesmo fator de escalonamento para todas as linhas).
    """
    df = df.copy()
    # RNG separada da do escalonamento para não interferir naquela sequência.
    rng = np.random.default_rng(settings.seed + 1)
    colunas_presentes = [c for c in _COLUNAS_RUIDO if c in df.columns]
    faltantes = set(_COLUNAS_RUIDO) - set(colunas_presentes)
    if faltantes:
        log.warning("Colunas de resultado ausentes (sem ruído): %s", sorted(faltantes))

    for col in colunas_presentes:
        ruido = rng.uniform(settings.ruido_min, settings.ruido_max, size=len(df))
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * ruido
    return df


def preservar_indicadores_originais(df: pd.DataFrame) -> pd.DataFrame:
    """Guarda os indicadores originais em colunas temporárias _Orig_X."""
    df = df.copy()
    for col in ("Indic__Liquidez_Imediata", "Indic__Indice_Capitalizacao",
                "Indic__ROE", "Indic__Indice_Eficiencia", "Indic__Margem_Servicos"):
        if col in df.columns:
            df[f"_Orig_{col}"] = pd.to_numeric(df[col], errors="coerce")
    return df


def recalcular_indicadores(df: pd.DataFrame, casas: int) -> pd.DataFrame:
    """Recalcula os 5 indicadores Assaf Neto a partir das contas anonimizadas."""
    df = df.copy()

    df["Indic__Liquidez_Imediata"] = _safe_divide(
        df["Contas__Caixa_Equivalentes"], df["Contas__Obrigacoes_Liquidacao"]
    ).round(casas)

    df["Indic__Indice_Capitalizacao"] = _safe_divide(
        df["Contas__Patrimonio_Liquido"], df["Contas__Ativo_Total"]
    ).round(casas)

    df["Indic__ROE"] = _safe_divide(
        df["Contas__Lucro_Liquido"], df["Contas__Patrimonio_Liquido"]
    ).round(casas)

    # Margem_Servicos é valor absoluto em R$ (subtração pura, sem divisão).
    # O enunciado pede apenas a subtração; não arredonda percentualmente.
    receita = pd.to_numeric(df["Contas__Receita_Servicos"], errors="coerce").fillna(0.0)
    despesa = pd.to_numeric(df["Contas__Despesa_Processamento"], errors="coerce").fillna(0.0)
    df["Indic__Margem_Servicos"] = (receita - despesa).round(2)

    df["Indic__Indice_Eficiencia"] = _safe_divide(
        df["Contas__Despesa_Operacional_Total"],
        pd.to_numeric(df["Contas__Receita_Servicos"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Contas__Resultado_Financeiro"], errors="coerce").fillna(0.0),
    ).round(casas)

    return df


def montar_mapping_auditoria(
    df: pd.DataFrame,
    fatores: dict[str, float],
) -> pd.DataFrame:
    """
    Constrói o DataFrame de mapping/auditoria.
    Colunas: inst_hash, Semestre_Ref, Ident__CNPJ, Ident__Razao_Social,
             Fator_Escalonamento, Var_%_ROE, Var_%_Eficiencia,
             Var_%_Liquidez, Var_%_Capitalizacao.
    """
    mapping = pd.DataFrame({
        "inst_hash": df["inst_hash"],
        "Semestre_Ref": df.get("Semestre_Ref"),
        "Ident__CNPJ": df.get("Ident__CNPJ"),
        "Ident__Razao_Social": df.get("Ident__Razao_Social"),
        "Fator_Escalonamento": df["inst_hash"].map(fatores),
        "Var_%_ROE": _var_pct(df["Indic__ROE"], df["_Orig_Indic__ROE"]),
        "Var_%_Eficiencia": _var_pct(df["Indic__Indice_Eficiencia"], df["_Orig_Indic__Indice_Eficiencia"]),
        "Var_%_Liquidez": _var_pct(df["Indic__Liquidez_Imediata"], df["_Orig_Indic__Liquidez_Imediata"]),
        "Var_%_Capitalizacao": _var_pct(df["Indic__Indice_Capitalizacao"], df["_Orig_Indic__Indice_Capitalizacao"]),
    })
    # Arredondar variações para 4 casas (auditoria precisa de precisão)
    for c in ("Var_%_ROE", "Var_%_Eficiencia", "Var_%_Liquidez", "Var_%_Capitalizacao"):
        mapping[c] = mapping[c].round(4)
    mapping["Fator_Escalonamento"] = mapping["Fator_Escalonamento"].round(6)
    return mapping


def finalizar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove PII e colunas auxiliares (_Orig_*, _Fator_*). Ordena por instituição e tempo."""
    df = df.copy()
    # Remove colunas PII explicitamente listadas
    for col in _COLUNAS_PII_REMOVIDAS:
        if col in df.columns:
            df = df.drop(columns=col)
    # Remove auxiliares _*
    aux_cols = [c for c in df.columns if c.startswith("_Orig_") or c.startswith("_Fator_")]
    df = df.drop(columns=aux_cols)
    # Ordenação estável: por hash e depois cronológica
    if {"Ano", "Mes"}.issubset(df.columns):
        df = df.sort_values(by=["inst_hash", "Ano", "Mes"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def run_anonymization(
    csv_root: Path,
    output_root: Path,
    corte: Corte,
    settings: AnonymizationSettings,
) -> dict[str, Path]:
    """
    Executa o pipeline completo. Retorna dict com caminhos dos arquivos gerados.
    """
    settings.validate()

    stats = PipelineStats()
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = output_root / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Pasta de saída desta execução: %s", output_dir)

    # ========== FASE 1: LOAD ==========
    log.info("[1/9] Carregando CSVs consolidados de %s", csv_root)
    csv_paths = descobrir_csvs(csv_root)
    stats.instituicoes_encontradas = len(csv_paths)
    if not csv_paths:
        raise RuntimeError(f"Nenhum CSV de instituição encontrado em {csv_root}")

    dataframes: list[pd.DataFrame] = []
    for path in csv_paths:
        df = carregar_csv_instituicao(path)
        if df is None:
            stats.csvs_entrada_malformados.append(path.name)
            continue
        # Garante que o CNPJ8 esteja preenchido corretamente (zero-pad)
        df["CNPJ8"] = df["CNPJ8"].astype(str).str.strip().str.zfill(8)
        stats.linhas_totais_antes_corte += len(df)
        dataframes.append(df)
    log.info("  → %d CSVs carregados (%d linhas totais).",
             len(dataframes), stats.linhas_totais_antes_corte)

    # ========== FASE 2: CORTE TEMPORAL ==========
    log.info("[2/9] Aplicando corte temporal: até %s (inclusive).", corte.label)
    dfs_apos_corte: list[pd.DataFrame] = []
    for df in dataframes:
        cnpj8 = df["CNPJ8"].iloc[0]
        df_cort = aplicar_corte(df, corte)
        if df_cort.empty:
            log.warning("  ⚠ Instituição %s: nenhuma linha até %s, excluída.", cnpj8, corte.label)
            stats.instituicoes_sem_dados_no_corte.append(cnpj8)
            continue
        dfs_apos_corte.append(df_cort)
    if not dfs_apos_corte:
        raise RuntimeError(f"Nenhuma instituição tem dados até o corte {corte.label}.")
    df_all = pd.concat(dfs_apos_corte, ignore_index=True)
    stats.linhas_totais_apos_corte = len(df_all)
    stats.instituicoes_incluidas = df_all["CNPJ8"].nunique()
    log.info("  → %d instituições × %d linhas após corte.",
             stats.instituicoes_incluidas, stats.linhas_totais_apos_corte)

    # ========== FASE 3: HASHING ==========
    log.info("[3/9] Gerando inst_hash SHA-256 (CNPJ8 + SALT).")
    df_all = aplicar_hashing(df_all, settings.salt)

    # ========== FASE 4: CATEGORIZAÇÃO DE AUDITORES ==========
    log.info("[4/9] Categorizando Ident__Auditor_Independente → Big 4 / Outros.")
    df_all = aplicar_categorizacao_auditor(df_all)
    if "Ident__Auditor_Independente" in df_all.columns:
        dist = df_all["Ident__Auditor_Independente"].value_counts().to_dict()
        log.info("  → Distribuição: %s", dist)

    # ========== FASE 5: ESCALONAMENTO ==========
    log.info("[5/9] Sorteando fatores por instituição ∈ [%.2f, %.2f] e escalonando.",
             settings.fator_min, settings.fator_max)
    fatores = gerar_fatores_por_instituicao(df_all["inst_hash"], settings)
    df_all = aplicar_escalonamento(df_all, fatores)
    log.info("  → %d fatores únicos gerados (amostra): %s",
             len(fatores),
             {k[:10] + "…": round(v, 4) for k, v in list(fatores.items())[:3]})

    # ========== FASE 6: RUÍDO SIMÉTRICO ==========
    log.info("[6/9] Aplicando ruído simétrico ∈ [%.2f, %.2f] nas contas de resultado.",
             settings.ruido_min, settings.ruido_max)
    df_all = aplicar_ruido(df_all, settings)

    # ========== FASE 7: RECÁLCULO DE INDICADORES ==========
    log.info("[7/9] Preservando indicadores originais e recalculando (arred. %d casas).",
             settings.indicadores_casas_decimais)
    df_all = preservar_indicadores_originais(df_all)
    df_all = recalcular_indicadores(df_all, settings.indicadores_casas_decimais)

    # ========== FASE 8: AUDITORIA ==========
    log.info("[8/9] Montando mapping_fatores.csv (auditoria).")
    mapping_df = montar_mapping_auditoria(df_all, fatores)

    # ========== FASE 9: EXPORT ==========
    log.info("[9/9] Gravando arquivos finais.")
    dataset_final = finalizar_dataset(df_all)
    stats.linhas_dataset_final = len(dataset_final)

    dataset_path = output_dir / f"dataset_anonymized_{corte.slug}.csv"
    mapping_path = output_dir / "mapping_fatores.csv"
    resumo_path = output_dir / "resumo_execucao.txt"

    dataset_final.to_csv(dataset_path, index=False, sep=";", encoding="utf-8-sig")
    mapping_df.to_csv(mapping_path, index=False, sep=";", encoding="utf-8-sig")
    _gravar_resumo(resumo_path, corte, settings, stats, csv_paths)

    log.info("✓ Dataset: %s (%d linhas)", dataset_path, stats.linhas_dataset_final)
    log.info("✓ Mapping: %s (%d linhas)", mapping_path, len(mapping_df))
    log.info("✓ Resumo:  %s", resumo_path)

    return {
        "dataset": dataset_path,
        "mapping": mapping_path,
        "resumo": resumo_path,
        "output_dir": output_dir,
    }


def _gravar_resumo(
    path: Path,
    corte: Corte,
    settings: AnonymizationSettings,
    stats: PipelineStats,
    csvs_lidos: list[Path],
) -> None:
    """Gera um TXT de auditoria. O SALT é mostrado como hash, NUNCA em claro."""
    salt_fingerprint = hashlib.sha256(settings.salt.encode("utf-8")).hexdigest()[:16]
    lines = [
        "=" * 70,
        "RESUMO DE EXECUÇÃO — Pipeline de Anonimização",
        "=" * 70,
        f"Timestamp:                   {datetime.now().isoformat(timespec='seconds')}",
        f"Corte temporal (inclusivo):  {corte.label}",
        "",
        "--- Parâmetros do .env ---",
        f"SEED:                        {settings.seed}",
        f"SALT (fingerprint SHA-256):  {salt_fingerprint} (primeiros 16 hex)",
        f"FATOR_MIN / FATOR_MAX:       {settings.fator_min} / {settings.fator_max}",
        f"RUIDO_MIN / RUIDO_MAX:       {settings.ruido_min} / {settings.ruido_max}",
        f"Casas decimais indicadores:  {settings.indicadores_casas_decimais}",
        "",
        "--- Estatísticas ---",
        f"Instituições encontradas:    {stats.instituicoes_encontradas}",
        f"Instituições incluídas:      {stats.instituicoes_incluidas}",
        f"Sem dados no corte:          {len(stats.instituicoes_sem_dados_no_corte)} "
            f"({', '.join(stats.instituicoes_sem_dados_no_corte) or '—'})",
        f"CSVs malformados:            {len(stats.csvs_entrada_malformados)} "
            f"({', '.join(stats.csvs_entrada_malformados) or '—'})",
        f"Linhas antes do corte:       {stats.linhas_totais_antes_corte}",
        f"Linhas após o corte:         {stats.linhas_totais_apos_corte}",
        f"Linhas no dataset final:     {stats.linhas_dataset_final}",
        "",
        "--- CSVs de entrada (ordem alfabética) ---",
    ]
    for p in sorted(csvs_lidos):
        # hash do arquivo: útil para provar que o dataset veio dessas exact fontes
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        lines.append(f"  {p.name:20s}  sha256={h}…")
    lines.append("")
    lines.append("=" * 70)
    path.write_text("\n".join(lines), encoding="utf-8")
