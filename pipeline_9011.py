# Autor: Weslley Silva
# Data: junho-2025
# Trabalho de conclusão de cursto - MBA USP

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_9011.py – ETL completo para CADOC 9011 json para csv
Versão: 2025-06-15 (inclui limpeza automática via shutil.rmtree)
"""

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import re
import shutil
import sys
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from jsonschema import Draft7Validator
from dateutil.parser import parse as dt_parse
from tqdm import tqdm

# ─── Diretórios base ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
RAW_DIR    = BASE_DIR / "data" / "raw"
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR   = BASE_DIR / "data" / "gold"
GOLD_DB    = GOLD_DIR / "cadoc_9011.duckdb"
SCHEMA_FP  = BASE_DIR / "9xx1_schema.json"

# ─── Configuração de logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ─── Mapa de prefixos → datas (@data) permitidas ────────────────────────────
FILTER_BY_FILE = {
    "202112": {"A122020"},
    "202212": {"A122021"},
    "202312": {"A122022"},
    "202412": {"A122023", "A122024"},
}

# ─── Padrões de nome para deduplicação I×S ──────────────────────────────────
PATTERNS = [
    re.compile(r'^(?P<cnpj>\d{8})_9011_(?P<tipo>[IS])_(?P<yyyymmdd>\d{8})\.json$'),
    re.compile(r'^(?P<yyyymm>\d{6})-9011-(?P<cnpj>\d{8})\.json$'),
]

# ─── Carrega schema oficial 9xx1 ─────────────────────────────────────────────
SCHEMA    = json.loads(Path(SCHEMA_FP).read_text(encoding="utf-8"))
VALIDATOR = Draft7Validator(SCHEMA)


def load_json_any(fp: Path) -> dict | None:
    """
    Lê JSON suportando gzip e múltiplos encodings,
    preservando acentuação sempre que possível.
    """
    raw = fp.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass

    # 1) UTF-8 BOM strict
    try:
        txt = raw.decode("utf-8-sig")
        return JSONDecoder().raw_decode(txt.lstrip())[0]
    except (UnicodeDecodeError, JSONDecodeError):
        pass

    # 2) UTF-8 strict
    try:
        txt = raw.decode("utf-8")
        return JSONDecoder().raw_decode(txt.lstrip())[0]
    except (UnicodeDecodeError, JSONDecodeError):
        pass

    # 3) Latin-1 com replace
    try:
        txt = raw.decode("latin-1", errors="replace")
        return JSONDecoder().raw_decode(txt.lstrip())[0]
    except JSONDecodeError:
        return None


def iso_from_dt(tag: str) -> str:
    """
    Converte tag de periodicidade (A/S/T/I) em data ISO.
    Ex: 'A122023' -> '2023-12-31'
    """
    p, mm, yyyy = tag[0], tag[1:3], tag[3:]
    if p == "A":
        return f"{yyyy}-12-31"
    if p == "S":
        return f"{yyyy}-{'06-30' if mm == '06' else '12-31'}"
    if p == "T":
        first = dt_parse(f"01/{mm}/{yyyy}", dayfirst=True).date()
        last  = (first.replace(day=28) + pd.Timedelta(days=4)).replace(day=1) - pd.Timedelta(days=1)
        return last.isoformat()
    if p == "I":
        return iso_from_dt(f"S{mm}{yyyy}")
    raise ValueError(f"Tag inválida: {tag}")


def flatten_contas(
    conta: dict,
    bloco: str,
    escopo: str,
    rows: list[dict],
    data_map: dict[str, str],
    meta: dict,
    dt_keep: set[str]
):
    """
    Desenha recursivamente todos os valores cujos vt['@dtBase'] estejam em dt_keep.
    """
    for vt in conta.get(f"valores{escopo}", []):
        if vt["@dtBase"] not in dt_keep:
            continue
        data_ref = data_map.get(vt["@dtBase"])
        if data_ref is None:
            logging.warning("%s: @dtBase %s sem referência – ignorado",
                            meta["@cnpj"], vt["@dtBase"])
            continue
        rows.append({
            "cnpj"      : meta["@cnpj"],
            "periodo"   : iso_from_dt(data_ref),
            "bloco"     : bloco,
            "escopo"    : escopo[:3].upper(),
            "nivel"     : conta["@nivel"],
            "descricao" : conta["@descricao"].strip(),
            "valor"     : vt["@valor"] * meta["_factor"],
        })
    for filho in conta.get("contas", []):
        flatten_contas(filho, bloco, escopo, rows, data_map, meta, dt_keep)


def process_file(fp: Path) -> Tuple[str, str, str, List[dict]]:
    """
    Worker que processa um único JSON:
    - descarta prefixos não listados
    - valida schema
    - filtra só as datas especificadas
    - retorna (status, nome, msg, rows)
    """
    logging.info("➡️  Iniciando %s", fp.name)
    raw = load_json_any(fp)
    if raw is None:
        return "error", fp.name, "JSON ilegível", []
    if raw.get("@codigoDocumento") != "9011":
        return "invalid", fp.name, "Não é 9011", []
    if list(VALIDATOR.iter_errors(raw)):
        return "invalid", fp.name, "Violou schema", []

    prefix  = fp.name[:6]
    allowed = FILTER_BY_FILE.get(prefix)
    if allowed is None:
        logging.info("%s descartado (prefixo %s sem regra)", fp.name, prefix)
        return "skip", fp.name, "prefixo sem regra", []

    # determina quais @id manter, conforme as datas permitidas
    dt_keep = {
        ref["@id"]
        for ref in raw.get("datasBaseReferencia", [])
        if ref.get("@data") in allowed
    }

    meta     = {"@cnpj": raw["@cnpj"], "_factor": int(raw["@unidadeMedida"])}
    data_map = {ref["@id"]: ref["@data"] for ref in raw.get("datasBaseReferencia", [])}

    rows: list[dict] = []
    for bloco_json, bp in [
        ("BalancoPatrimonial",             "BP"),
        ("DemonstracaoDoResultado",        "DR"),
        ("DemonstracaoDoResultadoAbrangente", "DRA"),
        ("DemonstracaoDosFluxosDeCaixa",   "DFC"),
        ("DemonstracaoDasMutacoesDoPatrimonioLiquido", "DMPL"),
    ]:
        for conta in raw.get(bloco_json, {}).get("contas", []):
            flatten_contas(conta, bp, "Individualizados", rows, data_map, meta, dt_keep)
            flatten_contas(conta, bp, "Consolidados",    rows, data_map, meta, dt_keep)

    logging.info("✅  Concluído %s (%d linhas)", fp.name, len(rows))
    return "valid", fp.name, "", rows


def build_file_index() -> List[Path]:
    """
    Dedup I×S e retorna lista de Path para processamento.
    """
    sel, tipo_mem = {}, {}
    for fp in RAW_DIR.glob("*.json"):
        m = next((p.match(fp.name) for p in PATTERNS if p.match(fp.name)), None)
        if not m:
            logging.warning("Nome fora do padrão: %s", fp.name)
            continue

        if "yyyymm" in m.groupdict():
            raw = load_json_any(fp)
            if raw is None:
                continue
            cnpj, base, tipo = raw["@cnpj"], raw["@dataBase"], raw.get("@tipoRemessa", "I")
        else:
            cnpj, base, tipo = (
                m.group("cnpj"),
                m.group("yyyymmdd")[:6],
                m.group("tipo"),
            )

        key = (cnpj, base)
        if key not in sel or (tipo == "S" and tipo_mem[key] == "I"):
            sel[key], tipo_mem[key] = fp, tipo

    return sorted(sel.values())


def qc_balanco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica diferenças ≥ 0.01 entre ativos (níveis 1.x) e passivo+PL (2.x).
    """
    bp      = df[(df["bloco"] == "BP") & (df["escopo"] == "IND")]
    ativos  = bp[bp["nivel"].str.startswith("1")].groupby(["cnpj","periodo"])["valor"].sum()
    passpl  = bp[bp["nivel"].str.startswith("2")].groupby(["cnpj","periodo"])["valor"].sum()
    out     = (ativos - passpl).abs().reset_index(name="diff")
    return out[out["diff"] >= 0.01]


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",    help="processa apenas um JSON específico")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count()-1, 1))
    return ap.parse_args()


def main():
    args = parse_cli()
    files = [RAW_DIR / args.file] if args.file else build_file_index()
    if not files:
        logging.error("Nenhum JSON para processar em %s", RAW_DIR)
        return

    # -- LIMPEZA AUTOMÁTICA das pastas de saída
    if SILVER_DIR.exists():
        shutil.rmtree(SILVER_DIR)
    if GOLD_DIR.exists():
        shutil.rmtree(GOLD_DIR)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    log_csv = GOLD_DIR / "runtime_log.csv"

    all_rows: list[dict] = []
    with open(log_csv, "w", newline="", encoding="utf-8") as lf, mp.Pool(args.workers) as pool:
        writer = csv.writer(lf)
        writer.writerow(["arquivo", "status", "mensagem"])
        pbar = tqdm(total=len(files), desc="Processando 9011")
        for status, fname, msg, rows in pool.imap_unordered(process_file, files):
            writer.writerow([fname, status, msg])
            if status == "valid":
                all_rows.extend(rows)
            pbar.set_postfix(file=fname[:30], status=status)
            pbar.update()
        pbar.close()

    if not all_rows:
        logging.error("Nenhum registro válido gerado.")
        return

    # grava Parquet particionado
    pq.write_to_dataset(
        pa.Table.from_pandas(pd.DataFrame(all_rows)),
        root_path=str(SILVER_DIR),
        partition_cols=["periodo", "cnpj"],
        existing_data_behavior="overwrite_or_ignore",
    )

    # gera CSV de inconsistências de balanço
    bad = qc_balanco(pd.DataFrame(all_rows))
    if not bad.empty:
        bad.to_csv(GOLD_DIR / "bp_inconsistencias.csv", index=False, encoding="utf-8")
        logging.warning("Balanço não fecha para %d instâncias.", len(bad))

    # carrega tudo no DuckDB
    con = duckdb.connect(GOLD_DB)
    con.execute(
        """
        CREATE OR REPLACE TABLE cadoc_9011 AS
        SELECT * FROM parquet_scan(?)
        """,
        (str(SILVER_DIR / "**" / "*.parquet"),),
    )
    con.close()

    logging.info(
        "✅ Pipeline concluído.\n  • Parquet: %s\n  • DuckDB : %s\n  • Log CSV: %s",
        SILVER_DIR, GOLD_DB, log_csv
    )

if __name__ == "__main__":
    main()
