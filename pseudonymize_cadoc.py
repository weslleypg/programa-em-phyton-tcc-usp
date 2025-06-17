# Autor: Weslley Silva
# Data: junho-2025
# Trabalho de conclusão de cursto - MBA USP
#!/usr/bin/env python3
"""pseudonymize_cadoc.py
----------------------------------------------------------------------------
Este script:
  • Lê o CSV CADOC bruto (coluna `cnpj`) em `data/gold/` **preservando zeros
    à esquerda**, tratando o CNPJ como string.
  • Gera `inst_hash` (hash SHA‑256 determinístico) usando `PROJETO_SALT`.
  • Aplica multiplicador secreto por instituição e ruído opcional.
  • Salva:
      - CSV pseudonimizado → `data/cadoc_anonimizado/cadoc_9011-pseudo.csv`
      - `mapping_fatores.csv` com `inst_hash`, `cnpj` e `k_multiplier`.
----------------------------------------------------------------------------
COMO UTILIZAR
    1. Defina `PROJETO_SALT` e `NOISE_PCT` abaixo.
    2. Coloque `cadoc_9011-anonimizar.csv` em `data/gold/`.
    3. Execute:
           python pseudonymize_cadoc.py
"""

# -------------------------------------------------------------------------
# CONFIGURAÇÕES DO USUÁRIO
# -------------------------------------------------------------------------
PROJETO_SALT = "troque_este_salt_por_um_valor_secreto"  # ⚠️ Altere para o seu SALT
NOISE_PCT = 0.00  # Ex.: 0.02 aplica ±2 % de ruído; 0 => sem ruído
MULT_MIN, MULT_MAX = 0.60, 1.40  # Faixa do multiplicador por instituição
# -------------------------------------------------------------------------

import hashlib
import secrets
import pathlib
import sys
import pandas as pd

# Caminhos e nomes de arquivos
RAW_FILENAME = "cadoc_9011.csv"
GOLD_DIR = "data/gold"
ANON_DIR = "data/cadoc_anonimizado"
ANON_FILENAME = "cadoc_9011-pseudo.csv"
MAP_FILE = "mapping_fatores.csv"

def hash_cnpj(cnpj: str) -> str:
    """Gera hash SHA‑256 determinístico do CNPJ usando o PROJETO_SALT."""
    return hashlib.sha256((PROJETO_SALT + cnpj).encode()).hexdigest()

def main():
    # 1. Garantir diretórios
    pathlib.Path(GOLD_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ANON_DIR).mkdir(parents=True, exist_ok=True)

    raw_path = pathlib.Path(GOLD_DIR) / RAW_FILENAME
    if not raw_path.exists():
        sys.exit(f"[ERRO] Arquivo bruto não encontrado: {raw_path}")

    # 2. Carregar CSV preservando zeros à esquerda do CNPJ
    #    dtype={"cnpj": str} evita que o pandas converta para inteiro.
    df = pd.read_csv(raw_path, dtype={"cnpj": str})
    if {"cnpj", "valor"}.difference(df.columns):
        sys.exit("[ERRO] O CSV precisa conter as colunas 'cnpj' e 'valor'.")

    # 3. Gerar inst_hash
    df["inst_hash"] = df["cnpj"].apply(hash_cnpj)

    # 4. Multiplicadores secretos por instituição
    rng = secrets.SystemRandom()
    multipliers = {h: rng.uniform(MULT_MIN, MULT_MAX) for h in df["inst_hash"].unique()}

    # 5. Transformar valores monetários
    def transformar(row):
        valor = row["valor"] * multipliers[row["inst_hash"]]
        if NOISE_PCT > 0:
            valor *= (1 + rng.uniform(-NOISE_PCT, NOISE_PCT))
        return valor

    df["valor"] = df.apply(transformar, axis=1)

    # 6. Salvar mapeamento (agora mantendo CNPJ como string intacta)
    mapeamento = (
        df[["inst_hash", "cnpj"]]
        .drop_duplicates()
        .assign(k_multiplier=lambda d: d["inst_hash"].map(multipliers))
    )
    mapeamento.to_csv(MAP_FILE, index=False)

    # 7. Remover CNPJ do dataset pseudonimizado
    df = df.drop(columns=["cnpj"])
    df = df[["inst_hash"] + [c for c in df.columns if c != "inst_hash"]]

    # 8. Salvar CSV pseudonimizado
    anon_path = pathlib.Path(ANON_DIR) / ANON_FILENAME
    df.to_csv(anon_path, index=False)

    print("✅ Dataset pseudonimizado salvo em:", anon_path)
    print("✅ Mapeamento salvo em (mantenha seguro):", MAP_FILE)
    print("Linhas processadas:", len(df))
    print("Instituições únicas:", len(mapeamento))

if __name__ == "__main__":
    main()
