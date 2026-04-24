# ==========================================================================
# Dockerfile — TCC Análise de IPs
# --------------------------------------------------------------------------
# Imagem slim para manter o build leve. python:3.12-slim é multi-arch
# (amd64 + arm64), então roda nativamente em Mac Silicon sem emulação.
# ==========================================================================
FROM python:3.12-slim AS base

# --- Variáveis de ambiente de boas práticas ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Dependências de sistema mínimas (ca-certificates para HTTPS confiável;
# tini para PID 1 decente, lidando com sinais SIGINT/SIGTERM corretamente).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos apenas requirements primeiro para aproveitar o cache de layer
# caso só o código mude (build mais rápido em iterações).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Agora o resto do código.
COPY . .

# Usuário não-root por segurança (evita escrever como root nos volumes montados).
RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app
USER app

# Tini como init — trata Ctrl+C corretamente quando rodar interativamente.
ENTRYPOINT ["/usr/bin/tini", "--", "python", "main.py"]

# Sem argumentos por padrão: processa tudo com cache ativo.
CMD []
