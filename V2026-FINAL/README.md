# TCC — Análise de Instituições de Pagamento

Pipeline em Python que lê os balanços públicos (CADOC 9010) de IPs reguladas pelo BCB, envia cada PDF para o Gemini e consolida os resultados em um CSV por instituição. Base para posterior análise preditiva (lucro/prejuízo próximo semestre).

## Estrutura do projeto

```
projeto_tcc/
├── .env                          # suas credenciais (crie a partir do .env.example)
├── .env.example                  # modelo das variáveis
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── config.py                     # configurações centralizadas
├── main.py                       # entrada CLI
├── requirements.txt
├── prompts/
│   └── extracao_cadoc9010.txt    # prompt do Gemini — edite livremente
├── src/
│   ├── gemini_client.py          # wrapper da API (File API + retry)
│   ├── institution_processor.py  # orquestração por CNPJ
│   ├── csv_builder.py            # montagem do CSV consolidado
│   └── utils.py                  # logging + parsing de nomes
├── PDF/                          # ENTRADA — coloque as pastas 2019-12, 2020-06, ...
│   ├── 2019-12/
│   │   ├── 201912-9010-13140088.pdf
│   │   └── ...
│   ├── 2020-06/
│   └── ...
├── CSV/                          # SAÍDA (criada automaticamente)
│   └── 13140088/
│       ├── 13140088.csv
│       ├── 201912-9010-13140088.json
│       ├── 202006-9010-13140088.json
│       └── ...
└── logs/
    └── run.log
```

## Setup (primeira vez)

### 1. Configurar credenciais

```bash
cp .env.example .env
```

Edite o `.env` e preencha `GEMINI_API_KEY` (pegue a sua em https://aistudio.google.com/apikey).

### 2. Colocar os PDFs

Crie as pastas semestrais dentro de `PDF/` seguindo o padrão:

```
PDF/2019-12/
PDF/2020-06/
PDF/2020-12/
...
PDF/2025-12/
```

E dentro de cada pasta, os 19 PDFs no padrão `AAAAMM-9010-CNPJ8.pdf`.

### 3. Build da imagem Docker

```bash
docker compose build
```

No Mac Silicon o build é ARM64 nativo — sem emulação.

## Execução

Todos os comandos assumem que você está na raiz do projeto.

### Modo padrão (processa tudo, usa cache)
```bash
docker compose run --rm app
```
Se um JSON já existir em `CSV/<cnpj8>/`, o arquivo não é reenviado ao Gemini (economia de custo e tempo). O CSV consolidado é regenerado de qualquer forma.

### Reprocessar tudo (ignora cache)
```bash
docker compose run --rm app --force
```

### Processar só uma instituição
```bash
docker compose run --rm app --only 13140088
```

### Dry-run (só lista, não chama API)
```bash
docker compose run --rm app --dry-run
```

### Log mais verboso (debug)
```bash
docker compose run --rm app --log-level DEBUG
```

## Saída

Para cada instituição (`CNPJ8`):

- `CSV/<cnpj8>/<cnpj8>.csv` — CSV consolidado (um semestre por linha, ordenado cronologicamente).
- `CSV/<cnpj8>/<AAAAMM-9010-cnpj8>.json` — um JSON bruto por semestre (exatamente como veio do Gemini).

O CSV usa `;` como separador e encoding `utf-8-sig` (abre direto no Excel em português).

## Colunas do CSV

A ordem é fixa em todos os CSVs. Prefixos indicam a origem no JSON:

| Prefixo      | Significado                     |
|--------------|---------------------------------|
| (sem prefixo)| Contexto (ano, semestre, CNPJ, arquivo) |
| `Ident__`    | Identificação e auditoria       |
| `Contas__`   | Contas contábeis extraídas      |
| `Indic__`    | Indicadores Assaf Neto          |

Se um dia você atualizar o prompt e adicionar campos novos, o pipeline avisa no log (`Campo novo no JSON não mapeado para CSV...`) e você só precisa incluir a coluna em `src/csv_builder.py`.

## Configurações úteis (`.env`)

| Variável               | Default                    | Função |
|------------------------|----------------------------|--------|
| `GEMINI_MODEL`         | `gemini-3.1-pro-preview`   | Modelo a usar. Troque para `gemini-3-flash-preview` se quiser economizar. |
| `MAX_RETRIES`          | `5`                        | Tentativas por arquivo antes de desistir. |
| `DELAY_BETWEEN_CALLS`  | `2`                        | Segundos entre chamadas bem-sucedidas (ajuda com rate limit). |
| `FILE_ACTIVE_TIMEOUT`  | `300`                      | Máximo de espera para PDF ficar ACTIVE na File API. |

## Rodando sem Docker (opcional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Troubleshooting

**`Variável obrigatória 'GEMINI_API_KEY' não configurada`**
Você esqueceu de preencher o `.env` ou ainda está com o placeholder.

**`JSON inválido em <arquivo>`**
Raro com `response_mime_type=application/json`, mas pode acontecer se o PDF for muito ruim. O JSON bruto não é salvo nesse caso — rode com `--log-level DEBUG` para ver o preview da resposta.

**Rate limit 429 recorrente**
Aumente `DELAY_BETWEEN_CALLS` no `.env` (ex.: `5`).

**Timeout esperando ACTIVE**
PDFs muito grandes (~50MB) podem demorar. Aumente `FILE_ACTIVE_TIMEOUT`.
