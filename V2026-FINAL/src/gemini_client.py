"""
src/gemini_client.py
====================
Wrapper da API Gemini para extração estruturada de balanços (CADOC 9010).

Estratégia:
    1. Upload do PDF via File API (obrigatório para PDFs > ~20MB, seguro para todos).
    2. Polling do estado do arquivo até ficar ACTIVE (ou falhar / esgotar timeout).
    3. Chamada a generate_content com o arquivo + prompt, em modo JSON (saída estruturada).
    4. Parse resiliente da resposta (lida com respostas que vêm com ```json ``` apesar
       da instrução do prompt — defesa em profundidade).
    5. Cleanup: remoção do arquivo uploaded (a File API retém arquivos por 48h de qualquer
       forma, mas apagar explicitamente é boa prática e evita acúmulo).

Retries são aplicados na chamada de geração (onde ocorrem os 429/500/503 típicos).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


class GeminiExtractionError(Exception):
    """Erro específico do fluxo de extração para diferenciar de erros genéricos."""


# Regex para remover cercas de markdown que o modelo às vezes insiste em incluir,
# mesmo o prompt pedindo JSON puro. Defesa em profundidade.
_MD_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class GeminiClient:
    """
    Encapsula todas as interações com a Gemini API para este projeto.

    Uma única instância deve ser reutilizada durante toda a execução
    (reaproveita o pool de conexões HTTP do SDK).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        prompt: str,
        max_retries: int,
        retry_initial_wait: int,
        retry_max_wait: int,
        file_active_timeout: int,
        file_poll_interval: int,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.file_active_timeout = file_active_timeout
        self.file_poll_interval = file_poll_interval

        # Config de retry dinâmica — não dá para usar @retry decorator estático
        # pois os parâmetros vêm de settings em runtime.
        self._max_retries = max_retries
        self._retry_initial_wait = retry_initial_wait
        self._retry_max_wait = retry_max_wait

        # Client do SDK novo (google-genai). A API key também é lida automaticamente
        # de GEMINI_API_KEY se não for passada, mas aqui passamos explicitamente
        # para ficar claro de onde veio.
        self._client = genai.Client(api_key=api_key)

        log.info("GeminiClient iniciado | modelo=%s | retries=%d", model, max_retries)

    # ------------------------------------------------------------------ #
    # API pública                                                        #
    # ------------------------------------------------------------------ #

    def extract_from_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Fluxo completo: upload → wait ACTIVE → generate → parse JSON → cleanup.

        Retorna o dicionário JSON já parseado. Levanta GeminiExtractionError
        se algo falhar em definitivo (após todos os retries).
        """
        if not pdf_path.is_file():
            raise GeminiExtractionError(f"Arquivo não existe: {pdf_path}")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        log.info("Enviando PDF: %s (%.2f MB)", pdf_path.name, file_size_mb)

        uploaded = None
        try:
            uploaded = self._upload(pdf_path)
            self._wait_until_active(uploaded)
            raw_text = self._generate_with_retry(uploaded)
            return self._parse_json(raw_text, source=pdf_path.name)
        finally:
            # Cleanup mesmo em caso de erro — evita lixo na File API.
            if uploaded is not None:
                self._safe_delete(uploaded)

    # ------------------------------------------------------------------ #
    # Etapas internas                                                    #
    # ------------------------------------------------------------------ #

    def _upload(self, pdf_path: Path):
        """Upload via File API. Para PDFs até 50MB é o método correto."""
        try:
            return self._client.files.upload(
                file=str(pdf_path),
                config={"mime_type": "application/pdf"},
            )
        except Exception as e:
            raise GeminiExtractionError(f"Falha no upload de {pdf_path.name}: {e}") from e

    def _wait_until_active(self, uploaded) -> None:
        """
        Faz polling até o arquivo ficar ACTIVE. PDFs grandes podem demorar alguns
        segundos para serem indexados pela Gemini antes de estarem prontos.
        """
        deadline = time.monotonic() + self.file_active_timeout
        current = uploaded

        while current.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                raise GeminiExtractionError(
                    f"Timeout ({self.file_active_timeout}s) aguardando arquivo "
                    f"{current.name} ficar ACTIVE."
                )
            log.debug("Arquivo %s ainda PROCESSING, aguardando...", current.name)
            time.sleep(self.file_poll_interval)
            current = self._client.files.get(name=current.name)

        if current.state.name != "ACTIVE":
            raise GeminiExtractionError(
                f"Arquivo {current.name} terminou em estado inesperado: {current.state.name}"
            )

        log.info("Arquivo pronto: %s (estado=ACTIVE)", current.name)
        # Atualiza os campos do objeto original para que a chamada seguinte use o estado correto
        uploaded.state = current.state
        uploaded.uri = current.uri
        uploaded.mime_type = current.mime_type

    def _generate_with_retry(self, uploaded) -> str:
        """Chama generate_content com retry configurável."""

        # Construímos o decorator dinamicamente com os parâmetros do settings.
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=self._retry_initial_wait,
                max=self._retry_max_wait,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "Tentativa %d falhou (%s). Aguardando %.1fs antes da próxima...",
                rs.attempt_number,
                type(rs.outcome.exception()).__name__ if rs.outcome else "desconhecido",
                rs.next_action.sleep if rs.next_action else 0,
            ),
        )
        def _do_call() -> str:
            # response_mime_type=application/json força o modelo a retornar JSON válido.
            # É o mecanismo oficial de "structured output" e reduz drasticamente
            # respostas malformadas.
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                # Temperatura é mantida em 1.0 (padrão) porque a documentação oficial
                # do Gemini 3 avisa que alterar pode degradar raciocínio complexo.
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[uploaded, self.prompt],
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                raise GeminiExtractionError("Resposta vazia do modelo.")
            return text

        try:
            return _do_call()
        except RetryError as e:
            raise GeminiExtractionError(
                f"Todas as {self._max_retries} tentativas falharam: {e}"
            ) from e

    @staticmethod
    def _parse_json(raw_text: str, source: str) -> dict[str, Any]:
        """
        Faz parse robusto do JSON. Mesmo com response_mime_type=json, vale ter
        tolerância a cercas de markdown que eventualmente aparecem.

        Também normaliza o caso observado em que o modelo embrulha a resposta em
        uma lista de 1 elemento (`[{...}]`) — comportamento esporádico que viola
        o schema solicitado no prompt.
        """
        cleaned = _MD_FENCE_RE.sub("", raw_text).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Log da resposta bruta é essencial para debug — truncado para não poluir.
            preview = cleaned[:500].replace("\n", " ")
            log.error("JSON inválido em %s. Preview: %s", source, preview)
            raise GeminiExtractionError(
                f"Resposta para {source} não é JSON válido: {e}"
            ) from e

        # Desembrulha lista de 1 elemento: o Gemini às vezes retorna [{...}] em
        # vez de {...} apesar do prompt pedir objeto. Assim, o JSON salvo em
        # disco fica sempre no formato canônico (dict).
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            log.warning(
                "Resposta de %s veio como lista de 1 elemento — desembrulhando.",
                source,
            )
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            raise GeminiExtractionError(
                f"Resposta de {source} não é um objeto JSON válido "
                f"(tipo recebido: {type(parsed).__name__})."
            )

        return parsed

    def _safe_delete(self, uploaded) -> None:
        """Remove o arquivo uploaded, sem levantar exceção se falhar."""
        try:
            self._client.files.delete(name=uploaded.name)
            log.debug("Arquivo %s removido da File API.", uploaded.name)
        except Exception as e:  # noqa: BLE001 — cleanup não deve quebrar o fluxo
            log.warning("Falha ao remover %s da File API: %s", uploaded.name, e)
