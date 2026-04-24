"""
src/predictor.py
================
Pipeline de predição multi-modelo para análise financeira.

Para cada instituição anonimizada (inst_hash), o pipeline dispara uma requisição
aos 4 provedores de LLM em paralelo (via ThreadPoolExecutor), passando a série
histórica como contexto e recebendo um JSON estruturado com a predição para o
próximo semestre.

Fluxo:
    1. Lê dataset_anonymized_*.csv e mapping_fatores.csv da pasta escolhida.
    2. Agrupa dados por inst_hash e ordena cronologicamente.
    3. Para cada instituição, dispara em paralelo as 4 LLMs.
    4. Salva o JSON bruto de cada resposta em json/.
    5. Ao final, consolida as predições em um CSV pivotado.

Os 4 clientes herdam de `BaseLLMClient` e implementam `generate_json()`. A
orquestração é provider-agnostic — adicionar um 5º provedor é adicionar uma
subclasse e registrá-la em PROVIDERS.
"""
from __future__ import annotations

import glob
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurações e constantes
# ---------------------------------------------------------------------------

# Colunas do dataset anonimizado que NÃO devem ir para o LLM.
# inst_hash é enviado separadamente no prompt; as demais são meta-informação.
_COLUNAS_NAO_ENVIAR_PARA_LLM: set[str] = {"inst_hash"}

# Regex para sanitizar nomes de arquivo vindos do model_id (ex: "meta-llama/..." → "meta-llama-...").
_INVALID_FS_CHARS = re.compile(r'[/\\:*?"<>|]')

# Modelos OpenAI que NÃO aceitam /v1/chat/completions e exigem /v1/responses.
# A OpenAI lançou o "gpt-5.4-pro" e "gpt-5.3-codex" somente via Responses API;
# modelos -pro em geral seguem esse padrão. Allow-list explícita para ser auditável —
# se a OpenAI lançar um novo modelo só-Responses, adicione aqui (substrings são
# casadas como `in`, então "gpt-5.4-pro" casa também com "gpt-5.4-pro-2026-03-05").
MODELOS_RESPONSES_API_ALLOWLIST: tuple[str, ...] = (
    "gpt-5.4-pro",
    "gpt-5.4.1-pro",
    "gpt-5.3-codex",
    "gpt-5.2-pro",
)


@dataclass
class PredictorSettings:
    """Configurações do pipeline preditor lidas do .env."""
    # Credenciais
    openai_api_key: str
    anthropic_api_key: str
    gemini_api_key: str
    groq_api_key: str
    # Modelos
    openai_model: str
    anthropic_model: str
    gemini_model: str
    groq_model: str
    # Comportamento
    max_retries: int
    retry_initial_wait: int
    retry_max_wait: int
    groq_delay_between_calls: int
    request_timeout: int
    # OpenAI: escolha da API (auto/chat/responses) e reasoning effort (só Responses API)
    # - openai_api_flavor: "auto" usa allow-list. "chat" força /v1/chat/completions.
    #   "responses" força /v1/responses (útil se o modelo novo ainda não está na allow-list).
    # - openai_reasoning_effort: só é enviado quando a chamada vai pra Responses API.
    #   Valores aceitos pela OpenAI: "none", "low", "medium", "high", "xhigh".
    openai_api_flavor: str = "auto"
    openai_reasoning_effort: str = "medium"

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY não configurada.")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY não configurada.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não configurada.")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY não configurada.")


@dataclass
class PredictionResult:
    """Resultado de uma única chamada a um provedor."""
    inst_hash: str
    provider_name: str          # "openai" | "anthropic" | "gemini" | "groq"
    model_id: str
    success: bool
    payload: dict[str, Any] | None    # JSON parseado se success, else None
    error: str | None                  # mensagem de erro se !success
    elapsed_seconds: float


class LLMError(Exception):
    """Erro levantado por um cliente LLM, capturado e registrado como falha."""


def sanitize_filename_part(text: str) -> str:
    """Troca caracteres inválidos em filesystem por hífen."""
    return _INVALID_FS_CHARS.sub("-", text)


# ---------------------------------------------------------------------------
# Base class dos clientes LLM
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Contrato comum para os 4 provedores. Cada subclasse implementa
    `_call_llm()` (chamada bruta) e herda `generate_json()` que orquestra
    retry, parse e tratamento de erros.
    """

    # Nome curto usado no nome do arquivo de saída e nos logs.
    PROVIDER_NAME: str = "base"

    def __init__(
        self,
        api_key: str,
        model_id: str,
        settings: PredictorSettings,
    ) -> None:
        if not api_key:
            raise ValueError(f"API key ausente para {self.PROVIDER_NAME}.")
        self.api_key = api_key
        self.model_id = model_id
        self.settings = settings

    # ---- API pública ----

    def generate_json(
        self,
        system_prompt: str,
        user_content: str,
    ) -> dict[str, Any]:
        """Executa chamada com retry e retorna JSON parseado."""

        @retry(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(
                multiplier=self.settings.retry_initial_wait,
                max=self.settings.retry_max_wait,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "[%s] Tentativa %d falhou (%s). Aguardando %.1fs...",
                self.PROVIDER_NAME, rs.attempt_number,
                type(rs.outcome.exception()).__name__ if rs.outcome else "?",
                rs.next_action.sleep if rs.next_action else 0,
            ),
        )
        def _call() -> str:
            return self._call_llm(system_prompt, user_content)

        try:
            raw_text = _call()
        except RetryError as e:
            raise LLMError(f"Todas as {self.settings.max_retries} tentativas falharam: {e}") from e
        except Exception as e:
            raise LLMError(f"Erro inesperado: {e}") from e

        return self._parse_json_response(raw_text, provider_hint=self.PROVIDER_NAME)

    # ---- Implementado pelas subclasses ----

    @abstractmethod
    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """Faz a chamada bruta e retorna o texto da resposta."""

    # ---- Utilitários reutilizados pelas subclasses ----

    @staticmethod
    def _parse_json_response(raw_text: str, provider_hint: str = "desconhecido") -> dict[str, Any]:
        """
        Parse robusto com 4 camadas de tolerância:
            1. Remove cercas de markdown ```json ... ``` (caso o modelo insira).
            2. Remove preâmbulos antes do primeiro `{` — ex: "Aqui está o JSON
               solicitado:" que alguns modelos (especialmente Claude sem prefill)
               podem adicionar antes do payload.
            3. Desembrulha listas de 1 elemento: [{...}] → {...}.
            4. Se o parse direto falhar, tenta reparar o JSON com `json-repair`
               (corrige aspas não escapadas, vírgulas faltando, chaves soltas, etc.).

        Camada 4 gera um WARNING no log indicando qual provedor precisou de reparo
        — isso ajuda a identificar JSONs suspeitos durante análise do TCC.

        Args:
            raw_text: texto bruto retornado pelo LLM.
            provider_hint: nome do provedor (ex: "anthropic") — só para logs.
        """
        cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw_text, flags=re.MULTILINE).strip()

        # --- Camada 2: remove preâmbulo antes do primeiro '{' ou '[' ---
        # Ex: "Aqui está o JSON solicitado:\n{..." → "{..."
        # Só aplica se o texto não começar já com JSON — caso contrário, preserva.
        if cleaned and cleaned[0] not in ("{", "["):
            idx_obj = cleaned.find("{")
            idx_arr = cleaned.find("[")
            # Pega o primeiro caractere de abertura de JSON que aparece
            candidatos = [i for i in (idx_obj, idx_arr) if i >= 0]
            if candidatos:
                primeiro = min(candidatos)
                log.debug(
                    "[%s] Removendo preâmbulo de %d chars antes do JSON",
                    provider_hint, primeiro,
                )
                cleaned = cleaned[primeiro:]

        # --- Camada 1: parse direto (happy path — 99% dos casos) ---
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e_original:
            # --- Camada 4: tenta reparar com json-repair ---
            parsed = BaseLLMClient._try_repair_json(cleaned, provider_hint, e_original)

        # --- Camada 3: desembrulha lista de 1 elemento ---
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            raise LLMError(f"Resposta não é objeto JSON (tipo={type(parsed).__name__}).")

        return parsed

    @staticmethod
    def _try_repair_json(
        cleaned: str, provider_hint: str, original_error: json.JSONDecodeError,
    ) -> Any:
        """
        Última tentativa: usar json-repair. Se o pacote não estiver instalado ou
        o reparo também falhar, levanta LLMError preservando o erro original.
        """
        try:
            # Import lazy: só carrega a lib quando realmente precisa.
            from json_repair import repair_json
        except ImportError:
            preview = cleaned[:300].replace("\n", " ")
            raise LLMError(
                f"Resposta não é JSON válido: {original_error}. "
                f"Preview: {preview}. "
                f"(json-repair não está instalado — considere adicionar ao requirements.txt)"
            ) from original_error

        try:
            # return_objects=True já retorna o Python object parseado, não string.
            repaired = repair_json(cleaned, return_objects=True)
        except Exception as e_repair:
            preview = cleaned[:300].replace("\n", " ")
            raise LLMError(
                f"Resposta não é JSON válido e o reparo também falhou: "
                f"original={original_error}, reparo={e_repair}. Preview: {preview}"
            ) from original_error

        # Reparo pode retornar string vazia se o input for irrecuperável.
        if repaired == "" or repaired is None:
            preview = cleaned[:300].replace("\n", " ")
            raise LLMError(
                f"Resposta não é JSON válido e o reparo produziu resultado vazio. "
                f"Preview: {preview}"
            ) from original_error

        log.warning(
            "[%s] JSON com defeitos reparado automaticamente via json-repair. "
            "Tamanho original=%d chars. Erro original: %s",
            provider_hint, len(cleaned), original_error.msg,
        )
        return repaired



# ---------------------------------------------------------------------------
# Clientes concretos — um por provedor
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """
    Cliente OpenAI híbrido: roteia entre /v1/chat/completions e /v1/responses
    dependendo do modelo configurado.

    Modelos como `gpt-5.4-pro` e `gpt-5.3-codex` foram lançados APENAS na
    Responses API; chamá-los via chat.completions retorna 404. A allow-list
    `MODELOS_RESPONSES_API_ALLOWLIST` lista esses modelos, e a env var
    `OPENAI_API_FLAVOR` (auto/chat/responses) permite override manual.

    Quando usamos Responses API, também aplicamos `reasoning_effort` —
    controla quanto "pensamento interno" o modelo gera antes da resposta.
    Isso só é suportado lá (Chat Completions não aceita esse parâmetro em
    modelos da família gpt-5.4+).
    """
    PROVIDER_NAME = "openai"

    # Limite de tokens na saída. Suficiente para o JSON estruturado do prompt
    # preditivo. Parâmetro se chama `max_output_tokens` em Responses API e
    # `max_tokens` em Chat Completions — tratamos em cada branch.
    MAX_OUTPUT_TOKENS = 4096

    def __init__(self, api_key: str, model_id: str, settings: PredictorSettings) -> None:
        super().__init__(api_key, model_id, settings)
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=settings.request_timeout)
        # Resolve uma única vez no construtor qual flavor vai ser usado.
        # Assim o routing fica explícito no log de inicialização.
        self._use_responses_api = self._resolve_api_flavor()
        flavor_nome = "Responses API (/v1/responses)" if self._use_responses_api else "Chat Completions (/v1/chat/completions)"
        log.info("[openai] Modelo '%s' usará %s", model_id, flavor_nome)

    def _resolve_api_flavor(self) -> bool:
        """
        Decide se usa Responses API (True) ou Chat Completions (False).

        Prioridade:
            1. Override explícito via OPENAI_API_FLAVOR=chat ou =responses
            2. Auto: casa o nome do modelo com MODELOS_RESPONSES_API_ALLOWLIST
               (match por substring — assim 'gpt-5.4-pro-2026-03-05' casa
               com 'gpt-5.4-pro' na allow-list).
        """
        flavor = (self.settings.openai_api_flavor or "auto").strip().lower()
        if flavor == "chat":
            return False
        if flavor == "responses":
            return True
        if flavor != "auto":
            log.warning(
                "[openai] OPENAI_API_FLAVOR='%s' desconhecido, assumindo 'auto'.", flavor
            )
        # Modo auto: verifica se o modelo está na allow-list de Responses-only
        return any(prefix in self.model_id for prefix in MODELOS_RESPONSES_API_ALLOWLIST)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        if self._use_responses_api:
            return self._call_responses_api(system_prompt, user_content)
        return self._call_chat_completions(system_prompt, user_content)

    def _call_chat_completions(self, system_prompt: str, user_content: str) -> str:
        """Chamada clássica — usado pela maioria dos modelos (gpt-5.4, gpt-5.4-mini, etc)."""
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("OpenAI retornou content vazio (chat.completions).")
        return content

    def _call_responses_api(self, system_prompt: str, user_content: str) -> str:
        """
        Chamada na Responses API — usado pelos modelos -pro/codex.

        Diferenças vs chat.completions:
            - system prompt vai em `instructions=` (não em messages)
            - user input vai em `input=`
            - JSON estruturado: `text={"format": {"type": "json_object"}}`
            - Saída acessada via `response.output_text` (helper do SDK)
            - Max tokens: `max_output_tokens`
            - Suporta `reasoning={"effort": "..."}` — reasoning models
        """
        # Monta kwargs incrementalmente — reasoning só entra se configurado
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "instructions": system_prompt,
            "input": user_content,
            "max_output_tokens": self.MAX_OUTPUT_TOKENS,
            "text": {"format": {"type": "json_object"}},
        }
        effort = (self.settings.openai_reasoning_effort or "").strip().lower()
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        response = self._client.responses.create(**kwargs)

        # output_text é um helper do SDK que concatena todas as partes de
        # texto da resposta (Responses API pode retornar múltiplos blocos).
        content = getattr(response, "output_text", None)
        if not content:
            # Fallback: percorre output manualmente
            content = self._extract_text_from_responses_output(response)

        if not content:
            raise LLMError("OpenAI retornou output vazio (responses).")
        return content

    @staticmethod
    def _extract_text_from_responses_output(response) -> str:
        """Fallback defensivo se output_text não estiver disponível no SDK."""
        partes: list[str] = []
        output = getattr(response, "output", None) or []
        for item in output:
            content_list = getattr(item, "content", None) or []
            for c in content_list:
                text = getattr(c, "text", None)
                if text:
                    partes.append(text)
        return "".join(partes).strip()


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Messages API.

    Diferente da OpenAI/Gemini/Groq, a API do Claude NÃO tem um parâmetro
    nativo `response_format=json_object` que garanta JSON válido.

    NOTA HISTÓRICA: versões anteriores deste cliente usavam "prefill" — uma
    técnica em que a última mensagem era {"role": "assistant", "content": "{"}
    para forçar o modelo a continuar a partir desse `{`. Porém, a partir do
    `claude-opus-4-7` a Anthropic passou a rejeitar isso com HTTP 400:
    "This model does not support assistant message prefill. The conversation
    must end with a user message."

    Solução adotada: confiar em duas camadas:
        1. Reforço explícito no system prompt ("retorne APENAS JSON válido").
           Isso já vem no prompt preditivo do usuário, então não adicionamos
           nada aqui — o modelo é disciplinado e tende a obedecer.
        2. `json-repair` como fallback no parser compartilhado, que recupera
           JSONs com aspas não escapadas, vírgulas faltando, etc.
    """
    PROVIDER_NAME = "anthropic"

    # Anthropic pede max_tokens explícito. 4096 é mais que suficiente para
    # o JSON estruturado que o prompt preditivo exige.
    MAX_TOKENS = 4096

    def __init__(self, api_key: str, model_id: str, settings: PredictorSettings) -> None:
        super().__init__(api_key, model_id, settings)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=settings.request_timeout)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content},
            ],
        )
        # Concatena todos os blocos de texto (Anthropic pode retornar múltiplos blocks)
        parts = [block.text for block in response.content if hasattr(block, "text")]
        content = "".join(parts).strip()
        if not content:
            raise LLMError("Anthropic retornou content vazio.")
        return content



class GeminiClient(BaseLLMClient):
    """Google Gemini com response_mime_type=application/json."""
    PROVIDER_NAME = "gemini"

    def __init__(self, api_key: str, model_id: str, settings: PredictorSettings) -> None:
        super().__init__(api_key, model_id, settings)
        # Usamos google-genai (SDK novo unificado), mesmo já usado no pipeline
        # de extração de PDFs — evita duplicar dependências.
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[user_content],
            config=config,
        )
        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini retornou text vazio.")
        return text


class GroqClient(BaseLLMClient):
    """
    Groq (infra para modelos Llama, Mistral, etc.). API é OpenAI-compatível,
    mas usamos o SDK oficial `groq` para ficar explícito.

    Groq tem rate limit baixo em tier free (30 RPM), por isso aplicamos um
    delay obrigatório antes de cada chamada, controlado por um lock global
    (para funcionar corretamente no ThreadPoolExecutor).
    """
    PROVIDER_NAME = "groq"

    # Lock de classe + timestamp da última chamada para throttling global.
    # Se 2 threads tentarem chamar o Groq ao mesmo tempo, o segundo espera.
    _throttle_lock = threading.Lock()
    _last_call_ts: float = 0.0

    def __init__(self, api_key: str, model_id: str, settings: PredictorSettings) -> None:
        super().__init__(api_key, model_id, settings)
        from groq import Groq
        self._client = Groq(api_key=api_key, timeout=settings.request_timeout)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        self._throttle()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("Groq retornou content vazio.")
        return content

    def _throttle(self) -> None:
        """Garante intervalo mínimo entre chamadas (importante no free tier)."""
        delay = self.settings.groq_delay_between_calls
        if delay <= 0:
            return
        with GroqClient._throttle_lock:
            agora = time.monotonic()
            espera = delay - (agora - GroqClient._last_call_ts)
            if espera > 0:
                log.debug("[groq] throttling: aguardando %.1fs", espera)
                time.sleep(espera)
            GroqClient._last_call_ts = time.monotonic()


# Registry: mapeia nome do provedor → classe do cliente
PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
}


# ---------------------------------------------------------------------------
# Formatação dos dados de entrada do LLM
# ---------------------------------------------------------------------------

def formatar_dados_instituicao(df_inst: pd.DataFrame) -> str:
    """
    Converte a série histórica de uma instituição em JSON compacto,
    remove colunas não relevantes e ordena por semestre.

    Saída: string contendo JSON (array de objetos, 1 por semestre).
    """
    df = df_inst.copy()
    # Ordenação cronológica confiável — usa Ano+Mes se tiver, senão Semestre_Ref.
    if {"Ano", "Mes"}.issubset(df.columns):
        df = df.sort_values(by=["Ano", "Mes"]).reset_index(drop=True)
    elif "Semestre_Ref" in df.columns:
        df = df.sort_values(by="Semestre_Ref").reset_index(drop=True)

    # Remove colunas que não devem ir ao LLM
    cols_para_remover = [c for c in _COLUNAS_NAO_ENVIAR_PARA_LLM if c in df.columns]
    df = df.drop(columns=cols_para_remover)

    # Converte para JSON array; orient="records" = lista de dicts
    return df.to_json(orient="records", force_ascii=False, indent=2)


def montar_user_content(inst_hash: str, dados_json: str) -> str:
    """Monta a mensagem do user: inst_hash + série histórica."""
    return (
        f"inst_hash: {inst_hash}\n\n"
        f"Série Histórica de Dados (JSON):\n{dados_json}"
    )


# ---------------------------------------------------------------------------
# Chamada por instituição
# ---------------------------------------------------------------------------

def processar_instituicao(
    inst_hash: str,
    df_inst: pd.DataFrame,
    clients: dict[str, BaseLLMClient],
    system_prompt: str,
    output_json_dir: Path,
    force: bool,
) -> list[PredictionResult]:
    """
    Processa uma instituição: dispara em paralelo os 4 LLMs.

    Args:
        inst_hash: hash SHA-256 da instituição.
        df_inst: DataFrame filtrado com os semestres dessa instituição.
        clients: dict provider_name → client instance.
        system_prompt: conteúdo do prompt_preditivo.txt.
        output_json_dir: pasta onde salvar os JSONs.
        force: se True, reprocessa mesmo que exista JSON em disco.

    Retorna a lista de resultados (um por provedor).
    """
    dados_json = formatar_dados_instituicao(df_inst)
    user_content = montar_user_content(inst_hash, dados_json)

    def _executar_provedor(provider_name: str, client: BaseLLMClient) -> PredictionResult:
        """Função executada em cada thread do pool."""
        json_path = _caminho_arquivo_saida(
            output_json_dir, inst_hash, provider_name, client.model_id, error=False
        )

        # Cache: JSON já existe e não foi pedido --force → pula
        if not force and json_path.is_file():
            log.info("[%s] Hash %s → [CACHE] JSON já existe, pulando.",
                     provider_name, inst_hash[:12])
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Cache corrompido em %s: %s — reprocessando.", json_path, e)
            else:
                return PredictionResult(
                    inst_hash=inst_hash, provider_name=provider_name,
                    model_id=client.model_id, success=True,
                    payload=payload, error=None, elapsed_seconds=0.0,
                )

        start = time.monotonic()
        try:
            payload = client.generate_json(system_prompt, user_content)
            elapsed = time.monotonic() - start
            # Grava JSON de sucesso
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info("[%s] Hash %s | modelo=%s | ✓ Sucesso (%.1fs)",
                     provider_name, inst_hash[:12], client.model_id, elapsed)
            return PredictionResult(
                inst_hash=inst_hash, provider_name=provider_name,
                model_id=client.model_id, success=True,
                payload=payload, error=None, elapsed_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            log.error("[%s] Hash %s | modelo=%s | ✗ Falha: %s",
                      provider_name, inst_hash[:12], client.model_id, e)
            # Salva .error.json para auditoria
            err_path = _caminho_arquivo_saida(
                output_json_dir, inst_hash, provider_name, client.model_id, error=True
            )
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(
                json.dumps(
                    {"error": str(e), "provider": provider_name, "model": client.model_id},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            return PredictionResult(
                inst_hash=inst_hash, provider_name=provider_name,
                model_id=client.model_id, success=False,
                payload=None, error=str(e), elapsed_seconds=elapsed,
            )

    # Executa os 4 provedores em paralelo (1 thread por provedor = 4 workers)
    resultados: list[PredictionResult] = []
    with ThreadPoolExecutor(max_workers=len(clients), thread_name_prefix=f"llm-{inst_hash[:6]}") as pool:
        future_to_prov = {
            pool.submit(_executar_provedor, prov, client): prov
            for prov, client in clients.items()
        }
        for future in as_completed(future_to_prov):
            resultados.append(future.result())

    return resultados


def _caminho_arquivo_saida(
    output_dir: Path, inst_hash: str, provider_name: str, model_id: str, error: bool,
) -> Path:
    """
    Gera o caminho do arquivo no padrão:
        {hash}_{provider}-{model}.json
        {hash}_{provider}-{model}.error.json  (se error=True)

    O modelo é sanitizado: caracteres inválidos em filesystem (como '/' do
    "meta-llama/llama-4-scout...") viram '-'.
    """
    modelo_safe = sanitize_filename_part(model_id)
    suffix = ".error.json" if error else ".json"
    return output_dir / f"{inst_hash}_{provider_name}-{modelo_safe}{suffix}"


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------

def extrair_campos_do_payload(
    payload: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Extrai campos-chave do JSON retornado pelo LLM.

    Retorna tupla (resultado, viabilidade, semestre_alvo, faixa_valor):
        - resultado: "Lucro" | "Prejuízo" | None
        - viabilidade: "Saudável" | "Atenção" | "Alto Risco" | None
        - semestre_alvo: ex. "2025-S1" (o que o LLM disse que está prevendo)
        - faixa_valor: string bruta da faixa, ex. "Entre 12.500.000 e 18.000.000"

    Todos os campos têm fallback para o nível raiz (caso algum LLM achate o JSON).
    """
    pred = payload.get("predicao_proximo_semestre") or {}
    resultado = pred.get("resultado_final_esperado") or payload.get("resultado_final_esperado")
    viabilidade = pred.get("viabilidade_operacional") or payload.get("viabilidade_operacional")
    semestre_alvo = pred.get("semestre_alvo") or payload.get("semestre_alvo")
    proj = pred.get("projecao_resultado_absoluto") or payload.get("projecao_resultado_absoluto") or {}
    faixa_valor = proj.get("faixa_estimada_valor") if isinstance(proj, dict) else None
    return resultado, viabilidade, semestre_alvo, faixa_valor


# Regex para capturar números de uma faixa tipo:
#   "Entre 12.500.000 e 18.000.000"
#   "Entre R$ 12,5M e R$ 18M"
#   "entre -500.000 e 1.200.000"
#   "5000000 a 8000000"  (sem separador)
#
# Pontos-chave do design:
# - Ordem das alternativas importa: tenta FIRST o número longo sem separador
#   (\d{4,}) para não fragmentar "5000000" em "500 / 000 / 0".
# - Depois tenta número com separador de milhar pt-BR/en-US.
# - Sufixos financeiros (M, mil, bi, etc) opcionais.
_NUMERO_RE = re.compile(
    r"-?\s*R?\$?\s*"                    # sinal opcional + R$ opcional
    r"("
    r"\d{4,}(?:[.,]\d+)?"               # 1º: número SEM separador (4+ dígitos seguidos)
    r"|"
    r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?"  # 2º: número COM separador de milhar obrigatório
    r"|"
    r"\d{1,3}(?:[.,]\d+)?"              # 3º: número curto (1-3 dígitos) com decimal opcional
    r")"
    r"\s*(mil|milhão|milhões|MM|mi|M|K|B|bi|bilhões)?",
    re.IGNORECASE,
)

_MULTIPLICADORES_SUFIXO = {
    "": 1,
    "k": 1_000,
    "mil": 1_000,
    "m": 1_000_000,
    "mi": 1_000_000,
    "mm": 1_000_000,
    "milhão": 1_000_000,
    "milhões": 1_000_000,
    "b": 1_000_000_000,
    "bi": 1_000_000_000,
    "bilhões": 1_000_000_000,
}


def _parse_numero_br(raw: str, sufixo: str) -> float | None:
    """
    Converte string numérica em formato pt-BR ou en-US para float.
    Exemplos:
        "12.500.000"  → 12500000.0
        "12,500,000"  → 12500000.0
        "12.5"        → 12.5
        "1,5"         → 1.5
    Aplica multiplicador se houver sufixo (M, mil, etc).
    """
    if not raw:
        return None
    s = raw.strip().replace(" ", "")
    # Heurística: se tem tanto '.' quanto ',', o último é o decimal.
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # vírgula é decimal (pt-BR): remove pontos, troca vírgula por ponto
            s = s.replace(".", "").replace(",", ".")
        else:
            # ponto é decimal (en-US): remove vírgulas
            s = s.replace(",", "")
    elif "," in s:
        # Só vírgula: pode ser milhar (en-US com 3 dígitos) ou decimal (pt-BR).
        # Heurística: se todos os grupos após a vírgula têm exatamente 3 dígitos,
        # é milhar; caso contrário, decimal.
        # EXCEÇÃO: igual ao bloco de ponto — se o primeiro grupo é "0",
        # é decimal ("0,125" = zero vírgula cento e vinte e cinco, não "125").
        partes = s.split(",")
        primeiro_eh_zero = partes[0] == "0" or partes[0] == "-0"
        if not primeiro_eh_zero and all(len(p) == 3 for p in partes[1:]):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        # Só ponto: pode ser milhar pt-BR ou decimal.
        # Heurística financeira: se TODOS os grupos depois do primeiro ponto têm
        # exatamente 3 dígitos, é milhar. Isso cobre casos como:
        #   "12.500"       → 12.500 (milhar, 12 mil e 500)
        #   "1.234.567"    → 1.234.567 (milhar completo)
        # E mantém decimais legítimos:
        #   "12.5"         → 12.5 (só 1 dígito)
        #   "0.25"         → 0.25 (só 2 dígitos)
        # EXCEÇÃO: se o primeiro grupo é "0", é decimal mesmo que tenha 3 dígitos
        # depois ("0.125" = zero vírgula cento e vinte e cinco, não "125").
        # Em contexto financeiro, "zero mil e algo" não faz sentido.
        partes = s.split(".")
        primeiro_eh_zero = partes[0] == "0" or partes[0] == "-0"
        if (
            not primeiro_eh_zero
            and len(partes) >= 2
            and all(len(p) == 3 for p in partes[1:])
        ):
            s = s.replace(".", "")
        # senão, é decimal padrão, mantém

    try:
        valor = float(s)
    except ValueError:
        return None

    mult = _MULTIPLICADORES_SUFIXO.get((sufixo or "").lower().strip(), 1)
    return valor * mult


def extrair_faixa_numerica(raw_faixa: str | None) -> tuple[float, float] | None:
    """
    Extrai min e max de uma faixa em texto livre retornada pelo LLM.

    Retorna (min, max) em float OU None se não conseguir extrair 2 números.
    Pega os PRIMEIROS 2 números que encontrar; se só achar 1, retorna (n, n).
    """
    if not raw_faixa or not isinstance(raw_faixa, str):
        return None
    matches = _NUMERO_RE.findall(raw_faixa)
    # Cada match é (número_limpo, sufixo). Pode haver falsos positivos com sufixo vazio —
    # filtramos para ter números reais.
    numeros: list[float] = []
    for num_raw, sufixo in matches:
        n = _parse_numero_br(num_raw, sufixo)
        if n is not None:
            # Preserva sinal negativo se vier antes do número na string original
            # (regex não captura sinal em grupo separado — fazemos lookup manual)
            idx = raw_faixa.find(num_raw)
            if idx > 0 and raw_faixa[idx - 1:idx].strip() == "-":
                n = -n
            numeros.append(n)
        if len(numeros) >= 2:
            break
    if len(numeros) == 0:
        return None
    if len(numeros) == 1:
        return (numeros[0], numeros[0])
    return (numeros[0], numeros[1])


def desanonimizar_faixa(
    raw_faixa: str | None, fator: float | None,
) -> tuple[str, float | None, float | None]:
    """
    Divide a faixa pelo fator de escalonamento para recuperar a escala real.

    Retorna (faixa_formatada, min_real, max_real).
    Se não conseguir extrair números ou fator inválido, formata como "N/A".
    """
    if not raw_faixa:
        return ("N/A (LLM não informou faixa)", None, None)
    if fator is None or fator == 0:
        return (f"N/A (fator inválido) [bruto: {raw_faixa}]", None, None)

    faixa = extrair_faixa_numerica(raw_faixa)
    if faixa is None:
        return (f"N/A (não foi possível extrair números) [bruto: {raw_faixa}]", None, None)

    min_real = faixa[0] / fator
    max_real = faixa[1] / fator
    return (f"Entre R$ {min_real:,.0f} e R$ {max_real:,.0f}", min_real, max_real)


def inferir_semestre_alvo_do_dataset(dataset_path: Path) -> str | None:
    """
    Lê o dataset anonimizado e descobre qual é o PRÓXIMO semestre depois do
    último disponível. Ex: último = 2024-S2 → retorna "2025-S1".
    """
    try:
        df = pd.read_csv(dataset_path, sep=";", encoding="utf-8-sig",
                         usecols=["Ano", "Mes"])
    except Exception as e:
        log.warning("Não consegui inferir semestre-alvo de %s: %s", dataset_path, e)
        return None

    if df.empty:
        return None

    ano = pd.to_numeric(df["Ano"], errors="coerce").max()
    if pd.isna(ano):
        return None
    # O mês >= 7 indica S2; senão é S1.
    df_max_ano = df[df["Ano"] == ano]
    mes_max = pd.to_numeric(df_max_ano["Mes"], errors="coerce").max()
    ano, mes_max = int(ano), int(mes_max)

    # Calcula o PRÓXIMO semestre.
    if mes_max <= 6:
        # Último foi S1 → próximo é S2 do mesmo ano.
        return f"{ano}-S2"
    else:
        # Último foi S2 → próximo é S1 do ano seguinte.
        return f"{ano + 1}-S1"


def buscar_resultado_real(
    cnpj8: str, semestre_alvo: str, csv_root: Path,
) -> tuple[str, float | None]:
    """
    Busca o resultado REAL (não-anonimizado) para uma instituição no semestre-alvo,
    lendo CSV/<cnpj8>/<cnpj8>.csv.

    Retorna (resultado, valor):
        - resultado: "Lucro" | "Prejuízo" | "N/A (...)"
        - valor: Contas__Lucro_Liquido do semestre alvo, ou None se indisponível.
    """
    if not cnpj8 or not semestre_alvo:
        return ("N/A (sem CNPJ ou semestre)", None)

    csv_path = csv_root / cnpj8 / f"{cnpj8}.csv"
    if not csv_path.is_file():
        return (f"N/A (arquivo {csv_path.name} não encontrado)", None)

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except Exception as e:
        log.warning("Erro ao ler %s: %s", csv_path, e)
        return (f"N/A (erro ao ler CSV)", None)

    # Semestre_Ref tem BOM zero-width em algumas células — normaliza strip()
    if "Semestre_Ref" not in df.columns:
        return ("N/A (CSV sem coluna Semestre_Ref)", None)

    linha = df[df["Semestre_Ref"].astype(str).str.strip() == semestre_alvo]
    if linha.empty:
        return (f"N/A ({semestre_alvo} ainda não disponível)", None)

    if "Contas__Lucro_Liquido" not in linha.columns:
        return ("N/A (CSV sem coluna Contas__Lucro_Liquido)", None)

    valor = pd.to_numeric(linha["Contas__Lucro_Liquido"].iloc[0], errors="coerce")
    if pd.isna(valor):
        return (f"N/A ({semestre_alvo} com valor inválido)", None)

    valor_float = float(valor)
    if valor_float > 0:
        return ("Lucro", valor_float)
    elif valor_float < 0:
        return ("Prejuízo", valor_float)
    else:
        return ("Neutro (zero)", valor_float)


def calcular_acuracia(previsto: str | None, real: str) -> str:
    """
    Marca acertou/errou comparando previsão do LLM com resultado real.
    Retorna "✓ Acertou", "✗ Errou", ou "—" se impossível avaliar.
    """
    if not previsto or previsto == "INDISPONÍVEL":
        return "—"
    # Extrai só a parte de Lucro/Prejuízo da célula "Lucro | Saudável"
    previsto_clean = previsto.split("|")[0].strip()
    if previsto_clean not in ("Lucro", "Prejuízo"):
        return "—"
    if real not in ("Lucro", "Prejuízo"):
        return "—"
    return "✓ Acertou" if previsto_clean == real else "✗ Errou"


def consolidar_predicoes(
    json_dir: Path,
    mapping_df: pd.DataFrame,
    output_path: Path,
    csv_root: Path | None = None,
    dataset_path: Path | None = None,
) -> pd.DataFrame:
    """
    Varre json_dir, extrai predições, desanonimiza faixa de lucro usando fator
    do mapping, cruza com resultado real (se disponível), marca acurácia.

    Schema do CSV gerado (por coluna):
        Identificação:
            - Ident__Razao_Social, CNPJ8, inst_hash
            - Semestre_Alvo_Inferido, Fator_Desanonimizacao_Usado
        Por (provider × modelo), 2 colunas:
            - [PROVIDER]_{modelo}_Resultado           → "Lucro | Saudável"
            - [PROVIDER]_{modelo}_Lucro_Projetado_Real → "Entre R$ X e R$ Y"
        Ground truth (últimas colunas):
            - Real_Resultado                           → "Lucro" | "Prejuízo" | "N/A ..."
            - Real_Lucro_Liquido
            - Real_Acuracia_{PROVIDER}_{modelo} (1 por modelo) → "✓ Acertou" | "✗ Errou" | "—"

    Args:
        csv_root: raiz dos CSVs originais (CSV/). Se None, ground truth não é calculado.
        dataset_path: caminho do dataset anonimizado (para inferir semestre-alvo).
    """
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Pasta de JSONs não existe: {json_dir}")

    arquivos = sorted(p for p in json_dir.glob("*.json") if not p.name.endswith(".error.json"))
    if not arquivos:
        raise RuntimeError(f"Nenhum JSON de sucesso em {json_dir}.")

    log.info("Consolidando %d JSONs de %s", len(arquivos), json_dir)

    # --- Passo 1: inferir semestre-alvo (único para toda a rodada) ---
    semestre_alvo_inferido: str | None = None
    if dataset_path is not None:
        semestre_alvo_inferido = inferir_semestre_alvo_do_dataset(dataset_path)
        if semestre_alvo_inferido:
            log.info("Semestre-alvo inferido: %s (calculado a partir do dataset)",
                     semestre_alvo_inferido)
        else:
            log.warning("Não foi possível inferir semestre-alvo do dataset.")

    # --- Passo 2: montar dict fator_por_hash (a partir do mapping) ---
    fator_por_hash: dict[str, float] = {}
    if "Fator_Escalonamento" in mapping_df.columns and "inst_hash" in mapping_df.columns:
        # Primeira linha por hash (o fator é igual em todas as linhas da mesma inst.)
        for inst_hash, sub in mapping_df.groupby("inst_hash"):
            fator_por_hash[inst_hash] = float(sub["Fator_Escalonamento"].iloc[0])
    else:
        log.warning("mapping_fatores.csv sem coluna Fator_Escalonamento — "
                    "desanonimização desabilitada.")

    # --- Passo 3: parse de cada JSON em formato long ---
    pattern = re.compile(r"^([a-f0-9]{64})_([a-z]+)-(.+)\.json$", re.IGNORECASE)
    registros: list[dict[str, Any]] = []
    for path in arquivos:
        m = pattern.match(path.name)
        if not m:
            log.warning("Nome de arquivo fora do padrão esperado: %s (ignorado)", path.name)
            continue
        inst_hash, provider, model = m.group(1), m.group(2), m.group(3)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("JSON inválido em %s: %s (ignorado)", path.name, e)
            continue

        resultado, viabilidade, sem_alvo_llm, faixa_bruta = extrair_campos_do_payload(payload)

        # Desanonimiza a faixa de lucro/prejuízo projetado
        fator = fator_por_hash.get(inst_hash)
        faixa_real, _, _ = desanonimizar_faixa(faixa_bruta, fator)

        registros.append({
            "inst_hash": inst_hash,
            "provider": provider,
            "model": model,
            "resultado_celula": _formatar_celula(resultado, viabilidade),
            "faixa_real": faixa_real,
            "semestre_alvo_llm": sem_alvo_llm,  # guardado para debug, não entra no CSV
        })

    if not registros:
        raise RuntimeError("Nenhum registro válido extraído dos JSONs.")

    df_long = pd.DataFrame(registros)

    # --- Passo 4: pivota (uma linha por hash, 2 colunas por provedor-modelo) ---
    df_long["col_base"] = df_long.apply(
        lambda r: f"{r['provider'].upper()}_{r['model']}", axis=1,
    )
    df_long["col_resultado"] = df_long["col_base"] + "_Resultado"
    df_long["col_faixa"] = df_long["col_base"] + "_Lucro_Projetado_Real"

    pivot_res = df_long.pivot_table(
        index="inst_hash", columns="col_resultado",
        values="resultado_celula", aggfunc="first",
    ).reset_index()
    pivot_faixa = df_long.pivot_table(
        index="inst_hash", columns="col_faixa",
        values="faixa_real", aggfunc="first",
    ).reset_index()

    for df_p in (pivot_res, pivot_faixa):
        df_p.columns.name = None

    # Fill para (hash, provedor) que falharam
    for col in pivot_res.columns:
        if col != "inst_hash":
            pivot_res[col] = pivot_res[col].fillna("INDISPONÍVEL")
    for col in pivot_faixa.columns:
        if col != "inst_hash":
            pivot_faixa[col] = pivot_faixa[col].fillna("INDISPONÍVEL")

    # --- Passo 5: merge com mapping (PII + fator) ---
    mapping_recente = (
        mapping_df.sort_values(by="Semestre_Ref")
        .groupby("inst_hash", as_index=False)
        .agg({
            "Ident__Razao_Social": "last",
            "Ident__CNPJ": "last",
            "Fator_Escalonamento": "first",  # fator é fixo por hash
        })
    )

    consolidado = mapping_recente.merge(pivot_res, on="inst_hash", how="right")
    consolidado = consolidado.merge(pivot_faixa, on="inst_hash", how="left")

    # Deriva CNPJ8
    consolidado["CNPJ8"] = consolidado["Ident__CNPJ"].apply(_extrair_cnpj8)

    # Adiciona metadados fixos
    consolidado["Semestre_Alvo_Inferido"] = semestre_alvo_inferido or "N/A"
    consolidado = consolidado.rename(columns={"Fator_Escalonamento": "Fator_Desanonimizacao_Usado"})

    # --- Passo 6: buscar resultado real + calcular acurácia por modelo ---
    col_resultado_llm = sorted([c for c in consolidado.columns if c.endswith("_Resultado")])
    col_faixa_llm = sorted([c for c in consolidado.columns if c.endswith("_Lucro_Projetado_Real")])

    if csv_root is not None and semestre_alvo_inferido:
        log.info("Buscando resultado real do semestre %s em %s/...",
                 semestre_alvo_inferido, csv_root)
        reais: list[tuple[str, float | None]] = []
        for cnpj8 in consolidado["CNPJ8"]:
            reais.append(buscar_resultado_real(cnpj8, semestre_alvo_inferido, csv_root))
        consolidado["Real_Resultado"] = [r[0] for r in reais]
        consolidado["Real_Lucro_Liquido"] = [r[1] for r in reais]

        # Colunas de acurácia: uma por (provedor, modelo)
        for col_res in col_resultado_llm:
            col_base = col_res.removesuffix("_Resultado")
            col_acuracia = f"Real_Acuracia_{col_base}"
            consolidado[col_acuracia] = consolidado.apply(
                lambda row: calcular_acuracia(row[col_res], row["Real_Resultado"]),
                axis=1,
            )

        # Log amigável: quantos acertos cada modelo teve
        col_acuracia_criadas = [c for c in consolidado.columns if c.startswith("Real_Acuracia_")]
        for c in col_acuracia_criadas:
            stats = consolidado[c].value_counts().to_dict()
            log.info("  %s → %s", c, stats)
    else:
        consolidado["Real_Resultado"] = "N/A (ground truth não consultado)"
        consolidado["Real_Lucro_Liquido"] = None

    # --- Passo 7: ordenar colunas final ---
    colunas_id = [
        "Ident__Razao_Social", "CNPJ8", "inst_hash",
        "Semestre_Alvo_Inferido", "Fator_Desanonimizacao_Usado",
    ]
    # Intercalar Resultado + Lucro_Projetado por provider-modelo, na ordem alfabética
    colunas_modelos: list[str] = []
    for col_res in col_resultado_llm:
        col_base = col_res.removesuffix("_Resultado")
        colunas_modelos.append(col_res)
        col_faixa_corresp = f"{col_base}_Lucro_Projetado_Real"
        if col_faixa_corresp in consolidado.columns:
            colunas_modelos.append(col_faixa_corresp)

    colunas_real = ["Real_Resultado", "Real_Lucro_Liquido"]
    colunas_acuracia = sorted([c for c in consolidado.columns
                                if c.startswith("Real_Acuracia_")])
    ordem_final = colunas_id + colunas_modelos + colunas_real + colunas_acuracia
    # Mantém só colunas que existem (defensivo)
    ordem_final = [c for c in ordem_final if c in consolidado.columns]
    consolidado = consolidado[ordem_final]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidado.to_csv(output_path, sep=";", encoding="utf-8-sig", index=False)
    log.info("✓ Consolidado salvo: %s (%d linhas × %d colunas)",
             output_path, len(consolidado), len(consolidado.columns))

    return consolidado


def _formatar_celula(resultado: str | None, viabilidade: str | None) -> str:
    """Junta resultado + viabilidade numa célula legível."""
    r = resultado if resultado else "?"
    v = viabilidade if viabilidade else "?"
    return f"{r} | {v}"


def _extrair_cnpj8(cnpj14: Any) -> str:
    """
    Extrai os primeiros 8 dígitos do CNPJ14 (retorna '' se inválido).

    IMPORTANTE: quando o pandas lê o mapping, ele converte CNPJ para float64
    porque é "só dígitos" — perdendo zero à esquerda. Ex: '01027058000191'
    (Cielo) vira 1027058000191.0 no pandas.

    Armadilha: str(1027058000191.0) = '1027058000191.0'. Se simplesmente
    removermos não-dígitos, o zero que sobra do '.0' final fica colado,
    gerando '10270580001910' — que tem 14 dígitos mas "deslocados".

    Solução: se for numérico, converter para int ANTES (elimina o .0).
    Depois aplicar zfill(14) para restaurar zeros à esquerda.
    """
    if pd.isna(cnpj14):
        return ""

    # Se for numérico (int, float, np.float64), converter para int remove o ".0".
    # Para outros tipos (string), usar str() normalmente.
    if isinstance(cnpj14, (int, float)) or (
        hasattr(cnpj14, "dtype") and "float" in str(cnpj14.dtype)
    ):
        try:
            texto = str(int(cnpj14))
        except (ValueError, OverflowError):
            texto = str(cnpj14)
    else:
        texto = str(cnpj14)

    # Remove tudo que não é dígito (barra, traço, etc. em CNPJ formatado).
    so_digitos = re.sub(r"\D", "", texto)
    if not so_digitos:
        return ""
    # Padroniza para 14 dígitos (zfill preserva zero à esquerda).
    cnpj_padded = so_digitos.zfill(14)
    return cnpj_padded[:8]


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def run_prediction_pipeline(
    pasta_trabalho: Path,
    system_prompt: str,
    settings: PredictorSettings,
    force: bool = False,
    apenas_consolidar: bool = False,
) -> Path:
    """
    Executa o pipeline completo (ou só a consolidação se apenas_consolidar=True).

    Retorna o caminho do CSV consolidado gerado.
    """
    settings.validate()

    # --- Localizar arquivos de entrada ---
    dataset_candidates = glob.glob(str(pasta_trabalho / "dataset_anonymized_*.csv"))
    if not dataset_candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo 'dataset_anonymized_*.csv' encontrado em {pasta_trabalho}."
        )
    if len(dataset_candidates) > 1:
        log.warning("Múltiplos datasets encontrados, usando o primeiro: %s", dataset_candidates[0])
    dataset_path = Path(dataset_candidates[0])
    mapping_path = pasta_trabalho / "mapping_fatores.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"mapping_fatores.csv não encontrado em {pasta_trabalho}.")

    json_dir = pasta_trabalho / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # --- Carregar mapping (usado em ambos os modos) ---
    log.info("Lendo mapping_fatores.csv: %s", mapping_path)
    mapping_df = pd.read_csv(mapping_path, sep=";", encoding="utf-8-sig")

    # --- Modo "apenas consolidar": pula as chamadas LLM ---
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_consolidado = pasta_trabalho / f"consolidado_predicoes_modelos_{timestamp}.csv"

    # Deduz a raiz dos CSVs originais: CSV/anonimizados/<timestamp>/ → CSV/
    csv_root_originais = pasta_trabalho.parent.parent

    if apenas_consolidar:
        log.info("Modo --apenas-consolidar ativo: pulando chamadas LLM.")
        consolidar_predicoes(
            json_dir, mapping_df, output_consolidado,
            csv_root=csv_root_originais,
            dataset_path=dataset_path,
        )
        return output_consolidado

    # --- Carregar dataset anonimizado ---
    log.info("Lendo dataset anonimizado: %s", dataset_path)
    df = pd.read_csv(dataset_path, sep=";", encoding="utf-8-sig")
    log.info("  → %d linhas × %d colunas.", len(df), len(df.columns))

    if "inst_hash" not in df.columns:
        raise ValueError("Dataset não contém coluna 'inst_hash'.")

    # --- Instanciar os 4 clientes ---
    log.info("Instanciando clientes LLM...")
    clients: dict[str, BaseLLMClient] = {}
    for name, cls in PROVIDERS.items():
        api_key = getattr(settings, f"{name}_api_key")
        model_id = getattr(settings, f"{name}_model")
        clients[name] = cls(api_key=api_key, model_id=model_id, settings=settings)
        log.info("  ✓ %s (modelo=%s)", name, model_id)

    # --- Agrupar por instituição e processar ---
    grupos = df.groupby("inst_hash", sort=False)
    total_instituicoes = len(grupos)
    log.info("Processando %d instituições × %d provedores = %d chamadas no total.",
             total_instituicoes, len(PROVIDERS), total_instituicoes * len(PROVIDERS))

    sucessos, falhas = 0, 0
    for idx, (inst_hash, df_inst) in enumerate(grupos, start=1):
        log.info("━" * 70)
        log.info("[%d/%d] Instituição %s (%d semestres)",
                 idx, total_instituicoes, inst_hash[:12] + "...", len(df_inst))
        resultados = processar_instituicao(
            inst_hash=inst_hash,
            df_inst=df_inst,
            clients=clients,
            system_prompt=system_prompt,
            output_json_dir=json_dir,
            force=force,
        )
        for r in resultados:
            if r.success:
                sucessos += 1
            else:
                falhas += 1

    log.info("━" * 70)
    log.info("✓ Chamadas concluídas: %d sucessos, %d falhas", sucessos, falhas)

    # --- Consolidar resultados ---
    log.info("Consolidando predições em CSV...")
    consolidar_predicoes(
        json_dir, mapping_df, output_consolidado,
        csv_root=csv_root_originais,
        dataset_path=dataset_path,
    )

    return output_consolidado
