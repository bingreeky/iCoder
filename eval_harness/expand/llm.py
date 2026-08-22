"""LLM client + router for multi-model expansion.

Two layers:

* :class:`LLM` — async wrapper around an OpenAI-compatible chat endpoint.
  Supports max_tokens=None (use model default — required for reasoning
  models, where reasoning tokens are billed against the same
  budget as content). Returns :class:`LLMResponse` with ``content``
  *and* ``reasoning_content`` separately so callers can persist both.
* :class:`LLMRouter` — splits ``traj`` calls and ``prompt`` calls. Traj
  calls always go to a single fixed model (the SFT assistant must come
  from one model for training stability); prompt calls round-robin
  through ``prompt_llms`` so problem-side text gets multi-model diversity.

For dry-run / smoke tests, :class:`DryRunLLM` returns a deterministic
mock transformation so the whole pipeline can be exercised offline.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop a leading think block if any. For non-reasoning
    models this is a belt-and-braces clean-up. For reasoning models the
    think block is part of the SFT assistant content and
    should NOT be stripped — call sites that want to preserve the full
    content read ``LLMResponse.content`` directly instead of the legacy
    string-returning :meth:`LLM.chat`."""
    if not text:
        return text
    out = THINK_RE.sub("", text)
    if "</think>" in out:
        out = out.split("</think>", 1)[1].lstrip()
    return out.strip()


@dataclass
class LLMResponse:
    """Full LLM response. ``content`` is the visible answer (the
    ``message.content`` field). ``reasoning_content`` is the parallel
    reasoning stream that reasoning models emit alongside; empty string
    on plain models.

    ``truncated`` is True when ``finish_reason`` indicates ``length``
    or when the content shape suggests truncation (e.g. open
    ``<think>`` without closing ``</answer>``). Caller
    should retry on truncation."""

    content: str = ""
    reasoning_content: str = ""
    finish_reason: str = ""
    model: str = ""
    truncated: bool = False
    raw_message: Optional[Any] = None  # for debugging only


class LLM:
    """Async wrapper around an OpenAI-compatible chat endpoint.

    Supports **multi-key rotation**: when constructed with ``api_keys``
    as a list, the LLM owns N AsyncOpenAI clients (one per key) and
    round-robins each call across them. Each credential gets its own
    ``asyncio.Semaphore(concurrency)`` so per-credential concurrency stays at
    the configured value. This supports providers that issue multiple scoped
    credentials with independent rate limits."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        api_keys: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = 2048,
        concurrency: int = 8,
        timeout: float = 1200.0,  # user spec: 1200s per-request (enough for 64k gen at cuda-graph+GDN fast decode ~1000tok/s; sane bound avoids infinite retry storms). Was 600→APITimeoutError on long gens.
        enable_thinking: bool = True,
    ):
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not installed")
        # Resolve key list: explicit ``api_keys`` overrides single
        # ``api_key`` (legacy single-key callers still work).
        keys = list(api_keys) if api_keys else [api_key]
        if not keys:
            keys = ["EMPTY"]
        self.api_keys: List[str] = keys
        self.clients = [
            AsyncOpenAI(base_url=endpoint, api_key=k, timeout=timeout)
            for k in keys
        ]
        # One semaphore per client = per-key concurrency limit. With N
        # keys + concurrency=20 the effective cap is 40 in-flight.
        self.sems: List[asyncio.Semaphore] = [
            asyncio.Semaphore(concurrency) for _ in keys
        ]
        # Round-robin counter; protected by lock for atomicity.
        self._next_client_idx = 0
        self._next_client_lock = asyncio.Lock()
        self.model = model
        # Per-call sampling args. ``max_tokens=None`` means "don't pass"
        # — required for reasoning models because reasoning_tokens share the
        # budget with content_tokens; capping prematurely truncates the
        # answer (60-97% of budget can be reasoning).
        self.sampling: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            self.sampling["max_tokens"] = max_tokens
        # ``chat_template_kwargs.enable_thinking`` controls whether the
        # provider returns the reasoning trace as ``reasoning_content``.
        # Default True so reasoning models surface their chain-of-thought
        # (used as the SFT think/answer wrapper body via synthesize_v4pro_wrap).
        # For non-reasoning models this kwarg is silently ignored.
        self.extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
        # Legacy single-key callers may read ``.sem``; alias to first.
        self.sem = self.sems[0]

    async def _pick_client(self):
        """Round-robin: return (client, sem) for the next API key."""
        async with self._next_client_lock:
            i = self._next_client_idx
            self._next_client_idx = (self._next_client_idx + 1) % len(self.clients)
        return self.clients[i], self.sems[i], i

    async def chat_full(self, system: str, user: str) -> LLMResponse:
        """Return the full response object (content + reasoning_content +
        finish_reason). Use this when the SFT trajectory needs the
        think/answer wrapper.
        Uses the OpenAI SDK's ``with_raw_response`` accessor to get the
        raw HTTP response and parse JSON manually — the SDK's strict
        Pydantic model (``ChatCompletion``) silently drops
        non-standard fields like ``reasoning_content`` (reasoning models emit it). Raw-JSON parse keeps the
        reasoning trace intact for the SFT ``<think>`` body.

        Retries up to 5 times. Quota / membership errors get a longer
        token-bucket-style backoff (30s/60s/90s/120s) since the
        upstream gateway refills its bucket on a ~minute
        window. Plain transients (502/503/timeouts) get the short
        path (3s/6s/12s/20s)."""
        if not self.model:
            raise RuntimeError(
                "LLM model is unset. Pass model=... to LLM(), or set "
                "EXPAND_LLM_MODEL / EXPAND_TRAJ_MODEL (and EXPAND_LLM_BASE_URL) "
                "before issuing any query.")
        import asyncio as _asyncio
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client, sem, key_idx = await self._pick_client()
                async with sem:
                    raw_resp = await (client.chat.completions
                        .with_raw_response.create(
                            model=self.model, messages=msgs,
                            extra_body=self.extra_body,
                            **self.sampling))
                body = raw_resp.http_response.json()
                break
            except Exception as e:
                msg = str(e)
                code = getattr(e, "status_code", None)
                low_msg = msg.lower()
                is_quota = "quota" in low_msg or "rate limit" in low_msg \
                    or code == 429
                is_membership = "membership" in low_msg or code == 402
                is_transient = (
                    is_quota or is_membership
                    or code in (500, 502, 503, 504)
                    or "Bad Gateway" in msg
                    or "service unavailable" in low_msg
                    or "timeout" in low_msg
                    or "timed out" in low_msg
                    or type(e).__name__ in ("APITimeoutError", "APIConnectionError")
                )
                if is_transient and attempt < max_retries - 1:
                    if is_quota:
                        delay = 30 * (attempt + 1)
                    elif is_membership:
                        delay = min(3 * (2 ** attempt), 20)
                    else:
                        delay = min(3 * (2 ** attempt), 20)
                    await _asyncio.sleep(delay)
                    continue
                import sys as _sys
                _kidx = locals().get("key_idx", "?")
                print(f"[LLM] model={self.model} key_idx={_kidx} "
                      f"attempt={attempt+1}/{max_retries} RAISE: "
                      f"{type(e).__name__}: {str(e)[:160]}",
                      file=_sys.stderr)
                raise

        choices = body.get("choices") or [{}]
        choice = choices[0]
        msg_dict = choice.get("message") or {}
        content = msg_dict.get("content") or ""
        # Non-standard reasoning stream emitted by reasoning models. The OpenAI SDK Pydantic
        # model would drop this — we parse JSON ourselves to preserve.
        reasoning = msg_dict.get("reasoning_content") or ""
        finish = choice.get("finish_reason") or ""
        truncated = (finish == "length")
        if not truncated and content and "<think>" in content.lower():
            if "</answer>" not in content.lower():
                truncated = True
        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            finish_reason=finish,
            model=self.model,
            truncated=truncated,
            raw_message=msg_dict,
        )

    async def chat(self, system: str, user: str) -> str:
        """Legacy plain-string interface. Returns ``content`` with any
        leading think block stripped (compatibility). For callers that need the *raw* think+
        answer wrapper, use :meth:`chat_full` instead."""
        resp = await self.chat_full(system, user)
        return strip_thinking(resp.content)

    async def chat_many(self, prompts: List[tuple]) -> List[str]:
        return await asyncio.gather(*(self.chat(s, u) for s, u in prompts))


class DryRunLLM:
    """Offline stub. Returns a deterministic mock transformation."""

    def __init__(self, model: str = "dryrun"):
        self.model = model

    async def chat_full(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(
            content=f"[DRYRUN model={self.model}]\n{user}\n[END]",
            model=self.model,
            finish_reason="stop",
        )

    async def chat(self, system: str, user: str) -> str:
        return (await self.chat_full(system, user)).content

    async def chat_many(self, prompts):
        return [await self.chat(s, u) for s, u in prompts]


# ---------- Router -----------------------------------------------------

class LLMRouter:
    """Splits ``traj`` and ``prompt`` calls.

    * ``chat_traj`` always uses ``traj_llm`` (single fixed model — the
      SFT assistant must come from one model).
    * ``chat_prompt`` rotates through ``prompt_llms`` deterministically
      by ``(seed_hash, variant_idx)``. ``variant_idx=0..M-1`` covers
      the M prompt-side variants per ``(seed, op)``.

    If ``prompt_llms`` is empty the router routes prompt calls to the
    traj_llm too (backward-compat for single-model runs). When you want
    multi-model prompts, pass ``prompt_llms`` from
    :func:`make_router_from_env`."""

    def __init__(self, traj_llm: Any, prompt_llms: Optional[Sequence[Any]] = None):
        self.traj_llm = traj_llm
        self.prompt_llms: List[Any] = list(prompt_llms) if prompt_llms else [traj_llm]

    @property
    def num_prompt_models(self) -> int:
        return len(self.prompt_llms)

    @property
    def traj_model_name(self) -> str:
        return getattr(self.traj_llm, "model", "?")

    def prompt_model_name(self, seed_hash: int = 0, variant_idx: int = 0) -> str:
        idx = (seed_hash + variant_idx) % len(self.prompt_llms)
        return getattr(self.prompt_llms[idx], "model", "?")

    async def chat_traj(self, system: str, user: str) -> str:
        """String interface for traj calls — strips ``<think>`` for
        non-reasoning models, like the legacy ``LLM.chat``."""
        return await self.traj_llm.chat(system, user)

    async def chat_traj_full(self, system: str, user: str) -> LLMResponse:
        """Full response: keeps ``<think>...</think><answer>...
        </answer>`` content intact for the SFT assistant."""
        return await self.traj_llm.chat_full(system, user)

    async def chat_prompt(self, system: str, user: str, *,
                          seed_hash: int = 0, variant_idx: int = 0) -> str:
        idx = (seed_hash + variant_idx) % len(self.prompt_llms)
        return await self.prompt_llms[idx].chat(system, user)

    async def chat_prompt_full(self, system: str, user: str, *,
                               seed_hash: int = 0,
                               variant_idx: int = 0) -> LLMResponse:
        idx = (seed_hash + variant_idx) % len(self.prompt_llms)
        return await self.prompt_llms[idx].chat_full(system, user)

    # Backward-compat: legacy expand methods that take a single ``llm``
    # parameter and call ``llm.chat(system, user)`` keep working —
    # router quacks like an LLM by routing to traj_llm.
    async def chat(self, system: str, user: str) -> str:
        return await self.chat_traj(system, user)

    async def chat_full(self, system: str, user: str) -> LLMResponse:
        return await self.chat_traj_full(system, user)


# ---------- Factories --------------------------------------------------

def collect_api_keys(prefix: str) -> List[str]:
    """Collect API keys for an env ``prefix`` (e.g. ``"EXPAND_LLM"`` or
    ``"EXPAND_TRAJ"``). Reads ``{prefix}_API_KEYS`` (comma-separated) first,
    else ``{prefix}_API_KEY`` + ``{prefix}_API_KEY1`` / _KEY2 / … Returns an
    empty list when none are set (caller decides the fallback)."""
    env_csv = os.environ.get(f"{prefix}_API_KEYS", "").strip()
    if env_csv:
        return [k.strip() for k in env_csv.split(",") if k.strip()]
    out: List[str] = []
    for suffix in ("", "1", "2", "3", "4", "5", "6", "7"):
        k = os.environ.get(f"{prefix}_API_KEY{suffix}", "").strip()
        if k:
            out.append(k)
    return out


def make_llm(
    dry_run: bool = False,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    if dry_run:
        return DryRunLLM(model=model or "dryrun")
    return LLM(
        endpoint=endpoint or os.environ.get(
            "EXPAND_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        model=model or os.environ.get("EXPAND_LLM_MODEL"),
        api_key=api_key or os.environ.get("EXPAND_LLM_API_KEY", "EMPTY"),
        **kwargs,
    )


def make_router_from_env(
    dry_run: bool = False,
    concurrency: int = 4,
    temperature: float = 0.2,
    traj_temperature: Optional[float] = None,
    traj_max_tokens: Optional[int] = None,
    prompt_max_tokens: Optional[int] = 16384,
    traj_concurrency: Optional[int] = None,
) -> LLMRouter:
    """Build a router from env vars.

    ``EXPAND_TRAJ_MODEL``      — single fixed model for SFT assistant
                                 (no default — set this, or queries fail fast).
    ``EXPAND_PROMPT_MODELS``   — comma-separated list of models for the
                                 problem-side rollouts. If unset, falls
                                 back to traj-only (single-model run).
    ``EXPAND_LLM_BASE_URL``    — OpenAI-compatible endpoint shared by the
                                 prompt-side models.
    ``EXPAND_LLM_API_KEY``     — single key shared across models on this
                                 endpoint.

    Temperature: ``temperature`` is the prompt-side default; ``traj_
    temperature`` overrides it for the traj model. Some reasoning models
    require temperature=1 server-side and reject
    other values — auto-detected by traj model name when
    ``traj_temperature`` is unset.

    For traj the default ``max_tokens`` is None (don't pass): reasoning
    models consume 60-97% of the budget and capping truncates the
    answer. For prompt-side we keep a sane cap (4096) for cost control;
    prompt rewrites are short."""
    if dry_run:
        traj = DryRunLLM(model=os.environ.get("EXPAND_TRAJ_MODEL", "dryrun-traj"))
        prompt_models_env = os.environ.get("EXPAND_PROMPT_MODELS", "")
        prompt_names = [m.strip() for m in prompt_models_env.split(",") if m.strip()]
        prompts = [DryRunLLM(model=n) for n in prompt_names] or [traj]
        return LLMRouter(traj_llm=traj, prompt_llms=prompts)

    base_url = os.environ.get("EXPAND_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    # Collect API keys for a given env prefix. Either ``{prefix}_API_KEYS``
    # (comma-separated) or ``{prefix}_API_KEY`` + ``{prefix}_API_KEY1`` / ...
    api_keys = collect_api_keys("EXPAND_LLM") or ["EMPTY"]

    # The trajectory model may use a separate endpoint from the prompt models.
    # ``EXPAND_TRAJ_BASE_URL`` / ``EXPAND_TRAJ_API_KEY(S)`` override
    # the shared ``EXPAND_LLM_*`` for the traj LLM only; unset => traj shares
    # the prompt endpoint (legacy single-gateway behaviour, unchanged).
    traj_base_url = os.environ.get("EXPAND_TRAJ_BASE_URL", "").strip() or base_url
    traj_api_keys = collect_api_keys("EXPAND_TRAJ") or api_keys
    traj_model = os.environ.get("EXPAND_TRAJ_MODEL")

    # Per-model temperature constraints. Some hosted models reject non-1.0
    # temperatures (BadRequestError("invalid temperature")); list them via
    # ``EXPAND_LOCKED_TEMP_MODELS`` (comma-separated, matched as substrings,
    # case-insensitive). Empty by default — add your models as needed.
    _LOCKED_TO_T1 = tuple(
        s.strip() for s in os.environ.get("EXPAND_LOCKED_TEMP_MODELS", "").split(",")
        if s.strip()
    )

    def _temp_for(model_name: str, default: float) -> float:
        low = model_name.lower()
        for tag in _LOCKED_TO_T1:
            if tag in low:
                return 1.0
        return default

    if traj_temperature is None:
        traj_temperature = _temp_for(traj_model or "", temperature)
    if traj_concurrency is None:
        # Default: honour the caller's `concurrency` directly. Caller can
        # pin a smaller `traj_concurrency` via make_router_from_env(
        # traj_concurrency=...) if a model's upstream quota needs it.
        traj_concurrency = concurrency

    traj = LLM(
        endpoint=traj_base_url, api_keys=traj_api_keys, model=traj_model,
        temperature=traj_temperature, max_tokens=traj_max_tokens,
        concurrency=traj_concurrency,
    )

    prompt_models_env = os.environ.get("EXPAND_PROMPT_MODELS", "").strip()
    if not prompt_models_env:
        return LLMRouter(traj_llm=traj, prompt_llms=[traj])
    prompt_names = [m.strip() for m in prompt_models_env.split(",") if m.strip()]
    # Per-prompt-model concurrency: providers may enforce different limits by
    # model. List models that should use ``traj_concurrency`` via
    # ``EXPAND_LOW_CONCURRENCY_MODELS`` (comma-separated, substring match,
    # case-insensitive). Unlisted models get the full ``concurrency`` budget.
    _LOW_CONCURRENCY = tuple(
        s.strip() for s in os.environ.get("EXPAND_LOW_CONCURRENCY_MODELS", "").split(",")
        if s.strip()
    )

    def _per_prompt_concurrency(name: str) -> int:
        low = name.lower()
        if any(tag and tag.lower() in low for tag in _LOW_CONCURRENCY):
            return traj_concurrency
        return concurrency

    # Per-model endpoint routing: namespaced model identifiers use the prompt
    # endpoint, while bare identifiers use the trajectory endpoint. When the
    # trajectory endpoint is unset, both resolve to the shared endpoint.
    def _endpoint_keys_for_prompt(name: str):
        if "/" in name:
            return base_url, api_keys          # namespaced id
        return traj_base_url, traj_api_keys    # bare id

    prompts = []
    for name in prompt_names:
        p_url, p_keys = _endpoint_keys_for_prompt(name)
        prompts.append(
            LLM(endpoint=p_url, api_keys=p_keys, model=name,
                temperature=_temp_for(name, temperature),
                max_tokens=prompt_max_tokens,
                concurrency=_per_prompt_concurrency(name))
        )
    return LLMRouter(traj_llm=traj, prompt_llms=prompts)
