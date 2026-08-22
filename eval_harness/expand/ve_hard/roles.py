"""Named LLM roles for VE-hard, configured without persisting secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from ..llm import LLM, LLMRouter


# Override VE_HARD_API_BASE (and the per-role model env vars below) with your
# own OpenAI-compatible gateway. No default gateway is assumed for public use.
DEFAULT_API_BASE = os.environ.get("VE_HARD_API_BASE", "http://127.0.0.1:8000/v1")


@dataclass(frozen=True)
class RoleConfig:
    api_base: str
    golden_model: str
    test_model: str
    prompt_model: str
    judge_model: str
    concurrency: int = 4
    timeout: float = 1200.0

    @classmethod
    def from_env(cls, config: Dict[str, object] | None = None) -> "RoleConfig":
        cfg = config or {}
        roles = cfg.get("roles", {}) if isinstance(cfg.get("roles", {}), dict) else {}
        return cls(
            api_base=os.environ.get("VE_HARD_API_BASE") or str(roles.get("api_base") or DEFAULT_API_BASE),
            golden_model=os.environ.get("VE_HARD_GOLDEN_MODEL") or str(roles.get("golden_model") or ""),
            test_model=os.environ.get("VE_HARD_TEST_MODEL") or str(roles.get("test_model") or ""),
            prompt_model=os.environ.get("VE_HARD_PROMPT_MODEL") or str(roles.get("prompt_model") or ""),
            judge_model=os.environ.get("VE_HARD_JUDGE_MODEL") or str(roles.get("judge_model") or ""),
            concurrency=int(roles.get("concurrency", 4)),
            timeout=float(roles.get("timeout", 1200.0)),
        )

    def validate(self) -> None:
        if not self.api_base.startswith(("http://", "https://")):
            raise ValueError("roles.api_base must be an HTTP(S) URL")
        for name in ("golden_model", "test_model", "prompt_model", "judge_model"):
            if not getattr(self, name):
                raise ValueError(
                    f"roles.{name} is unset. Set VE_HARD_{name.upper()} or "
                    "provide it in the ve_hard config before running.")


@dataclass
class RoleClients:
    golden: LLMRouter
    test: LLMRouter
    prompt: LLMRouter
    judge: LLMRouter
    config: RoleConfig


def build_role_clients(config: Dict[str, object] | None = None) -> RoleClients:
    role_config = RoleConfig.from_env(config)
    role_config.validate()
    api_key = os.environ.get("VE_HARD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VE_HARD_API_KEY is required; credentials are never read from YAML")

    def router(model: str, max_tokens: int | None, temperature: float) -> LLMRouter:
        llm = LLM(
            endpoint=role_config.api_base,
            api_key=api_key,
            model=model,
            concurrency=role_config.concurrency,
            timeout=role_config.timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMRouter(traj_llm=llm, prompt_llms=[llm])

    return RoleClients(
        golden=router(role_config.golden_model, None, 1.0),
        test=router(role_config.test_model, None, 1.0),
        prompt=router(role_config.prompt_model, None, 1.0),
        judge=router(role_config.judge_model, None, 1.0),
        config=role_config,
    )


async def health_check(clients: RoleClients) -> Dict[str, Dict[str, str]]:
    checks = {}
    for name, router in (
        ("golden", clients.golden), ("test", clients.test),
        ("prompt", clients.prompt), ("judge", clients.judge),
    ):
        response = await router.chat_full(
            "Reply with exactly OK.", "Health check. Reply with exactly OK."
        )
        content = (response.content or "").strip()
        checks[name] = {
            "model": getattr(router.traj_llm, "model", "?"),
            "ok": str("OK" in content).lower(),
            "finish_reason": response.finish_reason,
        }
    failed = [name for name, result in checks.items() if result["ok"] != "true"]
    if failed:
        raise RuntimeError(f"role health check failed: {', '.join(failed)}")
    return checks
