"""Model resolver: maps routing tiers to actual model endpoints."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default model assignments per tier (all overridable via config)
DEFAULT_TIER_MODELS = {
    "local": {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",  # Ollama doesn't need API key
        "reasoning": {"enabled": False},
        "max_tokens": 2048,
    },
    "cheap": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "reasoning": {"enabled": False},
        "max_tokens": 4096,
    },
    "standard": {
        "provider": "primary",  # Use Hermes primary model
        "model": "",  # Empty = inherit primary
        "reasoning": {"enabled": False},
        "max_tokens": 8192,
    },
    "code": {
        "provider": "deepseek",
        "model": "deepseek-coder",
        "api_key_env": "DEEPSEEK_API_KEY",
        "reasoning": {"enabled": True, "effort": "medium"},
        "max_tokens": 16384,
    },
    "reasoning": {
        "provider": "primary",  # Use strongest available (primary)
        "model": "",
        "reasoning": {"enabled": True, "effort": "high"},
        "max_tokens": 32768,
    },
}


class ModelResolver:
    """Resolves routing tiers to concrete model configurations.

    Supports:
    - Ollama local models
    - OpenAI-compatible API endpoints
    - Inheriting Hermes primary model
    - Per-tier config overrides
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        noesis_cfg = self.config.get("noesis", {})
        self.tier_configs = self._build_tier_configs(noesis_cfg)
        self._primary_runtime: dict[str, Any] = {}

    def _build_tier_configs(self, noesis_cfg: dict) -> dict[str, dict]:
        """Merge default tier configs with user overrides."""
        configs = {}
        user_tiers = noesis_cfg.get("models", {})

        for tier, defaults in DEFAULT_TIER_MODELS.items():
            merged = {**defaults}
            if tier in user_tiers:
                user_cfg = user_tiers[tier]
                if isinstance(user_cfg, dict):
                    merged.update(user_cfg)
                elif isinstance(user_cfg, str):
                    merged["model"] = user_cfg
            configs[tier] = merged

        return configs

    def set_primary_runtime(self, runtime: dict[str, Any]) -> None:
        """Store the primary Hermes model runtime for 'primary' provider tiers."""
        self._primary_runtime = dict(runtime)

    def resolve(self, tier: str) -> dict[str, Any]:
        """Resolve a tier to a complete model configuration.

        Returns:
            Dict with keys: model, provider, base_url, api_key,
                           reasoning_config, max_tokens, is_local.
            Falls back to primary model if tier config is invalid.
        """
        tier_cfg = self.tier_configs.get(tier, {})

        if not tier_cfg:
            logger.warning("NOESIS: unknown tier '%s', falling back to primary", tier)
            return self._primary_fallback()

        provider = tier_cfg.get("provider", "primary")
        model = tier_cfg.get("model", "")

        # "primary" means use whatever Hermes is configured with
        if provider == "primary" or not model:
            return self._primary_fallback(tier=tier)

        # Resolve API key
        api_key = ""
        api_key_env = tier_cfg.get("api_key_env", "")
        if api_key_env:
            api_key = os.getenv(api_key_env, "")
            if not api_key:
                logger.warning(
                    "NOESIS: %s tier requires %s but env var is not set, falling back",
                    tier, api_key_env,
                )
                return self._primary_fallback(tier=tier)

        is_local = provider == "ollama" or "localhost" in tier_cfg.get("base_url", "")

        # Check local model availability
        if is_local and not self._check_local_available(tier_cfg.get("base_url", "")):
            logger.warning(
                "NOESIS: local model endpoint not available for %s tier, falling back",
                tier,
            )
            return self._primary_fallback(tier=tier)

        return {
            "model": model,
            "provider": provider,
            "base_url": tier_cfg.get("base_url"),
            "api_key": api_key,
            "reasoning_config": tier_cfg.get("reasoning", {"enabled": False}),
            "max_tokens": tier_cfg.get("max_tokens", 4096),
            "is_local": is_local,
            "tier": tier,
        }

    def _primary_fallback(self, tier: str = "standard") -> dict[str, Any]:
        """Return the primary Hermes model configuration."""
        rt = self._primary_runtime
        return {
            "model": rt.get("model", ""),
            "provider": rt.get("provider", "auto"),
            "base_url": rt.get("base_url"),
            "api_key": rt.get("api_key", ""),
            "reasoning_config": {"enabled": False},
            "max_tokens": self.tier_configs.get(tier, {}).get("max_tokens", 8192),
            "is_local": False,
            "tier": tier,
            "fallback": True,
        }

    @staticmethod
    def _check_local_available(base_url: str) -> bool:
        """Check if Ollama/local endpoint is reachable."""
        if not base_url:
            return False
        try:
            import urllib.request
            # Quick health check to Ollama tags endpoint
            req = urllib.request.Request(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": "Bearer ollama"},
            )
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False

    def available_tiers(self) -> list[str]:
        """Return list of tiers that are currently usable."""
        available = []
        for tier in self.tier_configs:
            resolved = self.resolve(tier)
            if not resolved.get("fallback") or tier in ("standard", "reasoning"):
                available.append(tier)
        return available
