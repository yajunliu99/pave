"""Shared CreateAI per-request routing settings for clients and experiments."""
from __future__ import annotations

REQUEST_SOURCE = "override_params"
REQUEST_PROTOCOL_VERSION = "createai-vision-override-v1"
DEFAULT_MODEL_NAME = "gpt6_astra"
DEFAULT_MODEL_PROVIDER = "openai"
BACKUP_MODEL_NAME = "claude5_opus"
BACKUP_MODEL_PROVIDER = "aws"
MODEL_PRESETS = {
    "default": (DEFAULT_MODEL_NAME, DEFAULT_MODEL_PROVIDER),
    "backup": (BACKUP_MODEL_NAME, BACKUP_MODEL_PROVIDER),
}


class ModelRoutingError(RuntimeError):
    """The response does not identify the requested provider/model."""


def resolve_model(model_name=None, model_provider=None, *, preset="default"):
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {preset!r}")
    default_name, _ = MODEL_PRESETS[preset]
    name = model_name or default_name
    known_providers = {n: p for n, p in MODEL_PRESETS.values()}
    provider = model_provider or known_providers.get(name)
    if provider is None:
        raise ValueError("model_provider is required for a custom model_name")
    return name, provider


def validate_model_identity(response, model_name, model_provider):
    metadata = response.get("metadata") if isinstance(response, dict) else None
    details = metadata.get("model_details") if isinstance(metadata, dict) else None
    if not isinstance(details, dict):
        raise ModelRoutingError("CreateAI response is missing metadata.model_details")
    actual = (details.get("inference_provider"), details.get("inference_model"))
    expected = (model_provider, model_name)
    if actual != expected:
        raise ModelRoutingError(
            f"CreateAI model mismatch: requested {expected[0]}/{expected[1]}, "
            f"received {actual[0]}/{actual[1]}"
        )
    return details
