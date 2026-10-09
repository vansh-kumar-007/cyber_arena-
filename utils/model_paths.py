"""Resolve the model/checkpoint directory consistently across serving and training.

Set CYBERARENA_MODEL_DIR to a directory on durable storage when the deployment
supports a persistent mount. The legacy project-root ./models directory remains
the default for existing environments.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_model_dir(
    project_root: str | Path,
    explicit_override: str | Path | None = None,
) -> Path:
    """Return an absolute model directory, preferring explicit then env overrides."""
    if explicit_override is not None:
        override = str(explicit_override).strip()
        if not override:
            raise ValueError("explicit model directory override cannot be blank")
        configured = Path(override)
    else:
        environment_override = os.environ.get("CYBERARENA_MODEL_DIR", "").strip()
        configured = (
            Path(environment_override)
            if environment_override
            else Path(project_root) / "models"
        )
    return configured.expanduser().resolve()
