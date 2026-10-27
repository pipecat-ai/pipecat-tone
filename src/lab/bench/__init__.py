"""tone-bench's scenarios and models."""

from pathlib import Path

import yaml

from lab import TONE_BENCH

if not (TONE_BENCH / "bench" / "__init__.py").exists():
    raise ImportError(f"tone-bench not found at {TONE_BENCH}; set TONE_BENCH to its checkout")

from bench.run import SCENARIOS  # noqa: E402  (importable once the checkout is found)


def scenario_path(name: str) -> Path:
    return SCENARIOS / f"{name}.yaml"


def model_config(name: str):
    """A tone-bench LLM from its models.yaml; keys come from the environment."""
    from bench.agent import ModelConfig

    models = yaml.safe_load((TONE_BENCH / "models.yaml").read_text())["models"]
    if name not in models:
        raise ValueError(f"unknown model {name!r}; tone-bench's models.yaml has {list(models)}")
    return ModelConfig.from_yaml(name, models[name])
