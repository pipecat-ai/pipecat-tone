"""The reader's images and job helpers, on the shared app: ``DATA`` and ``TORCH`` from uv.lock, and the vLLM and speech
images, whose packages uv.lock can't hold, pinned here."""

import json
from pathlib import Path

import modal

from lab import ROOT
from lab.jobs.common import LAB, app, image_with, with_code
from lab.reader import storage

RESULTS = ROOT / "results" / "reader"
ENV = {"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}

DATA = LAB
TORCH = image_with(["lab", "reader"])


# The reader group's versions of the stack the speech round trip shares with it; a test keeps them equal.
READER_STACK = (
    "torch==2.14.1",
    "transformers==5.18.0",
    "safetensors==0.8.0",
    "tokenizers==0.23.2",
    "huggingface-hub==1.33.0",
    "numpy==2.2.6",
)

# The teacher that writes the voice-agent set: vLLM on NVIDIA's CUDA 13.0 devel image, since its FP8 and FlashInfer
# paths compile kernels at run time. Weights are cached on the tone-reader-hf-cache volume.
VLLM = with_code(
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.30.0", "pyyaml==6.0.3", "httpx==0.28.1", "loguru==0.7.3")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
)
hf_cache = modal.Volume.from_name("tone-reader-hf-cache", create_if_missing=True)

# The speech round trip: CUDA torch and the HF stack, Kokoro's inference package with its G2P (misaki, English:
# spaCy's small model, espeak-ng for words outside its dictionary), and scipy for resampling.
SPACY_MODEL = (
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/"
    "en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPEECH = with_code(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(READER_STACK[0], index_url="https://download.pytorch.org/whl/cu126")
    .uv_pip_install(
        *READER_STACK[1:], "peft==0.21.2", "accelerate==1.15.0", "pydantic==2.13.5", "pyyaml==6.0.3"
    )
    .uv_pip_install("kokoro==0.9.4", "misaki[en]==0.9.4", "spacy==3.8.16", "scipy==1.16.2", SPACY_MODEL)
    .uv_pip_install("loguru==0.7.3")
)


def job(image: modal.Image, **options):
    """A reader job on the shared app, with the data volume mounted beside any others."""
    volumes = {storage.MOUNT.as_posix(): storage.volume, **options.pop("volumes", {})}
    return app.function(image=image, volumes=volumes, env=ENV, **options)


def save(folder: str, name: str, obj) -> Path:
    """A local result as JSON under results/reader/<folder>: counts and scores, never line text."""
    return _write(RESULTS / folder / name, json.dumps(obj, indent=1) + "\n")


def save_yaml(folder: str, name: str, obj) -> Path:
    """A local manifest as YAML under results/reader/<folder>."""
    import yaml

    return _write(
        RESULTS / folder / name, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=140)
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
