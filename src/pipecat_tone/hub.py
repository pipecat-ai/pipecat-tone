"""The tone models on Hugging Face, ``pipecat-ai/tone-models``: a folder per model, a tag per release."""

from pathlib import Path

from ._onnx import need_extra

REPO = "pipecat-ai/tone-models"


class ModelDownloadError(RuntimeError):
    """A model's files couldn't be downloaded, or found in the local cache."""


def download(folder: str, revision: str, files: list[str]) -> Path:
    """A model's folder at a release, with these files from the local cache, downloaded the first time.

    Args:
        folder: The model's folder in the repo, such as ``tagger/17m``.
        revision: The release tag, such as ``tagger-rc1``.
        files: The files the model needs.

    Raises:
        ModelDownloadError: The files can't be had; the message says why.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ModuleNotFoundError as e:
        raise need_extra(e) from e

    where = f"{REPO}@{revision}"
    try:
        root = Path(
            snapshot_download(REPO, revision=revision, allow_patterns=[f"{folder}/{name}" for name in files])
        )
    except RepositoryNotFoundError as e:
        raise ModelDownloadError(f"can't open {REPO}: it isn't there, or HF_TOKEN can't read it") from e
    except RevisionNotFoundError as e:
        raise ModelDownloadError(f"{REPO} has no release {revision!r}") from e
    except LocalEntryNotFoundError as e:
        raise ModelDownloadError(
            f"{where} isn't in the local cache, and Hugging Face can't be reached ({e})"
        ) from e
    except Exception as e:
        raise ModelDownloadError(f"downloading {where} failed: {e}") from e
    missing = [name for name in files if not (root / folder / name).is_file()]
    if missing:
        raise ModelDownloadError(f"{where} has no {', '.join(f'{folder}/{name}' for name in missing)}")
    return root / folder
