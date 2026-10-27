"""What every job shares: the app, its image, the keys and the volumes.

The image is built from uv.lock, so a job runs this checkout's Pipecat, with this repo's code and,
when it's checked out, tone-bench's on top. Keys reach a job through the ``tone-tagger-keys`` secret, made from .env.
"""

from pathlib import Path

import modal

from lab import ROOT, TONE_BENCH

REMOTE_SRC = "/root/src"
REMOTE_TONE_BENCH = "/root/tone-bench"
RESULTS_DIR = "/results"
CORPUS_DIR = "/corpus"
MODELS_DIR = "/models"
REMOTE_RUNS = Path(RESULTS_DIR) / "runs"


def image_with(groups: list[str]) -> modal.Image:
    """The job image with these dependency groups and the runtime's model extra, and the code on top: tone-bench's
    too, when its checkout is here."""
    bench = (TONE_BENCH / "bench").is_dir()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_sync(str(ROOT), groups=groups, extras=["model"], extra_options="--no-default-groups")
    )
    if bench:
        image = image.env({"TONE_BENCH": REMOTE_TONE_BENCH})
    image = with_code(image)
    if bench:
        image = image.add_local_dir(
            TONE_BENCH, REMOTE_TONE_BENCH, ignore=[".env", ".git", ".venv", "results", "**/__pycache__"]
        )
    return image


def with_code(image: modal.Image) -> modal.Image:
    """An image with this repo's code on top, importable. Local files come last: nothing builds after them."""
    return image.env({"PYTHONPATH": REMOTE_SRC}).add_local_dir(
        ROOT / "src", REMOTE_SRC, ignore=["**/__pycache__"]
    )


LAB = image_with(["lab"])
app = modal.App("tone-tagger", image=LAB, secrets=[modal.Secret.from_name("tone-tagger-keys")])

# Every run a job saves, in tone-bench's layout; corpus batches; trained models.
results = modal.Volume.from_name("tone-tagger-results", create_if_missing=True)
corpus = modal.Volume.from_name("tone-tagger-corpus", create_if_missing=True)
models = modal.Volume.from_name("tone-tagger-models", create_if_missing=True)


def pack(folder: Path) -> dict[str, bytes]:
    """A run folder's files, to bring back from a job."""
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def unpack(files: dict[str, bytes], folder: Path) -> Path:
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)
    return folder
