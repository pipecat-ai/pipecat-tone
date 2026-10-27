"""Releases to Hugging Face, ``pipecat-ai/tone-models``: a folder per model, a tag per release, and a
MANIFEST.json of every model file's size and checksum.

    uv run python -m lab.release model reader-rc1 reader=<folder>
    uv run python -m lab.release model tagger-rc1 tagger/17m=models/ettin17-v3-bal-soup tagger/32m=models/ettin32-v3-share-soup
    uv run python -m lab.release data tagger/v3 data/corpus/v3
    uv run python -m lab.release check tagger-rc1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from lab import ROOT
from pipecat_tone.hub import REPO
from pipecat_tone.reader.model import MODEL_FILES

CARD = ROOT / "releases" / "huggingface.md"


def model_files(folder: Path) -> list[str]:
    """The files a model folder's runtime loads: a tagger's config names its ONNX file."""
    config = json.loads((folder / "config.json").read_text())
    onnx = [config["onnx"]] if "onnx" in config else list(MODEL_FILES.values())
    return ["config.json", "tokenizer.json", *onnx]


def release_model(tag: str, folders: dict[str, Path]):
    """Commit each model folder's runtime files, the card and an updated MANIFEST, then tag the commit."""
    api = HfApi()
    api.create_repo(REPO, exist_ok=True)
    manifest = _manifest(api, "main")
    files = {
        f"{path}/{name}": folder / name for path, folder in folders.items() for name in model_files(folder)
    }
    manifest["files"].update(
        {path: {"bytes": local.stat().st_size, "sha256": _sha256(local)} for path, local in files.items()}
    )
    manifest["releases"][tag.rsplit("-", 1)[0]] = tag
    operations = [CommitOperationAdd(path, str(local)) for path, local in files.items()]
    operations += [
        CommitOperationAdd("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True).encode()),
        CommitOperationAdd("README.md", str(CARD)),
    ]
    commit = api.create_commit(REPO, operations=operations, commit_message=f"Release {tag}")
    api.create_tag(REPO, tag=tag, revision=commit.oid)
    print(f"{tag}: {len(files)} files at {commit.oid[:10]}")


def release_data(path: str, folder: Path):
    """Upload a training data folder to ``data/<path>``: what training reads, without failed records, and without the
    reviewer's model id, which the cards leave as another vendor's model."""
    with tempfile.TemporaryDirectory() as staging:
        staged = Path(staging)
        for p in folder.glob("*.json"):
            call = json.loads(p.read_text())
            if p.name == "stats.json" or "error" in call:
                continue
            (call.get("reviewer") or {}).pop("model", None)
            (staged / p.name).write_text(json.dumps(call, indent=1))  # as lab.jobs.corpus writes calls
        for p in [*folder.glob("*.jsonl"), folder / "stats.json"]:
            (staged / p.name).symlink_to(p.resolve())
        HfApi().upload_folder(
            repo_id=REPO, folder_path=staged, path_in_repo=f"data/{path}", commit_message=f"Data: {path}"
        )
        print(f"data/{path}: {sum(1 for _ in staged.iterdir())} files")


def check(tag: str):
    """Download a release into a fresh cache through the runtime, and check each file against the MANIFEST."""
    model = tag.rsplit("-", 1)[0]
    with tempfile.TemporaryDirectory() as cache:
        manifest = json.loads(
            Path(hf_hub_download(REPO, "MANIFEST.json", revision=tag, cache_dir=cache)).read_text()
        )
        load = (
            f"from pipecat_tone.reader import ToneReaderModel as M; [M.download(revision={tag!r}, precision=p) for p in ('fp16', 'fp32')]"
            if model == "reader"
            else f"from pipecat_tone.tagger import ToneTaggerModel as M; [M.download(revision={tag!r}, size=s) for s in ('17m', '32m')]"
        )
        subprocess.run([sys.executable, "-c", load], env={**os.environ, "HF_HUB_CACHE": cache}, check=True)
        snapshot = next(Path(cache).glob("models--*/snapshots/*"))
        checked = 0
        for path, expected in manifest["files"].items():
            if path.startswith(f"{model}/"):
                assert _sha256(snapshot / path) == expected["sha256"], (
                    f"{tag}: {path} differs from the MANIFEST"
                )
                checked += 1
    print(f"{tag}: downloaded and loaded through the runtime; {checked} files match the MANIFEST")


def _manifest(api: HfApi, revision: str) -> dict:
    try:
        return json.loads(Path(hf_hub_download(REPO, "MANIFEST.json", revision=revision)).read_text())
    except Exception:  # the first release
        return {"name": "Pipecat tone models", "releases": {}, "files": {}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="release model folders at a tag")
    model.add_argument("tag")
    model.add_argument("folders", nargs="+", metavar="PATH=FOLDER")
    data = commands.add_parser("data", help="upload a training data folder")
    data.add_argument("path")
    data.add_argument("folder", type=Path)
    checking = commands.add_parser("check", help="check a release in a fresh cache")
    checking.add_argument("tag")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    if args.command == "model":
        release_model(
            args.tag, {path: Path(folder) for path, folder in (f.split("=", 1) for f in args.folders)}
        )
    elif args.command == "data":
        release_data(args.path, args.folder)
    else:
        check(args.tag)
