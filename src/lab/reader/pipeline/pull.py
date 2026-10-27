"""Fill the data volume from ``data/reader/`` in ``pipecat-ai/tone-models``, each file at its key (under `prefix`, if
given); files already there at the same size are kept. ``fetch`` then rebuilds the files published as stand-ins.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.pull [--revision <commit>] [--prefix <folder/>]
"""

import os
import tempfile
from pathlib import Path

from lab.reader import storage
from lab.reader.app import DATA, app, job
from pipecat_tone.hub import REPO

PREFIX = "data/reader/"


@job(DATA, timeout=3 * 60 * 60, cpu=4.0, memory=8192)
def pull(revision: str = "main", prefix: str = "") -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    storage.reload()
    api = HfApi()
    commit = api.repo_info(REPO, revision=revision).sha
    files = [
        f
        for f in api.list_repo_tree(REPO, path_in_repo=PREFIX, recursive=True, revision=commit)
        if hasattr(f, "size")
    ]
    fetched = 0
    with tempfile.TemporaryDirectory() as cache:
        for f in files:
            key = prefix + f.path.removeprefix(PREFIX)
            if storage.exists(key) and storage.path(key).stat().st_size == f.size:
                continue
            local = Path(hf_hub_download(REPO, f.path, revision=commit, cache_dir=cache))
            storage.upload(local, key)
            os.remove(local.resolve())
            fetched += 1
    gb = round(sum(f.size for f in files) / 1e9, 2)
    return {"commit": commit, "prefix": prefix, "files": len(files), "fetched": fetched, "gb": gb}


@app.local_entrypoint()
def main(revision: str = "main", prefix: str = ""):
    print(pull.remote(revision, prefix))
