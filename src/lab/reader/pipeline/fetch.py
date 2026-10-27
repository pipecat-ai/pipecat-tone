"""After `pull`, the reader's data made whole on the volume from the original sources (``lab.reader.rebuild``): the
open sets, which aren't published, from their archives at their pins, then the files published as stand-ins, in
order. A file is written only if its sha256 is the manifest's. Every file the manifest lists is then checked, those
`pull` copied too, since it keeps a file already there at the same size.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.pull
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.fetch

Sources not on the volume yet are downloaded: the Hugging Face files to upstream/, the open sets' archives to
runs/tonetext/open-raw/ (``open_ingest``). `--prefix <folder/>` reads and writes the published files under a folder,
as `pull --prefix` left them; `--no-download` downloads nothing; `--force` downloads every source again and rebuilds
files already there and right.
"""

import hashlib
import json
import tempfile
import time
from pathlib import Path

from lab import ROOT
from lab.reader import open_sources, rebuild, sources, storage, tone_sources
from lab.reader.app import DATA, app, job, save
from lab.reader.pipeline import open_ingest

SUMMARY_KEY = "rebuild/fetched.json"  # the last fetch's counts and hashes, beside the manifest


def _manifest(prefix: str) -> dict:
    return json.loads(storage.get(prefix + rebuild.MANIFEST_KEY))


def _sha(key: str) -> str | None:
    """A file's sha256 on the volume, None when it isn't there."""
    if not storage.exists(key):
        return None
    digest = hashlib.sha256()
    with storage.path(key).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(key: str, body: bytes, want: str) -> str:
    if rebuild.sha256(body) != want:
        return "differs: not written"
    storage.put(key, body)
    return "written"


@job(DATA, cpu=2.0, memory=4096, timeout=2 * 60 * 60)
def download_upstream(force: bool = False) -> dict:
    """The Hugging Face sources at their pins onto the volume, each checked against its sha256."""
    storage.reload()
    out = {}
    for source, s in tone_sources.SOURCES.items():
        for name, want in s["files"].items():
            key = sources.upstream_key(tone_sources.file_rel(source, name))
            if not force and _sha(key) == want["sha256"]:
                out[key] = "kept"
                continue
            with tempfile.TemporaryDirectory() as cache:
                storage.upload(tone_sources.fetch_file(source, name, Path(cache)), key)
            out[key] = "downloaded"
    return out


@job(DATA, cpu=2.0, memory=32768, timeout=60 * 60)
def rebuild_open(source: str, prefix: str = "", force: bool = False) -> dict:
    """One open set from its archive on the volume: the files the parser reads checked against the manifest, then the
    lines written if their sha256 is the manifest's."""
    storage.reload()
    t0 = time.perf_counter()
    pin = _manifest(prefix)["open"][source]
    key = prefix + pin["key"]
    out = {"source": source, "key": key, "sha256": pin["sha256"]}
    if not force and _sha(key) == pin["sha256"]:
        return {**out, "result": "kept"}
    members = open_ingest.load_raw(source, open_sources.FILES[source])
    differ = sorted(
        n for n, s in pin["members"].items() if n not in members or rebuild.sha256(members[n]) != s
    )
    extra = sorted(set(members) - set(pin["members"]))
    out.update(archive_files_differ=len(differ), archive_files_extra=len(extra), first=(differ + extra)[:10])
    body = open_ingest.build_lines(source, members)
    out.update(result=_write(key, body, pin["sha256"]), built_sha256=rebuild.sha256(body), bytes=len(body))
    out["seconds"] = round(time.perf_counter() - t0)
    return out


@job(DATA, cpu=4.0, memory=32768, timeout=3 * 60 * 60)
def rebuild_standins(prefix: str = "", force: bool = False) -> dict:
    """The stand-ins' files in the manifest's order, each from its stand-in, the sources and the files rebuilt
    before it, written if its sha256 is the manifest's."""
    storage.reload()
    manifest = _manifest(prefix)
    ctx = rebuild.Context(
        tone_sources.Upstream(storage.MOUNT / sources.UPSTREAM), read=lambda k: storage.get(prefix + k)
    )
    out = {}
    for key in manifest["order"]:
        t0 = time.perf_counter()
        entry = manifest["standins"][key]
        if not force and _sha(prefix + key) == entry["sha256"]:
            out[key] = {"result": "kept"}
            continue
        try:
            body = rebuild.rebuild_file(storage.get(prefix + sources.standin_key(key)), entry["kind"], ctx)
        except (rebuild.Mismatch, KeyError, FileNotFoundError) as e:
            out[key] = {"result": f"failed: {type(e).__name__}: {e}"}
            continue
        out[key] = {
            "result": _write(prefix + key, body, entry["sha256"]),
            "built_sha256": rebuild.sha256(body),
            "bytes": len(body),
            "seconds": round(time.perf_counter() - t0),
        }
    return out


@job(DATA, cpu=2.0, memory=4096, timeout=60 * 60)
def check_keys(prefix: str = "") -> dict:
    """Every file the manifest lists, as the volume holds it, against its sha256."""
    storage.reload()
    manifest = _manifest(prefix)
    want = {
        **{k: v["sha256"] for k, v in manifest["kept"].items()},
        **{v["key"]: v["sha256"] for v in manifest["open"].values()},
        **{k: v["sha256"] for k, v in manifest["standins"].items()},
        **{v["standin"]: v["standin_sha256"] for v in manifest["standins"].values()},
    }
    got = {k: _sha(prefix + k) for k in want}
    return {
        "files": len(want),
        "right": sum(got[k] == want[k] for k in want),
        "missing": sorted(k for k in want if got[k] is None),
        "differ": sorted(k for k in want if got[k] not in (None, want[k])),
    }


@job(DATA, cpu=1.0, memory=2048, timeout=6 * 60 * 60)
def run(prefix: str = "", download: bool = True, force: bool = False) -> dict:
    """The sources downloaded, the open sets rebuilt one container each, then the stand-ins' files, then every file
    checked. It runs on Modal, so with --detach it outlives the client; its counts and hashes are kept on the volume
    too, at SUMMARY_KEY under the prefix."""
    storage.reload()
    out: dict = {"prefix": prefix}
    if download:
        out["upstream"] = download_upstream.remote(force)
        storage.reload()
        units = [
            u
            for u in open_sources.UNITS.values()
            if force or not all(storage.exists(sources.open_raw_key(s, "download.json")) for s in u)
        ]
        out["downloads"] = {}
        for res in open_ingest.download_unit.map(units):
            info = res["info"]
            out["downloads"][info["sources"][0]] = {"archive": info["archive"], "seconds": info["seconds"]}
    out["open"] = {r["source"]: r for r in rebuild_open.starmap([(s, prefix, force) for s in rebuild.OPEN])}
    out["standins"] = rebuild_standins.remote(prefix, force)
    out["check"] = check_keys.remote(prefix)
    storage.put(prefix + SUMMARY_KEY, json.dumps(out, indent=1).encode())
    return out


@app.local_entrypoint()
def main(prefix: str = "", download: bool = True, force: bool = False):
    """``run``, with its counts and hashes saved to results/reader/fetch/fetch.json."""
    out = run.remote(prefix, download, force)
    path = save("fetch", "fetch.json", out)
    print(json.dumps({k: out[k] for k in ("open", "standins", "check")}, indent=1))
    print("wrote", path.relative_to(ROOT))
