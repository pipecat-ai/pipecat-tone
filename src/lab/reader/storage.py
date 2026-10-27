"""The reader's data: the ``tone-reader-data`` Modal volume, every file at its key. A job mounts it and commits each
write at once; a local entrypoint reads it through Modal's API. A missing file is an error."""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import modal

NAME = "tone-reader-data"
MOUNT = Path("/reader-data")
PART = ".part"  # a file being written, renamed into place once whole
volume = modal.Volume.from_name(NAME, create_if_missing=True)

# Commits reload the volume, which fails while a file on it is open: reads and writes take turns.
_lock = threading.RLock()


def mounted() -> bool:
    return MOUNT.is_dir()


def path(key: str) -> Path:
    """A file on the mounted volume, to read in place."""
    p = MOUNT / key
    if not p.is_file():
        raise FileNotFoundError(f"{key} is not on {NAME}")
    return p


def reload() -> None:
    """The volume as last committed, at the start of a job."""
    with _lock:
        volume.reload()


def get(key: str) -> bytes:
    if mounted():
        with _lock:
            return path(key).read_bytes()
    try:
        return b"".join(volume.read_file(key))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{key} is not on {NAME}") from e


def exists(key: str) -> bool:
    if mounted():
        return (MOUNT / key).is_file()
    try:
        return any(e.path.strip("/") == key for e in volume.listdir(key))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False


def keys(prefix: str) -> list[str]:
    """Every file under a folder, sorted, without the half-written ones."""
    folder = prefix.strip("/")
    if mounted():
        root = MOUNT / folder
        found = [str(p.relative_to(MOUNT)) for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    else:
        try:
            entries = volume.listdir(folder, recursive=True)
        except (FileNotFoundError, modal.exception.NotFoundError):
            return []
        found = [e.path.strip("/") for e in entries if e.type == modal.volume.FileEntryType.FILE]
    return sorted(k for k in found if not k.endswith(PART))


def wait_for(keys: list[str], hours: float) -> list[dict]:
    """Each key's JSON, waiting up to `hours` in all for those not written yet, such as a spawned job's result."""
    start, out = time.time(), []
    for key in keys:
        while not exists(key):
            if time.time() - start > hours * 3600:
                raise SystemExit(f"{key} isn't there after {hours} h")
            time.sleep(60)
        out.append(json.loads(get(key)))
    return out


def put(key: str, body: bytes) -> None:
    _write(key, lambda dest: dest.write_bytes(body))


def upload(local: Path, key: str) -> None:
    """A local file, such as a downloaded archive, copied to the volume."""
    _write(key, lambda dest: shutil.copyfile(local, dest))


def copy(src: str, dst: str) -> None:
    upload(path(src), dst)


def _write(key: str, write) -> None:
    if not mounted():
        raise RuntimeError(f"{key}: the reader's data is written by its jobs, which mount {NAME}")
    dest = MOUNT / key
    with _lock:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + PART)
        write(part)
        part.replace(dest)
        volume.commit()


def jsonl(key: str) -> list[dict]:
    return [json.loads(x) for x in get(key).decode().splitlines() if x.strip()]


def put_jsonl(key: str, rows) -> None:
    put(key, "".join(json.dumps(r) + "\n" for r in rows).encode())


def put_json(key: str, obj) -> None:
    put(key, json.dumps(obj, default=str).encode())


def fetch(prefix: str, names, folder: Path | None = None) -> Path:
    """Files of a folder on the volume in a local folder, a new temporary one by default; files already there are
    kept."""
    folder = folder or Path(tempfile.mkdtemp())
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (folder / name).exists():
            (folder / name).write_bytes(get(f"{prefix}/{name}"))
    return folder
