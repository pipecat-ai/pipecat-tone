"""The data volume as the stages use it: mounted in a job, read through Modal's API from an entrypoint, and a missing
file always an error."""

import modal
import pytest

from lab.reader import storage


class Entry:
    def __init__(self, path: str, kind=modal.volume.FileEntryType.FILE):
        self.path, self.type = path, kind


class FakeVolume:
    """The volume's API over a dict of files, as an entrypoint sees it."""

    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = files or {}
        self.commits = self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1

    def read_file(self, key: str):
        if key not in self.files:
            raise FileNotFoundError(key)
        yield self.files[key]

    def listdir(self, path: str, recursive: bool = False):
        if path in self.files:
            return [Entry(path)]
        under = [k for k in self.files if k.startswith(path.rstrip("/") + "/")]
        if not under:
            raise modal.exception.NotFoundError(path)
        return [Entry(k) for k in under] + [Entry(path + "/sub", modal.volume.FileEntryType.DIRECTORY)]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    volume = FakeVolume()
    monkeypatch.setattr(storage, "MOUNT", tmp_path)
    monkeypatch.setattr(storage, "volume", volume)
    return volume


@pytest.fixture
def local(tmp_path, monkeypatch):
    volume = FakeVolume({"runs/a/lines.jsonl": b'{"x": 1}\n\n{"x": 2}\n', "runs/a/b/summary.json": b"{}"})
    monkeypatch.setattr(storage, "MOUNT", tmp_path / "not-mounted")
    monkeypatch.setattr(storage, "volume", volume)
    return volume


def test_a_job_writes_each_file_whole_and_commits_it(mounted):
    storage.put_jsonl("runs/x/lines.jsonl", ({"i": i} for i in range(3)))
    storage.put_json("runs/x/summary.json", {"at": storage.MOUNT})
    assert mounted.commits == 2
    assert storage.jsonl("runs/x/lines.jsonl") == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert storage.get("runs/x/summary.json") == b'{"at": "' + str(storage.MOUNT).encode() + b'"}'
    assert storage.exists("runs/x/lines.jsonl") and not storage.exists("runs/x/missing.jsonl")
    assert (
        storage.keys("runs/x/") == ["runs/x/lines.jsonl", "runs/x/summary.json"]
        and storage.keys("runs/none") == []
    )
    assert not list(storage.MOUNT.rglob("*.part"))
    (storage.MOUNT / "runs/x/part-0001.jsonl.part").write_text('{"i": ')  # a write cut short
    assert storage.keys("runs/x") == ["runs/x/lines.jsonl", "runs/x/summary.json"]
    storage.reload()
    assert mounted.reloads == 1


@pytest.mark.parametrize("where", ["mounted", "local"])
def test_a_missing_file_is_an_error(where, request):
    request.getfixturevalue(where)
    with pytest.raises(FileNotFoundError, match="runs/x/missing.jsonl"):
        storage.jsonl("runs/x/missing.jsonl")


def test_a_job_copies_files_on_the_volume_and_fetches_them(mounted, tmp_path):
    archive = tmp_path / "download.tar.gz"
    archive.write_bytes(b"archive")
    storage.upload(archive, "runs/raw/a/archive.tar.gz")
    storage.copy("runs/raw/a/archive.tar.gz", "runs/raw/b/archive.tar.gz")
    assert storage.path("runs/raw/b/archive.tar.gz").read_bytes() == b"archive"
    folder = tmp_path / "fetched"
    folder.mkdir()
    (folder / "kept.txt").write_text("already here")
    storage.put("runs/raw/a/kept.txt", b"on the volume")
    storage.fetch("runs/raw/a", ("archive.tar.gz", "kept.txt"), folder)
    assert (folder / "archive.tar.gz").read_bytes() == b"archive" and (
        folder / "kept.txt"
    ).read_text() == "already here"
    with pytest.raises(FileNotFoundError):
        storage.path("runs/raw/c/archive.tar.gz")


def test_an_entrypoint_reads_through_the_api_and_never_writes(local):
    assert storage.jsonl("runs/a/lines.jsonl") == [{"x": 1}, {"x": 2}]
    assert (
        storage.exists("runs/a/b/summary.json")
        and not storage.exists("runs/a")
        and not storage.exists("runs/zz")
    )
    assert (
        storage.keys("runs/a/") == ["runs/a/b/summary.json", "runs/a/lines.jsonl"]
        and storage.keys("runs/zz") == []
    )
    with pytest.raises(FileNotFoundError, match="runs/a/none.json"):
        storage.get("runs/a/none.json")
    with pytest.raises(RuntimeError, match="written by its jobs"):
        storage.put("runs/a/new.json", b"{}")
