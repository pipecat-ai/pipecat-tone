from pathlib import Path

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

from pipecat_tone import hub
from pipecat_tone.tagger import ToneTaggerModel
from pipecat_tone.tagger.model import TAGGER_REVISION

MODEL = Path(__file__).resolve().parents[1] / "models" / "current"
RESPONSE = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))


def fails_with(monkeypatch, error):
    def snapshot_download(*args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)


@pytest.mark.parametrize(
    "error,says",
    [
        (RepositoryNotFoundError("x", response=RESPONSE), "HF_TOKEN can't read it"),
        (RevisionNotFoundError("x", response=RESPONSE), "no release 'tagger-rc9'"),
        (LocalEntryNotFoundError("offline"), r"isn't in the local cache.*\(offline\)"),
        (OSError("size mismatch"), "downloading .* failed: size mismatch"),
    ],
)
def test_download_errors_say_why(monkeypatch, error, says):
    fails_with(monkeypatch, error)
    with pytest.raises(hub.ModelDownloadError, match=says):
        hub.download("tagger/17m", "tagger-rc9", ["config.json"])


def test_a_release_without_the_files(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path))
    with pytest.raises(hub.ModelDownloadError, match="has no tagger/64m/config.json"):
        hub.download("tagger/64m", "tagger-rc1", ["config.json"])


@pytest.mark.skipif(not MODEL.exists(), reason="no local model")
def test_download_fetches_only_the_files_the_model_needs(monkeypatch, tmp_path):
    (tmp_path / "tagger").mkdir()
    (tmp_path / "tagger" / "17m").symlink_to(MODEL.resolve())
    calls = []

    def snapshot_download(repo, *, revision, allow_patterns):
        calls.append((repo, revision, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    onnx = ToneTaggerModel.download().config["onnx"]
    assert calls == [
        (hub.REPO, TAGGER_REVISION, ["tagger/17m/config.json"]),
        (
            hub.REPO,
            TAGGER_REVISION,
            ["tagger/17m/config.json", "tagger/17m/tokenizer.json", f"tagger/17m/{onnx}"],
        ),
    ]
