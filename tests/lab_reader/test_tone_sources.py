"""The Tone-era sources: their pins, the Tone pipeline's transforms, and lines read back by source_ref from made-up
files. No network. Every text here is made up for the test."""

import hashlib
import re
from pathlib import Path

import pytest

from lab.reader import tone_sources
from lab.reader.tone_sources import Dialogues, Upstream


def test_every_file_is_pinned():
    for s in tone_sources.SOURCES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", s["revision"]) and s["licence"] and s["credit"]
        for want in s["files"].values():
            assert want["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", want["sha256"])
    assert tone_sources.file_rel("gutenberg", "dev.txt") == (
        "hf/willwade/Gutenberg-dialog-en@f3a0a1df18275cd9f61f7d161ad6eed3ce35f88c/dev.txt"
    )
    assert set(tone_sources.CIVIL_FILES.values()) == set(tone_sources.SOURCES["civil_comments"]["files"])


def test_the_tone_pipelines_transforms():
    assert tone_sources.detok("i do n't know , do you ? yes .") == "I don't know, do you? Yes."
    assert tone_sources.ws("  two\t words\n here ") == "two words here"
    assert tone_sources.noisy_base('Well, "Mara" -- is it OK?  Yes!') == "well mara -- is it ok yes"
    assert tone_sources.civil_bands({"threat": 0.3}) == "hostile"
    assert tone_sources.civil_bands({"insult": 0.5, "toxicity": None}) == "hostile"
    assert tone_sources.civil_bands({"toxicity": 0.6}) == "hostile"
    assert tone_sources.civil_bands({"toxicity": 0.2}) == "heated"
    assert tone_sources.civil_bands({"toxicity": 0.19, "insult": 0.49, "threat": 0.29}) == "civil"
    assert len(tone_sources.GOEMOTIONS_LABELS) == 28 and tone_sources.GOEMOTIONS_LABELS[17] == "joy"


@pytest.mark.parametrize(
    "text",
    [
        "a\nb\n\nc\n",
        "\n\nfirst\n\n\n\nsecond\nthird\n\n  \n\nlast",
        "one\n\n\nlead blank\n \nkept\n\n",
        "",
        "only",
    ],
)
def test_dialogues_by_index_are_the_pipelines_list(text):
    found = Dialogues(text)
    want = tone_sources.gutenberg_dialogues(text)
    assert len(found) == len(want) and [found[i] for i in range(len(found))] == want


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def slice(self, i, n):
        return FakeTable(self.rows[i : i + n])

    def to_pylist(self):
        return self.rows


def write(up: Upstream, source: str, name: str, text: str) -> None:
    path = up.path(source, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_lines_read_back_by_source_ref(tmp_path):
    up = Upstream(tmp_path)
    up._files["goemotions"] = {"e1": {"id": "e1", "text": "  so [NAME]  said\thi ", "labels": [17, 2]}}
    civil = tone_sources.CIVIL_FILES
    up._files[civil[""]] = FakeTable([{"text": "first"}, {"text": " you  again ", "toxicity": 0.3}])
    up._files[civil["train1"]] = FakeTable([{"text": "x", "threat": 0.9}])
    write(
        up, "gutenberg", "dev.txt", "hello there .\nyou are late , are n't you ?\n\nsecond one\nand you ?\n"
    )
    write(up, "gutenberg", "train.txt", "a .\nb , you see ?\n")
    assert up.text("goemotions", "e1") == "so [NAME] said hi" and up.label("goemotions", "e1") == "joy"
    assert up.text("civil_comments", "row1") == "you again" and up.label("civil_comments", "row1") == "heated"
    assert up.label("civil_comments", "train1:row0") == "hostile"
    assert up.text("gutenberg", "dev:0:1") == "You are late, aren't you?"
    assert up.previous("gutenberg", "dev:0:1") == "Hello there."
    assert up.text("gutenberg", "dev:1:1") == "And you?" and up.label("gutenberg", "dev:1:1") is None
    assert up.text("gutenberg", "train:0:1") == "B, you see?"
    with pytest.raises(KeyError):
        up.text("civil_comments", "rowx")


def test_download_keeps_a_right_file_and_checks_a_new_one(tmp_path, monkeypatch):
    body = b"made-up source\n"
    want = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    monkeypatch.setattr(
        tone_sources,
        "SOURCES",
        {"x": {"repo": "a/b", "revision": "r", "files": {"f.txt": want}, "licence": "-", "credit": "-"}},
    )
    fetched = []

    def fake_fetch(source, name, cache):
        fetched.append(name)
        p = Path(cache) / "blob"
        p.write_bytes(body)
        return p

    monkeypatch.setattr(tone_sources, "fetch_file", fake_fetch)
    assert tone_sources.download(tmp_path) == {"hf/a/b@r/f.txt": "downloaded"}
    assert (tmp_path / "hf/a/b@r/f.txt").read_bytes() == body
    assert tone_sources.download(tmp_path) == {"hf/a/b@r/f.txt": "kept"} and fetched == ["f.txt"]
    assert [p.name for p in tmp_path.iterdir()] == ["hf"]


def test_a_file_that_isnt_the_pinned_one_is_refused(tmp_path, monkeypatch):
    import huggingface_hub

    wrong = tmp_path / "wrong.parquet"
    wrong.write_bytes(b"not it")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(wrong))
    with pytest.raises(RuntimeError, match="pinned"):
        tone_sources.fetch_file("goemotions", "simplified/train-00000-of-00001.parquet", tmp_path)
