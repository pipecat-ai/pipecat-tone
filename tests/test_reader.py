import time
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import run_test

from pipecat_tone import hub
from pipecat_tone.hub import ModelDownloadError
from pipecat_tone.reader import ToneReaderModel, ToneReaderParams, ToneReaderProcessor
from pipecat_tone.reader.model import MODEL_FILES
from pipecat_tone.tagger.readings import EMOTION_TO_TONE

ROOT = Path(__file__).resolve().parents[1]


def rc1_folder(precision: str) -> Path:
    """A folder with rc1's files for a precision: from the local cache, or the release copied here."""
    files = ["config.json", "tokenizer.json", MODEL_FILES[precision]]
    for repo, revision in ((hub.REPO, "reader-rc1"), ("pipecat-ai/tone", "rc1")):
        try:
            root = Path(
                huggingface_hub.snapshot_download(
                    repo, revision=revision, allow_patterns=files, local_files_only=True
                )
            )
        except LocalEntryNotFoundError:
            continue
        folder = root / "reader" if repo == hub.REPO else root
        if all((folder / name).is_file() for name in files):
            return folder
    if all((ROOT / "data" / "release" / "reader-rc1" / name).is_file() for name in files):
        return ROOT / "data" / "release" / "reader-rc1"
    pytest.skip(f"rc1's {precision} files aren't here")


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_reads_exactly_as_pipecat_tone(rc1, precision):
    model = ToneReaderModel(rc1_folder(precision), precision=precision, threads=1)
    assert model.emotions == rc1["emotions"]
    for turn in rc1["turns"]:
        assert model.tag(turn["text"], turn["previous"]) == turn[precision]


def test_the_tagger_folds_every_emotion_the_reader_reads(rc1):
    assert set(EMOTION_TO_TONE) == set(rc1["emotions"])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_download_fetches_the_precisions_files(monkeypatch, tmp_path, precision):
    calls = []
    monkeypatch.setattr(hub, "download", lambda *args: calls.append(args) or tmp_path)
    monkeypatch.setattr(ToneReaderModel, "__init__", lambda self, folder, **kw: calls.append((folder, kw)))
    ToneReaderModel.download(precision=precision, threads=1)
    assert calls == [
        ("reader", "reader-rc1", ["config.json", "tokenizer.json", MODEL_FILES[precision]]),
        (tmp_path, {"precision": precision, "threads": 1}),
    ]


class StubModel:
    emotions = ["happy", "neutral", "angry"]

    def __init__(self, fails: bool = False):
        self.fails = fails
        self.reads: list[tuple[str, str | None]] = []

    def tag(self, text: str, previous: str | None = None) -> dict:
        self.reads.append((text, previous))
        if self.fails:
            raise RuntimeError("bad input")
        return {
            "emotion": {"probabilities": {"happy": 0.1, "neutral": 0.3, "angry": 0.6}},
            "sarcasm": {"probability": 0.123},
        }


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def perceptions(frames) -> list:
    return [f.perception for f in frames if isinstance(f, TranscriptionFrame)]


async def test_a_turn_is_read_with_the_reply_before_it():
    model = StubModel()
    down, _ = await run_test(
        ToneReaderProcessor(model=model),
        frames_to_send=[
            LLMContextAssistantTurnFrame(text="<<calm>> Your code is <<slow: 4 8 2>>.", timestamp=""),
            transcription("What?"),
        ],
        ignore_start=False,
    )
    assert [type(f) for f in down][:2] == [StartFrame, PerceptionMetadataFrame]
    assert down[1].perception.signals[0].labels == ["happy", "neutral", "angry"]
    assert model.reads == [("What?", "Your code is 4 8 2.")]
    reading = perceptions(down)[0]
    assert reading[0].labels == [("angry", 0.6), ("neutral", 0.3)] and reading[1].value == 0.12


async def test_a_turn_that_cant_be_read_goes_on_unread():
    processor = ToneReaderProcessor(model=StubModel(fails=True))
    down, up = await run_test(processor, frames_to_send=[transcription("Hi."), transcription("Hello?")])
    assert perceptions(down) == [None, None]
    assert len([f for f in up if isinstance(f, ErrorFrame)]) == 2 and processor.is_usable


def downloads(monkeypatch, download) -> ToneReaderProcessor:
    monkeypatch.setattr(ToneReaderModel, "download", download)
    return ToneReaderProcessor(params=ToneReaderParams(revision="reader-rc9", precision="fp32", threads=2))


async def test_the_model_is_downloaded_as_params_say(monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return StubModel()

    down, _ = await run_test(downloads(monkeypatch, download), frames_to_send=[transcription("Hi.")])
    assert calls == [{"revision": "reader-rc9", "precision": "fp32", "threads": 2}]
    assert perceptions(down)[0] is not None


async def test_a_slow_download_doesnt_hold_up_the_start(monkeypatch):
    def download(**kwargs):
        time.sleep(0.5)
        return StubModel()

    down, _ = await run_test(
        downloads(monkeypatch, download), frames_to_send=[transcription("Hi.")], start_timeout=0.2
    )
    assert perceptions(down)[0] is not None


async def test_a_model_that_wont_download_is_reported_and_turns_go_on_unread(monkeypatch):
    def download(**kwargs):
        raise ModelDownloadError("no such release")

    processor = downloads(monkeypatch, download)
    down, up = await run_test(processor, frames_to_send=[transcription("Hi.")])
    assert (
        "couldn't load the tone reader: no such release"
        in [f for f in up if isinstance(f, ErrorFrame)][0].error
    )
    assert perceptions(down) == [None] and not processor.is_usable
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
