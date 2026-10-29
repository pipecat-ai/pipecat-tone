#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The tone reader's model: how a turn of text was meant, read on the CPU in a few milliseconds.

It reads one turn, with the turn before it, and returns the speaker's emotion, whether the turn is
sarcastic, the move it makes and whether it's only a listening sound, each with a calibrated
confidence. A port of Pipecat Tone's runtime from Pipecat; releases are on Hugging Face as
``reader`` in ``pipecat-ai/tone-models``.
"""

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .. import hub
from .._onnx import session, tokenizer

READER_REVISION = "reader-rc2"
MODEL_FILES = {"fp16": "model_fp16.onnx", "fp32": "model.opt.onnx"}


class ToneReaderModel:
    """The tone reader's model, on the CPU. One model can serve every call."""

    def __init__(
        self, model_dir: str | Path, *, precision: Literal["fp16", "fp32"] = "fp16", threads: int = 4
    ):
        """Load the model from a folder.

        Args:
            model_dir: A folder with the model's ``config.json``, ``tokenizer.json`` and ONNX file.
            precision: ``"fp16"`` for the weights in 16 bits, which give the same tags at half the size,
                or ``"fp32"``.
            threads: The CPU threads for one read.
        """
        folder = Path(model_dir)
        self.config = json.loads((folder / "config.json").read_text())
        self._tokenizer = tokenizer(folder / "tokenizer.json")
        self._tokenizer.enable_truncation(self.config["max_len"])
        self._tokenizer.no_padding()
        self._session = session(folder / MODEL_FILES[precision], threads)
        calibration = self.config.get("calibration") or {}
        self._emotion_temperature = float(calibration.get("emotion_temperature", 1.0))
        self._move_temperature = float(calibration.get("move_temperature", 1.0))
        self._sarcasm_curve = calibration.get("sarcasm")
        self._sarcasm_threshold = float(self.config.get("sarcasm_tau_raw", 0.5))

    @classmethod
    def download(
        cls, *, revision: str = READER_REVISION, precision: Literal["fp16", "fp32"] = "fp16", threads: int = 4
    ) -> "ToneReaderModel":
        """Load a release from Hugging Face, downloading its files the first time.

        Raises:
            ModelDownloadError: The release can't be downloaded or found in the cache.
        """
        files = ["config.json", "tokenizer.json", MODEL_FILES[precision]]
        return cls(hub.download("reader", revision, files), precision=precision, threads=threads)

    @property
    def emotions(self) -> list[str]:
        """The emotions the model reads."""
        return self.config["emotions"]

    def tag(self, text: str, previous: str | None = None) -> dict[str, Any]:
        """Read one turn, with the turn before it if there is one.

        Returns:
            The model's tags, in its own form: ``emotion``, ``move``, ``sarcasm`` and ``backchannel``.
        """
        return self.tags_from(self.outputs(text, previous))

    def outputs(self, text: str, previous: str | None = None) -> dict[str, np.ndarray]:
        """One turn's raw outputs by name, before calibration."""
        encoding = self._tokenizer.encode(render(text, previous))
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        results = self._session.run(None, inputs)
        return dict(zip(self.config["outputs"], (cast(np.ndarray, r)[0] for r in results)))

    def tags_from(self, outputs: dict[str, np.ndarray]) -> dict[str, Any]:
        """The tags in one turn's raw outputs, calibrated."""
        emotions = softmax(outputs["emotion"], self._emotion_temperature)
        moves = softmax(outputs["move"], self._move_temperature)
        emotion = self.config["emotions"][int(emotions.argmax())]
        intensity = self.config["intensities"][int(softmax(outputs["emotion_intensity"]).argmax())]
        raw_sarcasm = float(softmax(outputs["sarcasm"])[1])
        backchannel = float(softmax(outputs["backchannel"])[1])
        return {
            "emotion": {
                "label": emotion,
                "intensity": "low" if emotion == "neutral" else intensity,
                "confidence": round(float(emotions.max()), 4),
                "probabilities": _probabilities(self.config["emotions"], emotions),
            },
            "move": {
                "label": self.config["moves"][int(moves.argmax())],
                "confidence": round(float(moves.max()), 4),
                "probabilities": _probabilities(self.config["moves"], moves),
            },
            "sarcasm": {
                "sarcastic": raw_sarcasm >= self._sarcasm_threshold,
                "probability": round(platt(raw_sarcasm, self._sarcasm_curve), 4),
            },
            "backchannel": {"is_backchannel": backchannel >= 0.5, "probability": round(backchannel, 4)},
        }


def render(text: str, previous: str | None = None) -> str:
    """The model's input: the turn, then the turn before it, so truncation can only cut the turn before."""
    prev = (previous or "").strip()
    return f'"{text.strip()}" Before: "{prev}"' if prev else f'"{text.strip()}"'


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Over the last axis."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _probabilities(labels: list[str], probabilities: np.ndarray) -> dict[str, float]:
    return {label: round(float(p), 4) for label, p in zip(labels, probabilities)}


def platt(p: float, curve: dict | None) -> float:
    """The calibrated chance, ``sigmoid(a * logit(p) + b)``, or ``p`` without a curve."""
    if not curve:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = curve["a"] * math.log(p / (1 - p)) + curve["b"]
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
