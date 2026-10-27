#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The tone reader in a Pipecat pipeline::

    transport input -> STT -> ToneReaderProcessor -> user aggregator -> LLM -> ...

Each final transcription is read, with the bot's reply before it, and the reading is set as its
``perception``: its two likeliest emotions with their confidences, and the chance it's sarcastic. Turn on
expressive conversation (``PipelineParams(enable_expressive_conversation=True)``) for the LLM to read it.
"""

import asyncio
from functools import partial
from typing import Literal

from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pydantic import BaseModel, ConfigDict, PositiveInt

from .._shared import load_at_start, shown_words
from .model import READER_REVISION, ToneReaderModel

_EMOTION = (
    "The feeling the words show. Read from the words alone, so neutral means they show no "
    "strong feeling, not that the user feels none: go by what they say."
)
_SARCASM = (
    "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: "
    "a sarcastic turn can read happy."
)


class ToneReaderParams(BaseModel):
    """How the tone reader's model is downloaded and run.

    Parameters:
        revision: The model's release.
        precision: ``"fp16"`` for the weights in 16 bits, which give the same tags at half the size, or ``"fp32"``.
        threads: The CPU threads for one read.
    """

    model_config = ConfigDict(extra="forbid")

    revision: str = READER_REVISION
    precision: Literal["fp16", "fp32"] = "fp16"
    threads: PositiveInt = 4


class ToneReaderProcessor(FrameProcessor):
    """Reads how each user turn was meant; goes between the STT and the user aggregator.

    Example::

        pipeline = Pipeline([transport.input(), stt, ToneReaderProcessor(), user_aggregator, llm, ...])
    """

    def __init__(
        self, *, model: ToneReaderModel | None = None, params: ToneReaderParams | None = None, **kwargs
    ):
        """Initialize the processor.

        Args:
            model: A loaded model, which calls can share, or None to download one at start.
            params: How the model is downloaded and run.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(**kwargs)
        self._model = model
        self._params = params or ToneReaderParams()
        self._previous: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip() and self._model:
            frame.perception = await self._read(frame.text)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._previous = " ".join(shown_words(frame.text).split())
        await self.push_frame(frame, direction)
        # After the start goes on, so a first download can't hold up the pipeline's start; frames after it
        # wait for the model.
        if isinstance(frame, StartFrame):
            await self._start()

    async def _start(self):
        p = self._params
        download = partial(
            ToneReaderModel.download, revision=p.revision, precision=p.precision, threads=p.threads
        )
        self._model = self._model or await load_at_start(self, "the tone reader", download)
        if not self._model:
            return
        await self.push_frame(
            PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
        )

    def _capabilities(self) -> PerceptionCapabilities:
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm("emotion", _EMOTION, labels=list(self._model.emotions)),
                PerceptionTerm("sarcasm", _SARCASM),
            ]
        )

    async def _read(self, text: str) -> list[PerceptionSignal] | None:
        try:
            tags = await asyncio.to_thread(self._model.tag, text, self._previous)
        except Exception as e:
            await self.push_error(f"the tone reader couldn't read the turn: {e}", e)
            return None
        likeliest = sorted(tags["emotion"]["probabilities"].items(), key=lambda p: -p[1])[:2]
        return [
            PerceptionSignal("emotion", labels=likeliest),
            PerceptionSignal("sarcasm", value=round(tags["sarcasm"]["probability"], 2)),
        ]
