"""What the reader and the tagger share in a pipeline: words read out of markup, and a model loaded at start."""

import asyncio
from collections.abc import Callable
from typing import TypeVar

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.formatter import ExpressionFormatter
from pipecat.frames.frames import TextType
from pipecat.processors.frame_processor import FrameProcessor

T = TypeVar("T")


def shown_words(text: str, adapter: BaseExpressionAdapter | None = None) -> str:
    """The words of markup, as Pipecat's formatter shows them to clients.

    With a TTS's adapter, a name it declares keeps its kind in the other kind's brackets, as when it speaks.
    """
    formatter = ExpressionFormatter(lambda: adapter or BaseExpressionAdapter(), owner="pipecat-tone")
    return ExpressionFormatter.shown_text(formatter.parse(text, whole=True), TextType.SENTENCE)


async def load_at_start(processor: FrameProcessor, what: str, load: Callable[[], T]) -> T | None:
    """A model loaded off the event loop, or None once the reason it couldn't be is reported.

    The processor is then unusable, and the pipeline worker's policy decides what happens next.
    """
    try:
        return await asyncio.to_thread(load)
    except Exception as e:
        await processor.push_error(f"couldn't load {what}: {e}", e, force_treat_as_permanent=True)
        return None
