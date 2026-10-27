"""The tagger in a Pipecat pipeline::

    user aggregator -> LLM -> ToneTaggerProcessor -> TTS -> output -> PlainContext -> assistant aggregator

The LLM's text is aggregated into sentences as an expressive TTS aggregates it, and each goes on tagged,
whole, for the TTS to speak at once. The TTS's capabilities go upstream without their tags, so the LLM
never learns to write them: give the pipeline :func:`llm_policy` of your policy and the processor all of it.
"""

import time
from typing import Literal

from loguru import logger
from pipecat.expressive.formatter import EXPRESSION_TAGS
from pipecat.expressive.policy import ExpressionPolicy
from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TextType,
    TTSMetadataFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, standard_message_text
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator

from .._shared import load_at_start, shown_words
from . import readings
from .features import PRESET_NAMES
from .labels import TTS_ADAPTERS, LabelSpace, label_map, tts_key
from .markup import ReplyRenderer, has_markup, without_markup, without_tts_markup
from .model import TAGGER_REVISION, ModelTagger, ToneTaggerModel
from .pace import NORMAL, requested_pace, speed_frame
from .rules import RulesTagger
from .tagger import Message, Tagger, TurnContext

# The labels.yaml key for a TTS it doesn't list: labels are matched to what the TTS declares.
OTHER_TTS = "other"


def llm_policy(policy: ExpressionPolicy | None) -> ExpressionPolicy:
    """The LLM's policy beside a tagger: the developer's guidance lines, without the presets that teach tags."""
    return ExpressionPolicy(
        guidance=[line for line in (policy.guidance if policy else []) if line not in PRESET_NAMES]
    )


class ToneTaggerParams(BaseModel):
    """How the tone tagger runs.

    Parameters:
        rules: Tag with hand-written rules instead of the model, which then isn't loaded.
        size: The model's size, when it's downloaded.
        revision: The model's release, when it's downloaded.
        threads: The CPU threads for one sentence, when the model's downloaded.
        tts: The TTS's labels.yaml key, ``"other"`` for a TTS it doesn't list, or None to take it from
            the TTS service's name.
        vocabulary: Names to try first for a canonical label, such as ``{"warm": ["cheerful"]}``.
        pace: Send the caller's standing pace requests to the TTS as a speed setting.
    """

    model_config = ConfigDict(extra="forbid")

    rules: bool = False
    size: Literal["17m", "32m"] = "17m"
    revision: str = TAGGER_REVISION
    threads: PositiveInt = 1
    tts: str | None = None
    vocabulary: dict[str, list[str]] = {}
    pace: bool = True

    @field_validator("tts")
    @classmethod
    def _known_tts(cls, tts: str | None) -> str | None:
        if tts not in (None, OTHER_TTS, *TTS_ADAPTERS):
            raise ValueError(f"tts must be one of {[*TTS_ADAPTERS, OTHER_TTS]}")
        return tts

    @field_validator("vocabulary")
    @classmethod
    def _known_labels(cls, vocabulary: dict[str, list[str]]) -> dict[str, list[str]]:
        labels = label_map()
        if unknown := sorted(set(vocabulary) - set(labels.deliveries) - set(labels.sounds)):
            raise ValueError(f"no such labels: {unknown}")
        return vocabulary


class ToneTaggerProcessor(LLMTextProcessor):
    """Tags the LLM's replies a sentence at a time, before the TTS.

    Example::

        tagging = ToneTaggerProcessor(context=context, policy=policy)
        Pipeline([..., llm, tagging, tts, transport.output(), tagging.plain_context(), assistant_aggregator])
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        policy: ExpressionPolicy,
        model: ToneTaggerModel | None = None,
        tagger: Tagger | None = None,
        params: ToneTaggerParams | None = None,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            context: The LLM's context, read at the start of each reply.
            policy: The full expression policy; give the pipeline ``llm_policy`` of it.
            model: A loaded model, which calls can share, or None to download one at start.
            tagger: A tagger of your own, instead of the model.
            params: How the tagger runs.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(text_aggregator=SkipTagsAggregator(EXPRESSION_TAGS), **kwargs)
        self.add_text_transformer(self._tag, TextType.SENTENCE)
        self._context = context
        self._policy = policy
        self._model = model
        self._tagger = tagger
        self._params = params or ToneTaggerParams()
        self._plain_context: PlainContext | None = None
        self._warned_tagged_context = False
        self._space: LabelSpace | None = None
        self._service = ""
        self._renderer: ReplyRenderer | None = None
        self._sentences: list[str] = []
        self._markup: list[str] = []
        self.speed = NORMAL
        self.sentence_ms: list[float] = []
        self.turn_ms: float | None = None

    @property
    def space(self) -> LabelSpace | None:
        """The TTS's label space, once its metadata has arrived."""
        return self._space

    def plain_context(self) -> "PlainContext":
        """The processor that keeps the LLM's context plain; put it just before the assistant aggregator."""
        if self._plain_context is None:
            self._plain_context = PlainContext()
        return self._plain_context

    def turn_report(self) -> dict:
        """The last reply: its sentences, plain and tagged, the tagger's times in milliseconds, the TTS speed."""
        return {
            "sentences": list(self._sentences),
            "markup": list(self._markup),
            "turn_ms": None if self.turn_ms is None else round(self.turn_ms, 2),
            "sentence_ms": [round(t, 2) for t in self.sentence_ms],
            "speed": self.speed,
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TTSMetadataFrame):
            self._on_tts_metadata(frame)
            if direction == FrameDirection.UPSTREAM:
                frame.expression = None
        elif isinstance(frame, InterruptionFrame):
            self._renderer = None
        await super().process_frame(frame, direction)
        # After the start goes on, so a first download can't hold up the pipeline's start; frames after
        # it wait for the tagger.
        if isinstance(frame, StartFrame):
            await self._start()
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._start_reply()

    async def _start(self):
        self.speed = NORMAL
        self._tagger = self._tagger or await load_at_start(self, "the tone tagger", self._load_tagger)
        if self._tagger:
            self._tagger.start_call()

    def _load_tagger(self) -> Tagger:
        p = self._params
        if p.rules:
            return RulesTagger()
        return ModelTagger(
            self._model or ToneTaggerModel.download(size=p.size, revision=p.revision, threads=p.threads)
        )

    def _on_tts_metadata(self, frame: TTSMetadataFrame):
        self._service = frame.service_name or ""
        if frame.expression is None:
            logger.warning(
                f"{self}: TTS {frame.service_name!r} performs no expression; replies pass through untagged"
            )
            self._space = None
            return
        tts = self._params.tts or tts_key(self._service) or OTHER_TTS
        self._space = LabelSpace.resolve(tts, frame.expression, self._policy, self._params.vocabulary)

    async def _start_reply(self):
        # Text left from a reply cut short isn't part of this one.
        await self.reset()
        self._sentences, self._markup, self.sentence_ms, self.turn_ms = [], [], [], None
        if not self._space or not self._tagger:
            self._renderer = None
            return
        if not (self._plain_context and self._plain_context.running) and not self._warned_tagged_context:
            logger.warning(
                f"{self}: no plain_context() before the assistant aggregator; the LLM's context keeps the tags"
            )
            self._warned_tagged_context = True
        self._renderer = ReplyRenderer(self._space)
        started = time.perf_counter()
        turn = TurnContext(messages=_messages(self._context), space=self._space, policy=self._policy)
        await self._tagger.start_turn(turn)
        self.turn_ms = (time.perf_counter() - started) * 1000
        if self._params.pace and turn.caller:
            await self._apply_pace(turn.caller.text)

    async def _apply_pace(self, caller: str):
        speed = requested_pace(caller)
        if speed is None or speed == self.speed:
            return
        if frame := speed_frame(self._space.tts, speed, self._service):
            await self.push_frame(frame)
            self.speed = speed

    async def _tag(self, sentence: str, text_type: TextType | str) -> str:
        """One sentence of the reply, tagged; a text transform. Text that isn't spoken stays as written."""
        # LLMTextProcessor keeps whether the text it's aggregating is spoken.
        if not self._space or not self._tagger or self._skip_tts:
            return sentence
        plain = shown_words(sentence)
        if has_markup(sentence):
            logger.warning(f"{self}: the LLM wrote tags; tagging its words without them: {sentence!r}")
        if not plain:
            return ""
        if not self._renderer:
            self._renderer = ReplyRenderer(self._space)
        started = time.perf_counter()
        try:
            labels = await self._tagger.tag_sentence(plain, list(self._sentences))
            markup = self._renderer.render(plain, labels)
        except Exception as e:
            # A sentence the tagger can't label is still spoken: a transform that raises drops it.
            logger.exception(f"{self}: couldn't tag {plain!r}, speaking it untagged: {e}")
            markup = plain
        self.sentence_ms.append((time.perf_counter() - started) * 1000)
        self._sentences.append(plain)
        self._markup.append(markup)
        return markup


class PlainContext(FrameProcessor):
    """Strips the tagger's markup from what the assistant aggregator adds to the LLM's context.

    A TTS reports its words as shown (``text``) and as sent (``raw_text``, tagged); the aggregator
    gets the shown words. ``raw_text`` is dropped, not stripped: after a tag, Pipecat's word
    tracking drifts and can repeat words. Words a TTS never reported are filled from the text it
    was sent, its own markup and all, so that is stripped too.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.running = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.running = True
        elif (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TextFrame)
            and frame.append_to_context
        ):
            if isinstance(frame, AggregatedTextFrame):
                frame.raw_text = None
                frame.text = without_tts_markup(frame.text)
            else:
                frame.text = without_markup(frame.text)
        await self.push_frame(frame, direction)


async def split_sentences(text: str) -> list[str]:
    """A whole reply's plain sentences, split as the processor splits a streamed one."""
    aggregator = SkipTagsAggregator(EXPRESSION_TAGS)
    units = [aggregation.text async for aggregation in aggregator.aggregate(text)]
    if rest := await aggregator.flush():
        units.append(rest.text)
    return [words for unit in units if (words := shown_words(unit))]


def _messages(context: LLMContext) -> list[Message]:
    """The call so far from the LLM's context, each caller reading parsed and folded."""
    messages = []
    for m in context.get_messages():
        role = m.get("role") if isinstance(m, dict) else None
        text = standard_message_text(m)
        if role not in ("user", "assistant") or not text.strip():
            continue
        heard, words = readings.parse(text) if role == "user" else (None, text)
        messages.append(Message(role, shown_words(words), heard))
    return messages
