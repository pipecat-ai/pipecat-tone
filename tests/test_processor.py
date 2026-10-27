import re
import time

import pytest
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.perception import PerceptionSignal, format_perception
from pipecat.expressive.policy import ExpressionPolicy
from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSMetadataFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pydantic import ValidationError

from pipecat_tone.hub import ModelDownloadError
from pipecat_tone.tagger import (
    NoOpTagger,
    SentenceLabels,
    Span,
    Tagger,
    ToneTaggerModel,
    ToneTaggerParams,
    ToneTaggerProcessor,
    TurnContext,
)
from pipecat_tone.tagger.labels import TTS_ADAPTERS
from pipecat_tone.tagger.processor import split_sentences


class SlowCodes(Tagger):
    """Slows any token that looks like a code, and is sympathetic when the caller sounds sad."""

    def __init__(self):
        self.turns: list[TurnContext] = []

    async def start_turn(self, turn: TurnContext) -> None:
        self.turns.append(turn)

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        heard = self.turns[-1].heard if self.turns else None
        sad = any(label == "sad" and c >= 0.5 for s in heard or [] for label, c in s.labels)
        spans = []
        if "4417" in sentence:
            start = sentence.index("4417")
            spans.append(Span(start, start + 4, "slow"))
        return SentenceLabels(delivery="sympathetic" if sad and not earlier else None, spans=spans)


def metadata(tts: str) -> TTSMetadataFrame:
    return TTSMetadataFrame(service_name=tts, expression=TTS_ADAPTERS[tts]().capabilities())


def reply(*chunks: str):
    return [LLMFullResponseStartFrame(), *(LLMTextFrame(text=c) for c in chunks), LLMFullResponseEndFrame()]


def texts(frames) -> list[str]:
    """The sentences the TTS would speak, one frame each."""
    return [f.text for f in frames if isinstance(f, AggregatedTextFrame) and not f.skip_tts]


def remembered(frames) -> list[str]:
    """What the assistant aggregator would add to the LLM's context, with no TTS in between."""
    return [f.text for f in frames if isinstance(f, AggregatedTextFrame) and f.append_to_context]


def tagging(tagger: Tagger, tts: str | None = None, context: LLMContext | None = None) -> ToneTaggerProcessor:
    return ToneTaggerProcessor(
        context=context or LLMContext(),
        policy=ExpressionPolicy(),
        tagger=tagger,
        params=ToneTaggerParams(tts=tts),
    )


def with_plain_context(processor: ToneTaggerProcessor) -> Pipeline:
    return Pipeline([processor, processor.plain_context()])


class FakeTTS(TTSService):
    """A TTS that speaks silence, with Cartesia's expression markup, as a real TTS reports text."""

    def __init__(self):
        super().__init__(sample_rate=16000, settings=TTSSettings(model=None, voice=None, language=None))
        self.requests: list[str] = []

    def get_expression_adapter(self):
        return TTS_ADAPTERS["cartesia"]()

    async def run_tts(self, text: str, context_id: str):
        self.requests.append(text)
        yield TTSAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1, context_id=context_id)


class WordTimedTTS(TTSService):
    """A TTS that reports the words it spoke, with ElevenLabs' audio tags, and never the last
    word of a request, as ElevenLabs often doesn't. Pipecat then completes the sentence itself."""

    def __init__(self):
        super().__init__(
            sample_rate=16000,
            push_text_frames=False,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
        )

    def get_expression_adapter(self):
        return TTS_ADAPTERS["elevenlabs"]()

    async def run_tts(self, text: str, context_id: str):
        yield TTSAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1, context_id=context_id)
        words = re.sub(r"\[[^\]]*\]", "", text).split()[:-1]
        await self.add_word_timestamps([(w, i / 10) for i, w in enumerate(words)], context_id)


class Broken(Tagger):
    """Fails on every sentence."""

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        raise RuntimeError("no model")


class SympatheticFirst(Tagger):
    """Sympathetic on a reply's first sentence, so the next one opens with ``<<neutral>>``."""

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        return SentenceLabels(delivery=None if earlier else "sympathetic")


async def test_hides_tts_capabilities_from_the_llm():
    processor = tagging(NoOpTagger())
    _, up = await run_test(
        processor,
        frames_to_send=[metadata("cartesia")],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[TTSMetadataFrame],
    )
    assert up[0].expression is None
    assert processor.space is not None and processor.space.tts == "cartesia"


async def test_sentences_go_on_whole_one_frame_each():
    processor = tagging(NoOpTagger(), "fish")
    down, _ = await run_test(
        processor,
        frames_to_send=[
            metadata("fish"),
            *reply("Sure", ", one moment.", " Your total", " is $3", ".99 today."),
        ],
    )
    assert texts(down) == ["Sure, one moment.", "Your total is $3.99 today."]


async def test_tags_sentences_from_the_callers_reading():
    heard = [PerceptionSignal("tone", labels=[("sad", 0.72), ("neutral", 0.2)])]
    context = LLMContext()
    context.add_message(
        {"role": "user", "content": f"{format_perception(heard)} My card ending 4417 was stolen."}
    )
    tagger = SlowCodes()
    processor = tagging(tagger, "cartesia", context)
    down, _ = await run_test(
        processor,
        frames_to_send=[
            metadata("cartesia"),
            *reply("I'm so sorry.", " I've blocked the card ending 4417.", " "),
        ],
    )
    assert texts(down) == [
        "<<sympathetic>> I'm so sorry.",
        "<<neutral>> I've blocked the card ending <<slow: 4417>>.",
    ]
    assert tagger.turns[0].messages[0].text == "My card ending 4417 was stolen."
    assert processor.turn_ms is not None and len(processor.sentence_ms) == 2


async def test_the_tagger_reads_the_full_policy():
    policy = ExpressionPolicy(guidance=["Keep it brief."])
    tagger = SlowCodes()
    await run_test(
        ToneTaggerProcessor(
            context=LLMContext(), policy=policy, tagger=tagger, params=ToneTaggerParams(tts="cartesia")
        ),
        frames_to_send=[metadata("cartesia"), *reply("Card 4417.")],
    )
    assert tagger.turns[0].policy is policy


async def test_strips_tags_the_llm_wrote():
    processor = tagging(NoOpTagger(), "minimax")
    down, _ = await run_test(
        processor, frames_to_send=[metadata("minimax"), *reply("<<calm>> All set.", " Bye!")]
    )
    assert texts(down) == ["All set.", "Bye!"]


async def test_interruption_drops_the_unfinished_sentence():
    processor = tagging(NoOpTagger(), "minimax")
    down, _ = await run_test(
        processor,
        frames_to_send=[
            metadata("minimax"),
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="One moment while I"),
            InterruptionFrame(),
            *reply("Yes?"),
        ],
    )
    assert texts(down) == ["Yes?"]


async def test_without_tts_labels_sentences_go_on_untagged():
    processor = tagging(NoOpTagger())
    down, _ = await run_test(processor, frames_to_send=reply("<<calm>> Hi", " there."))
    assert texts(down) == ["<<calm>> Hi there."]


async def test_tagged_context_without_plain_context():
    processor = tagging(SlowCodes(), "cartesia")
    down, _ = await run_test(processor, frames_to_send=[metadata("cartesia"), *reply("Card 4417.")])
    assert remembered(down) == texts(down) == ["Card <<slow: 4417>>."]


async def test_plain_context_strips_the_tags_without_a_tts():
    processor = tagging(SlowCodes(), "cartesia")
    down, _ = await run_test(
        with_plain_context(processor),
        frames_to_send=[metadata("cartesia"), *reply("Sorry.", " Card 4417.", " ")],
    )
    assert remembered(down) == ["Sorry.", "Card 4417."]


async def test_plain_context_with_a_real_tts():
    """The TTS speaks the tags, and the LLM's context gets the reply without them."""
    heard = [PerceptionSignal("tone", labels=[("sad", 0.8)])]
    context = LLMContext()
    context.add_message({"role": "user", "content": f"{format_perception(heard)} Card 4417 is gone."})
    processor = tagging(SlowCodes(), "cartesia", context)
    tts = FakeTTS()
    await run_test(
        Pipeline([processor, tts, processor.plain_context(), LLMAssistantAggregator(context)]),
        frames_to_send=[metadata("cartesia"), *reply("I'm so sorry.", " Card 4417 is blocked.")],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )
    assert tts.requests[0] == '<emotion value="sympathetic"/> I\'m so sorry.'
    assert context.get_messages()[-1] == {
        "role": "assistant",
        "content": "I'm so sorry. Card 4417 is blocked.",
    }


async def test_plain_context_with_word_timestamps_that_miss_the_last_word():
    """No stray letters ("for r you?") when Pipecat completes a sentence after a tag itself."""
    context = LLMContext()
    processor = tagging(SympatheticFirst(), "elevenlabs", context)
    await run_test(
        Pipeline([processor, WordTimedTTS(), processor.plain_context(), LLMAssistantAggregator(context)]),
        frames_to_send=[
            metadata("elevenlabs"),
            *reply("I'm sorry about the wait.", " Could I look into it for you?"),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )
    assert context.get_messages()[-1] == {
        "role": "assistant",
        "content": "I'm sorry about the wait. Could I look into it for you?",
    }


async def test_a_slow_down_request_sets_the_tts_speed_once():
    context = LLMContext()
    context.add_message({"role": "user", "content": "Whoa, slow down a sec."})
    processor = tagging(NoOpTagger(), "minimax", context)
    down, _ = await run_test(
        processor, frames_to_send=[metadata("minimax"), *reply("Sure."), *reply("Okay.")]
    )
    updates = [f for f in down if isinstance(f, TTSUpdateSettingsFrame)]
    assert len(updates) == 1 and updates[0].delta.speed == 0.85
    assert processor.speed == 0.85


async def test_no_pace_request_no_settings():
    context = LLMContext()
    context.add_message({"role": "user", "content": "Say that again?"})
    processor = tagging(NoOpTagger(), "cartesia", context)
    down, _ = await run_test(processor, frames_to_send=[metadata("cartesia"), *reply("Sure.")])
    assert not [f for f in down if isinstance(f, TTSUpdateSettingsFrame)]


async def test_split_sentences_as_the_processor_does():
    assert await split_sentences("<<calm>> Sure, one moment. Your total is $3.99 today.") == [
        "Sure, one moment.",
        "Your total is $3.99 today.",
    ]


async def test_a_sentence_the_tagger_fails_on_is_spoken_untagged():
    processor = tagging(Broken(), "cartesia")
    down, _ = await run_test(processor, frames_to_send=[metadata("cartesia"), *reply("Hi there.", " Bye.")])
    assert texts(down) == ["Hi there.", "Bye."]


async def test_text_that_isnt_spoken_is_left_as_written():
    processor = tagging(SlowCodes(), "cartesia")
    unspoken = LLMTextFrame(text="Card 4417.")
    unspoken.skip_tts = True
    down, _ = await run_test(
        processor,
        frames_to_send=[
            metadata("cartesia"),
            LLMFullResponseStartFrame(),
            unspoken,
            LLMFullResponseEndFrame(),
        ],
    )
    assert [f.text for f in down if isinstance(f, AggregatedTextFrame)] == ["Card 4417."]


async def test_a_function_call_doesnt_hide_the_caller():
    context = LLMContext()
    context.add_message({"role": "user", "content": "Where's my parcel?"})
    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "track", "arguments": "{}"}}],
        }
    )
    context.add_message({"role": "tool", "tool_call_id": "1", "content": "out for delivery"})
    tagger = SlowCodes()
    await run_test(
        tagging(tagger, "cartesia", context),
        frames_to_send=[metadata("cartesia"), *reply("It's out for delivery.")],
    )
    assert [m.role for m in tagger.turns[0].messages] == ["user"]


async def test_the_rules_need_no_model():
    params = ToneTaggerParams(rules=True, tts="cartesia")
    processor = ToneTaggerProcessor(context=LLMContext(), policy=ExpressionPolicy(), params=params)
    down, _ = await run_test(processor, frames_to_send=[metadata("cartesia"), *reply("Your code is 7Q2K9X.")])
    assert texts(down) == ["Your code is <<slow: 7Q2K9X>>."]


class StubModel:
    """Slows every sentence's first word."""

    config = {}

    def predict(self, turn: TurnContext, earlier: list[str], sentence: str) -> SentenceLabels:
        return SentenceLabels(spans=[Span(0, sentence.index(" "), "slow")])


def downloads(monkeypatch, download):
    monkeypatch.setattr(ToneTaggerModel, "download", download)
    params = ToneTaggerParams(size="32m", revision="tagger-rc9", threads=2, tts="cartesia")
    return ToneTaggerProcessor(context=LLMContext(), policy=ExpressionPolicy(), params=params)


async def test_the_model_is_downloaded_as_params_say(monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return StubModel()

    down, _ = await run_test(
        downloads(monkeypatch, download), frames_to_send=[metadata("cartesia"), *reply("Card 4417.")]
    )
    assert calls == [{"size": "32m", "revision": "tagger-rc9", "threads": 2}]
    assert texts(down) == ["<<slow: Card>> 4417."]


async def test_a_slow_download_doesnt_hold_up_the_start(monkeypatch):
    def download(**kwargs):
        time.sleep(1.5)
        return StubModel()

    down, _ = await run_test(
        downloads(monkeypatch, download),
        frames_to_send=[metadata("cartesia"), *reply("Card 4417.")],
        start_timeout=1.0,
    )
    assert texts(down) == ["<<slow: Card>> 4417."]


async def test_a_model_that_wont_download_is_reported_and_replies_go_on_untagged(monkeypatch):
    def download(**kwargs):
        raise ModelDownloadError("no such release")

    down, up = await run_test(
        downloads(monkeypatch, download), frames_to_send=[metadata("cartesia"), *reply("Card 4417.")]
    )
    assert (
        "couldn't load the tone tagger: no such release"
        in [f for f in up if isinstance(f, ErrorFrame)][0].error
    )
    assert texts(down) == ["Card 4417."]


async def test_a_tts_labels_yaml_doesnt_list_with_a_vocabulary():
    caps = ExpressionCapabilities(delivery=[ExpressionTerm("cheerful"), ExpressionTerm("slow")])
    params = ToneTaggerParams(vocabulary={"warm": ["cheerful"]})
    processor = ToneTaggerProcessor(
        context=LLMContext(), policy=ExpressionPolicy(), tagger=NoOpTagger(), params=params
    )
    await run_test(
        processor, frames_to_send=[TTSMetadataFrame(service_name="AcmeTTSService#0", expression=caps)]
    )
    assert processor.space.tts == "other"
    assert processor.space.deliveries == {"warm": "cheerful", "slow": "slow"}


@pytest.mark.parametrize(
    "bad", [{"sise": "32m"}, {"threads": 0}, {"tts": "openai"}, {"vocabulary": {"warmth": ["x"]}}]
)
def test_params_that_would_do_nothing_are_refused(bad):
    with pytest.raises(ValidationError):
        ToneTaggerParams(**bad)
