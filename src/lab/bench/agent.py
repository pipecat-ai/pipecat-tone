"""A tone-bench call with a tagger::

    user aggregator -> LLM -> recorder -> tagger -> TTS stand-in -> plain context -> assistant aggregator

A copy of tone-bench's ``bench.agent.run_call`` with the tagger added (it has no hook for one): keep
the two in step. The LLM gets ``llm_policy`` of the scenario's policy, the tagger all of it. The
recorder times the LLM's first token; the reply scored is the tagger's.
"""

import asyncio
import time
from dataclasses import dataclass

from bench.agent import (
    REPLY_TIMEOUT,
    USER_SPEECH_TIMEOUT,
    ModelConfig,
    ReplyRecorder,
    TurnRecord,
    build_llm,
    warm_up,
)
from bench.profiles import PERCEPTION, tts_adapter
from bench.scenario import Scenario
from pipecat.expressive.expression import ExpressionCapabilities
from pipecat.expressive.instructions import expression_instructions
from pipecat.expressive.perception import format_perception
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
    TTSMetadataFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from pipecat_tone.tagger import Tagger, ToneTaggerParams, ToneTaggerProcessor, llm_policy


@dataclass
class TaggedTurnRecord(TurnRecord):
    """tone-bench's turn record plus ``ToneTaggerProcessor.turn_report()``."""

    tagger: dict | None = None


class TTSStandIn(FrameProcessor):
    """Sends the TTS's metadata upstream at start, as a TTS does."""

    def __init__(self, tts: str, expression: ExpressionCapabilities):
        super().__init__()
        self._tts = tts
        self._expression = expression

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(
                TTSMetadataFrame(service_name=self._tts, expression=self._expression), FrameDirection.UPSTREAM
            )


def expressive_instruction(scenario: Scenario) -> str:
    """The instruction an LLM tagging itself composes, shown to the judge so criteria match."""
    expression = expression_instructions(
        tts_adapter(scenario.tts).capabilities(), scenario.policy, perception=PERCEPTION
    )
    return "\n\n".join(p for p in (scenario.system_prompt, expression) if p)


async def run_call(
    scenario: Scenario, config: ModelConfig, tagger: Tagger
) -> tuple[str | None, list[TaggedTurnRecord]]:
    """Hold the scenario's call; returns the LLM's composed instruction and a record per turn."""
    await warm_up(config)  # never time a call against a model that fell asleep
    adapter = tts_adapter(scenario.tts)
    llm = build_llm(config, scenario.system_prompt)
    recorder = ReplyRecorder()
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT)]
            )
        ),
    )
    tagging = ToneTaggerProcessor(
        context=context, policy=scenario.policy, tagger=tagger, params=ToneTaggerParams(tts=scenario.tts)
    )
    worker = PipelineWorker(
        Pipeline(
            [
                user_aggregator,
                llm,
                recorder,
                tagging,
                TTSStandIn(scenario.tts, adapter.capabilities()),
                tagging.plain_context(),
                assistant_aggregator,
            ]
        ),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            enable_expressive_conversation=True,
            expression_policy=llm_policy(scenario.policy),
        ),
    )
    reply_done = asyncio.Event()

    # After the reply is in the context, so the next turn can't race it.
    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        reply_done.set()

    records: list[TaggedTurnRecord] = []

    async def hold_call():
        await worker.queue_frames([PerceptionMetadataFrame(service_name="perception", perception=PERCEPTION)])
        try:
            for index, turn in enumerate(scenario.turns, 1):
                llm.start_turn()
                recorder.start_turn()
                reply_done.clear()
                started = time.monotonic()
                await worker.queue_frames(
                    [
                        VADUserStartedSpeakingFrame(),
                        TranscriptionFrame(
                            text=turn.user,
                            user_id="caller",
                            timestamp=time_now_iso8601(),
                            perception=turn.heard,
                        ),
                        VADUserStoppedSpeakingFrame(),
                    ]
                )
                await asyncio.wait_for(reply_done.wait(), timeout=REPLY_TIMEOUT)
                user_messages = [m for m in context.get_messages() if m.get("role") == "user"]
                report = tagging.turn_report()
                records.append(
                    TaggedTurnRecord(
                        index=index,
                        user=turn.user,
                        heard=format_perception(turn.heard) if turn.heard else None,
                        user_message=str(user_messages[-1]["content"]) if user_messages else "",
                        reply=" ".join(report["markup"]),
                        reasoning=llm.reasoning.strip(),
                        completion_tokens=llm.usage.completion_tokens if llm.usage else None,
                        reasoning_tokens=llm.usage.reasoning_tokens if llm.usage else None,
                        first_token_s=(recorder.first_token_at - started)
                        if recorder.first_token_at
                        else None,
                        total_s=time.monotonic() - started,
                        latency=llm.latency(),
                        tagger=report,
                    )
                )
        finally:
            await worker.queue_frames([EndFrame()])

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), hold_call())
    return llm.system_instruction, records
