"""The tone reader in a live call: the LLM hears how each turn was meant, and answers in kind.

    uv run --group examples examples/reader/bot.py    # then open http://localhost:7860/client

Needs DEEPGRAM_API_KEY, OPENAI_API_KEY and CARTESIA_API_KEY in .env. The reader is downloaded from Hugging
Face the first time.
"""

import os

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_tone.reader import ToneReaderProcessor

load_dotenv(override=True)

SYSTEM_INSTRUCTION = """\
You are Sam, a friendly support agent on the phone for Northwind Outfitters, an online shop for outdoor \
gear. Keep replies to a sentence or two, with nothing that can't be spoken.\
"""

# The LLM reads each turn's reading, and voices its own replies.
POLICY = ExpressionPolicy(
    guidance=[
        ExpressionGuidance.MATCH_STYLE,
        "When the caller clearly sounds angry, apologize once and keep calm.",
        "When a turn is likely sarcastic, answer what they mean, not what they say.",
    ]
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model="gpt-4.1", system_instruction=SYSTEM_INSTRUCTION),
    )
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID") or "86e30c1d-714b-4074-a1f2-1cb6b552fb49"
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ToneReaderProcessor(),  # how the caller sounds, for the LLM
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True, enable_expressive_conversation=True, expression_policy=POLICY
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the caller in one short sentence and ask how you can help.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args, {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
    )
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
