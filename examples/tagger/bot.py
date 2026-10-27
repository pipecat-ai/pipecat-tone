"""The tone tagger in a live call: the LLM writes plain words and the tagger voices them.

    uv run --group examples examples/tagger/bot.py    # then open http://localhost:7860/client

Needs DEEPGRAM_API_KEY, CARTESIA_API_KEY, PHONELLM_MINI_URL and PHONELLM_MINI_API_KEY in .env. The reader
and the tagger are downloaded from Hugging Face the first time.
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
from pipecat.services.baseten.llm import BasetenLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_tone.reader import ToneReaderProcessor
from pipecat_tone.tagger import ToneTaggerProcessor, llm_policy

load_dotenv(override=True)

# One caller's order, so the agent has real details to give and the tagger has details to slow.
SYSTEM_INSTRUCTION = """\
You are Sam, a friendly support agent on the phone for Northwind Outfitters, an online shop for outdoor \
gear. The caller's account is open on your screen:
- Order NW-48213: a two-person tent and a rain fly, 289 dollars, placed October 3.
- The courier sent it to the wrong depot, so it was stuck for two weeks. It's back on the van and arrives \
tomorrow before 8 PM. Tracking number 1Z 84A 770 3.
- For the delay you can offer a 20 percent refund, 57 dollars and 80 cents, back to their card in 3 to 5 days.
- Case number for this call: CS-30917. Support email: help@northwind.com.

Answer from the account straight away; never say you'll look something up. Give numbers when asked, and \
read them back when asked. Keep replies to a sentence or two, with nothing that can't be spoken.\
"""

# The tagger follows the full policy; the LLM gets it without the presets that teach tags.
POLICY = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE, ExpressionGuidance.SLOW_FOR_DETAILS])


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = BasetenLLMService(
        api_key=os.environ["PHONELLM_MINI_API_KEY"],
        base_url=os.environ["PHONELLM_MINI_URL"],
        settings=BasetenLLMService.Settings(
            model="pipecat-ai/phonellm-mini-alpha-2",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
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
    tagging = ToneTaggerProcessor(context=context, policy=POLICY)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ToneReaderProcessor(),  # how the caller sounds, for the LLM and the tagger
            user_aggregator,
            llm,
            tagging,
            tts,
            transport.output(),
            tagging.plain_context(),  # keeps the tags out of the LLM's context
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True, enable_expressive_conversation=True, expression_policy=llm_policy(POLICY)
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
