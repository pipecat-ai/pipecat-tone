# pipecat-tone

Two small CPU models for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents, and the code that trains them. Each answers in milliseconds beside a live call.

| Model | What it does | Latest release | Weights |
|---|---|---|---|
| [Tone reader](releases/reader/) | Hears how each caller turn was meant, from its words: the emotion, and whether it's sarcastic, for the LLM to read | [rc1](releases/reader/rc1/) | [`reader/` at `reader-rc1`](https://huggingface.co/pipecat-ai/tone-models/tree/reader-rc1/reader) |
| [Tone tagger](releases/tagger/) | Voices the LLM's plain reply: adds Pipecat's expression markup a sentence at a time (delivery, slow details, stress, sounds) in the TTS's own names | [rc1](releases/tagger/rc1/) | [`tagger/` at `tagger-rc1`](https://huggingface.co/pipecat-ai/tone-models/tree/tagger-rc1/tagger) |

Weights and training data are in [`pipecat-ai/tone-models`](https://huggingface.co/pipecat-ai/tone-models) on Hugging Face. A model downloads when its pipeline first starts; every release has a manifest of checksums.

## Install

Both build on Pipecat's expressive conversation, [pipecat-ai/pipecat#6319](https://github.com/pipecat-ai/pipecat/pull/6319), which isn't released yet: install Pipecat from that branch.

```bash
uv add "pipecat-tone[model] @ git+https://github.com/pipecat-ai/pipecat-tone"
```

## Tone reader

Between the STT and the user aggregator. It sets each final transcription's perception, which the LLM reads with expressive conversation on.

```python
from pipecat_tone.reader import ToneReaderProcessor

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        ToneReaderProcessor(),
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
params = PipelineParams(enable_expressive_conversation=True, expression_policy=policy)
worker = PipelineWorker(pipeline, params=params)
```

[examples/reader/bot.py](examples/reader/bot.py) is a whole call, with an LLM that voices its own replies.

## Tone tagger

Between the LLM and the TTS. The LLM writes plain words, and the tagger voices them.

```python
from pipecat_tone.tagger import ToneTaggerProcessor, llm_policy

tagging = ToneTaggerProcessor(context=context, policy=policy)
pipeline = Pipeline(
    [
        ...,
        user_aggregator,
        llm,
        tagging,
        tts,
        transport.output(),
        tagging.plain_context(),
        assistant_aggregator,
    ]
)
params = PipelineParams(enable_expressive_conversation=True, expression_policy=llm_policy(policy))
worker = PipelineWorker(pipeline, params=params)
```

The tagger follows the full policy. The LLM gets it without the presets that teach tags, and `plain_context()` keeps tags out of its context, so it never learns to write them. Breathing is on request: with the tagger's `BREATHE` preset in the policy's guidance, it puts an audible breath before a long or difficult sentence now and then, on a TTS that has one (ElevenLabs, MiniMax, Fish). It breathes at most once a reply, about as often as people do, and each breath is light, deep or an exhale, in people's shares.

```python
from pipecat_tone.tagger import BREATHE

guidance = [ExpressionGuidance.MATCH_STYLE, ExpressionGuidance.SLOW_FOR_DETAILS, BREATHE]
policy = ExpressionPolicy(guidance=guidance)
```
 `ToneTaggerParams` sets the rest: `rules=True` (no model), `size="32m"`, `revision`, `threads`, the TTS, and a `vocabulary` in your TTS's own words. [examples/tagger/bot.py](examples/tagger/bot.py) runs both models.

On tone-bench, Pipecat's benchmark for expressive conversation, phonellm-mini scores 90.9 with the tagger after it, against 57.9 alone.

## Both

`model=ToneReaderModel.download()` or `model=ToneTaggerModel.download()` shares one loaded model between calls. If a download fails, the error is reported and the call goes on without that model.

## Training

[`src/lab/`](src/lab/) trains both models on Modal: the reader's pipeline, and the tagger's corpus, training and benchmark runs. Each release's changelog says what it measured.

## Licence

BSD-2-Clause: the code, both models and our training data ([LICENSE](LICENSE)). The public datasets the reader learned from keep their own licences; training downloads them from their sources.
