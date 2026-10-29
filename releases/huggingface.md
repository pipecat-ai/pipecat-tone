---
license: bsd-2-clause
library_name: onnx
language:
  - en
pipeline_tag: text-classification
base_model:
  - sentence-transformers/all-MiniLM-L6-v2
  - jhu-clsp/ettin-encoder-17m
  - jhu-clsp/ettin-encoder-32m
tags:
  - pipecat
  - voice-agents
  - onnx
  - emotion-recognition
  - expressive-tts
---

# Pipecat Tone

Two small CPU models for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents, built for Pipecat's [Realtime Context Standard](https://docs.pipecat.ai)<!-- placeholder: the standard's docs page -->:

- **Tone reader** (`reader/`) reads how each user turn was meant, from its words: the emotion and whether it's sarcastic, with calibrated confidences.
- **Tone tagger** (`tagger/`) voices an LLM's plain reply, a sentence at a time: a delivery, slow spans for details, stressed words, sounds and breaths, in the names the TTS performs.

Each runs in milliseconds on a CPU.

## Realtime Context Standard

The standard carries how a conversation sounds alongside its words. A reading of how the user spoke heads their message in the LLM's context, and each reply reaches the TTS with the expression tags it performs. The reader writes the reading; the tagger writes the tags, so the LLM writes only words.

```text
LLM reads:  ([emotion angry=0.71, neutral=0.20],[sarcasm value=0.99]) Oh great, a forty minute wait.
TTS gets:   <<apologetic>> Sorry about the wait. [[quick breath]] Your order, <<slow: 48213>>, ships ^^today^^.
```

## Use

In Pipecat, install `pipecat-ai[tone]`, turn on `PipelineParams(enable_expressive_conversation=True)`, and add the two processors:

```python
from pipecat.processors.tone.reader import PipecatToneReader
from pipecat.processors.tone.tagger import PipecatToneTagger

pipeline = Pipeline([
    transport.input(), stt, PipecatToneReader(), user_aggregator,
    llm, PipecatToneTagger(), tts, transport.output(), assistant_aggregator,
])
```

The models download from here the first time a pipeline starts; `PipecatToneTagger(size="32m")` picks the larger tagger. Breaths need `ExpressionGuidance.BREATHE` in the expression policy.

## Releases

| Tag | Folder | Model |
| --- | --- | --- |
| `reader-rc2` | `reader/` | MiniLM-L6, 24.2M parameters: 48 MB (fp16) or 96 MB (fp32) |
| `reader-rc1` | `reader/` | MiniLM-L6, 24.2M parameters: 48 MB (fp16) or 96 MB (fp32) |
| `tagger-rc1` | `tagger/17m`, `tagger/32m` | Ettin 17M or 32M, six heads, int8: 63 MB or 115 MB |

`MANIFEST.json` lists each model file's size and SHA-256.

## Scores

**Reader** (rc2), against the majority vote of three LLMs, on turns it never trained on:

| | Real calls (1,500 turns) | Calls with more feeling (897 turns) |
| --- | --- | --- |
| Emotion macro-F1 | 0.936 (neutral, happy) | 0.677 (all nine) |
| Sarcasm | 1 of 1,496 sincere turns called sarcastic | F1 0.903 |
| Calibration error: emotion, sarcasm | 0.077, 0.015 | 0.027, 0.010 |

rc2 reads sarcasm with the agent's line as context. On 3,752 turns of support calls its sarcasm F1 is 0.959 (rc1 0.903), and of 365 pairs of a praise or thanks turn, one after a line it's meant after and one after a line that makes it mean the opposite, it reads both right in 94.0% (rc1 49.0%).

p95 per turn on 4 CPU threads (Apple M1 Max): 5.0 ms.

**Tagger**, against its teacher's labels on 439 held-out calls (combined score):

| | Score | p50 per sentence, 1 thread |
| --- | --- | --- |
| 17M (default) | 0.753 | 13.8 ms |
| 32M | 0.767 | 37.1 ms |

It breathes at most once a reply, about as often as people do in turns that long (ICSI Meeting Corpus), each breath drawn as light, deep or an exhale at people's shares (59%, 36%, 5%).

## Data

- **Tagger** (`data/tagger/v0`–`v3`): 4,999 synthetic support calls, held by deployment LLMs through a real Pipecat pipeline, every sentence labeled by a Claude teacher, and split by domain. Names, emails and numbers are made up.
- **Reader** (`data/reader`): our own files, among them rc2's 30,000 turns of synthetic support calls and 2,000 contrast pairs, and stand-ins for lines from public datasets, which `rebuild/manifest.json` pins to their sources. No third-party text is hosted here.

The reader's public datasets, each under its own licence: NatCS (CDLA-Permissive-2.0); Taskmaster-1, -2 and -3 (CC BY 4.0); ABCD, MultiWOZ 2.2 and STAR (MIT); CCPE-M and Harper Valley Bank (CC BY 4.0); the AMI Meeting Corpus's annotations (CC BY 4.0); DiPCo (CDLA-Permissive-1.0); DynaSent round 2 (CC BY 4.0); iSarcasmEval (MIT); GoEmotions (Apache 2.0); Civil Comments (CC0 1.0); the Gutenberg Dialogue Dataset (MIT). Topical-Chat's test splits (CDLA-Sharing-1.0) are for evaluation only.

## Limits

- For shaping how a voice agent sounds, not for judging or profiling the people it talks to.
- English only. The tagger learned from support calls; the reader from support, assistant and meeting conversations.
- The reader reads words, not voices: a flat transcript reads neutral.
- The tagger knows Cartesia, ElevenLabs, Fish Audio and MiniMax by name; another TTS is matched on the names it declares.
- Scores measure agreement with model labels, not with listeners.

## Licence

BSD-2-Clause ([LICENSE](LICENSE)): both models and the data here. The public datasets keep their own licences.
