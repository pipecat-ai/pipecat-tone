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

# Pipecat tone models

Two small CPU models for voice agents built with [Pipecat](https://github.com/pipecat-ai/pipecat), and the data they learn from:

- **Tone reader** (`reader/`): hears how each caller turn was meant, from its words: the emotion, whether it's sarcastic, the move it makes and whether it's only a listening sound, each with a calibrated confidence.
- **Tone tagger** (`tagger/`): adds Pipecat's expression markup to an LLM's plain reply, a sentence at a time, so the LLM only writes words: a delivery, slow spans for details, stress and sounds, in the call's TTS names.

Each answers in milliseconds on a CPU, beside a live call. They build on Pipecat's expressive conversation ([pipecat-ai/pipecat#6319](https://github.com/pipecat-ai/pipecat/pull/6319)).

## Use

[`pipecat-tone`](https://github.com/pipecat-ai/pipecat-tone) downloads them from here when a pipeline first starts:

```bash
uv add "pipecat-tone[model] @ git+https://github.com/pipecat-ai/pipecat-tone"
```

```python
from pipecat_tone.reader import ToneReaderProcessor
from pipecat_tone.tagger import ToneTaggerProcessor

reader = ToneReaderProcessor()  # between the STT and the user aggregator
tagger = ToneTaggerProcessor(context=context, policy=policy)  # between the LLM and the TTS
```

## Releases

| Tag | Folder | Model |
| --- | --- | --- |
| `reader-rc1` | `reader/` | A 24.2M-parameter MiniLM-L6 encoder: `model_fp16.onnx` (48 MB) and `model.opt.onnx` (96 MB) |
| `tagger-rc1` | `tagger/17m`, `tagger/32m` | Ettin 17M and 32M encoders with six heads, int8: 63 MB and 115 MB |

At a release's tag, its folder holds what the runtime loads, and `MANIFEST.json` lists every model file's size and SHA-256.

## Tone reader

Labels are the majority vote of three large language models; none of these turns were trained on.

| | Real calls (1,500 turns) | Calls with more feeling (897 turns) |
| --- | --- | --- |
| Emotion macro-F1 | 0.940 (neutral, happy) | 0.598 (all nine) |
| Move macro-F1 | 0.843 | 0.800 |
| Sarcasm | 1 of 1,496 sincere turns called sarcastic | F1 0.887; 1.0% of sincere turns called sarcastic |
| Calibration error: emotion, move, sarcasm | 0.079, 0.027, 0.016 | 0.036, 0.039, 0.014 |

p95 per turn on 4 CPU threads (Apple M1 Max): 5.0 ms with the turn before, 3.7 ms without; 11.9 ms on 1 thread. The 16-bit file reads the same tags at half the size.

## Tone tagger

Held-out test domains, against the teacher's labels, without breaths (combined score; the rules alone score 0.38 and 0.37 on v0's and v1's):

| | v0 domains | v1 domains | v2 domains | v3 domains | All 439 test calls | 1-thread p50 |
| --- | --- | --- | --- | --- | --- | --- |
| **17M** | 0.751 | 0.759 | 0.748 | 0.742 | **0.753** | 13.8 ms |
| **32M** | 0.757 | 0.763 | 0.783 | 0.789 | **0.767** | 37.1 ms |

On tone-bench, phonellm-mini with the 17M after it scores 90.9 over two passes, judged by Claude Opus 5.5: expression 97.1 (recall 100.0, precision 94.4). With the tagger's rules instead it scores 90.0, and phonellm-mini alone 57.9.

Breaths come only when the policy's guidance has the tagger's `BREATHE` preset, on a TTS that has one. The tagger breathes at most once a reply, and in replies of each length about as often as people do, from the breaths the ICSI Meeting Corpus's transcribers marked: the 17M in 22% of replies on the held-out domains, as people do in turns that long. Each breath is drawn as light, deep or an exhale, at people's shares (59%, 36% and 5%). The 17M is the default.

## Data

### Tagger: `data/tagger/v0` to `v3`

The tagger's corpus: 999, 2,000, 1,000 and 1,000 synthetic support calls. Claude wrote each scenario; deployment LLMs held the calls through a real Pipecat pipeline; a Claude teacher labeled every sentence without changing a word; another vendor's model reviewed each reply. Half of v2's calls ask to breathe, and each of v3's has a moment for a rare delivery. Each call's JSON has its spec, scenario, replies, the teacher's labels and the reviewer's verdicts, and its breath pass: which replies breathe, drawn at people's rate for their length, and the sentence the teacher chose for each breath. The rows training reads (`train`, `dev` and `test.jsonl`) are split by domain, with `stats.json`. The domain families tone-bench tests are left out. Names, emails, phone and account numbers in the calls are made up.

### Reader: `data/reader`

What the reader's training and checks read, with no third-party text in it: the files that are ours as they are (rc1's run, its teachers' soft labels, label sets of our own lines), and stand-ins for the files that mix our lines with lines from public datasets, which say how to make each of those lines again from its source. `rebuild/manifest.json` pins every source and every file's SHA-256, and `lab.reader.pipeline.fetch` in pipecat-tone downloads the sources and rebuilds the exact files training reads. [data/reader/README.md](data/reader/README.md) has the details.

The public datasets, each under its own licence: NatCS (CDLA-Permissive-2.0); Taskmaster-1, -2 and -3 (CC BY 4.0); ABCD (MIT); MultiWOZ 2.2 (MIT); STAR (MIT); CCPE-M (CC BY 4.0); Harper Valley Bank (CC BY 4.0); the AMI Meeting Corpus's manual annotations (CC BY 4.0); DiPCo (CDLA-Permissive-1.0); DynaSent round 2 (CC BY 4.0); iSarcasmEval (MIT); GoEmotions (Apache 2.0); Civil Comments (CC0 1.0); and the Gutenberg Dialogue Dataset (MIT, from public-domain books). Topical-Chat's test splits (CDLA-Sharing-1.0) are read for evaluation only.

## Intended use and limits

- For shaping how a voice agent sounds and what its LLM reads, not for judging or profiling the people it talks to.
- English only. The tagger learned from support calls; the reader from support, assistant and meeting conversations.
- The reader reads words, not voices: it can't hear tone of voice, laughter or a noisy line, and a flat transcript reads neutral.
- The tagger writes the names `labels.yaml` gives Cartesia, ElevenLabs, Fish and MiniMax; another TTS is matched on the names it declares.
- Scores against the teacher measure agreement with Claude's labels, not with listeners.

## Licence

BSD-2-Clause: both models and the data here ([LICENSE](LICENSE)). The public datasets the reader learned from keep their own licences, listed above; training downloads them from their sources, not from here.
