# Tone reader rc1

Hears how each turn of a call was meant, from its words: the speaker's emotion, whether the turn is sarcastic, the move it makes, and whether it's only a listening sound. Each comes with a calibrated confidence. It runs on a CPU in a few milliseconds. It was first released as Pipecat Tone rc1.

```python
from pipecat_tone.reader import ToneReaderParams, ToneReaderProcessor

ToneReaderProcessor(params=ToneReaderParams(revision="reader-rc1"))  # precision="fp32" for full precision
```

## Output

On each final transcription, as its `perception`: the two likeliest of nine emotions (neutral, happy, amused, angry, sad, afraid, surprised, disgusted, contemptuous) with their confidences, and the chance the turn is sarcastic. `ToneReaderModel.tag()` also gives the emotion's intensity, the move (asks, answers, hedges, doesnt_know, withholds, closes, other) and whether the turn is a backchannel. The turn before sharpens the move and sarcasm: the processor reads each turn with the bot's reply before it.

## How well it reads calls

Labels are the majority vote of three large language models; none of these turns were trained on.

| | Real calls (1,500 turns) | Calls with more feeling (897 turns) |
|---|---|---|
| Emotion macro-F1 | 0.940 (neutral, happy) | 0.598 (all nine) |
| Move macro-F1 | 0.843 | 0.800 |
| Sarcasm | 1 of 1,496 sincere turns called sarcastic | F1 0.887; 1.0% of sincere turns called sarcastic |
| Calibration error: emotion, move, sarcasm | 0.079, 0.027, 0.016 | 0.036, 0.039, 0.014 |

It reads words, not voices: it can't hear tone of voice, laughter or a noisy line, and a flat transcript reads neutral.

## Training

A 24.2M-parameter MiniLM-L6 encoder (`sentence-transformers/all-MiniLM-L6-v2`), trained on 1,083,920 lines with the soft labels of five Ettin-1B teachers, then calibrated. The lines come from 16 openly licensed datasets of conversation and comments, from lines an LLM wrote for the purpose, and from speech-to-text twins of both; a committee of three large language models labelled them. `data/reader/` in `pipecat-ai/tone-models` holds what training reads, without third-party text, and a manifest of those datasets, each under its own licence; `lab.reader.pipeline.fetch` downloads them and rebuilds the rest.

## Files

`reader/` in `pipecat-ai/tone-models` at `reader-rc1`:

| File | Bytes | sha256 |
|---|---|---|
| `config.json` | 1,240 | `6ef49ccfa37164fce382738a2e8c2a2298cffa8f6e4acc420c3e9619a0cce3a8` |
| `tokenizer.json` | 711,649 | `1b19820b21bee2ab9a5d3f2ac461f9517c4589536c7d751d60d38877993c5e8b` |
| `model_fp16.onnx` | 48,247,715 | `8129621c02dad6f3875905be04fb4ac2928690e21ac0ec9862e7cfe88718050a` |
| `model.opt.onnx` | 96,298,762 | `1fd07dd2f4e65edbc5b82de28265b44128bcfd0dd31c3b26270fc5f0659834bb` |

p95 per turn on 4 CPU threads (Apple M1 Max): 5.0 ms with the turn before, 3.7 ms without; 11.9 ms on 1 thread. The 16-bit file: 6.3 ms on 4 threads, 11.9 ms on 1. Measured by `lab.reader.pipeline.latency`: [results/reader/latency](../../../results/reader/latency/). Licence: BSD-2-Clause.
