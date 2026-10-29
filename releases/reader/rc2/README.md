# Tone reader rc2

Hears how each turn of a call was meant, from its words: the speaker's emotion, whether the turn is sarcastic, the move it makes, and whether it's only a listening sound. Each comes with a calibrated confidence. It runs on a CPU in a few milliseconds.

rc2 reads a turn as the Realtime Context Standard does: the emotion its words show, and sarcasm with the agent's line before it as context, not as the turn. rc1 read a turn that was sarcastic on its own words as sincere when the agent's line before it was plain: "Well, that's just great, isn't it? It's gonna be late." after "It's still processing, so it hasn't shipped yet." read happy, sarcasm 0.07. rc2 reads it angry, sarcasm 0.97. It still reads "Oh, wonderful." as sarcastic after "Your delivery is delayed again." and as sincere on its own or after good news.

```python
from pipecat_tone.reader import ToneReaderParams, ToneReaderProcessor

ToneReaderProcessor(params=ToneReaderParams(revision="reader-rc2"))  # precision="fp32" for full precision
```

## Output

On each final transcription, as its `perception`: the two likeliest of nine emotions (neutral, happy, amused, angry, sad, afraid, surprised, disgusted, contemptuous) with their confidences, and the chance the turn is sarcastic. `ToneReaderModel.tag()` also gives the emotion's intensity, the move (asks, answers, hedges, doesnt_know, withholds, closes, other) and whether the turn is a backchannel. The turn before sharpens the move and sarcasm: the processor reads each turn with the bot's reply before it.

## How well it reads calls

Labels are the majority vote of three large language models; none of these turns were trained on.

| | Real calls (1,500 turns) | Calls with more feeling (897 turns) |
|---|---|---|
| Emotion macro-F1 | 0.936 (neutral, happy) | 0.677 (all nine) |
| Move macro-F1 | 0.846 | 0.774 |
| Sarcasm | 1 of 1,496 sincere turns called sarcastic | F1 0.903; 0.9% of sincere turns called sarcastic |
| Calibration error: emotion, move, sarcasm | 0.077, 0.022, 0.015 | 0.027, 0.040, 0.010 |

It reads words, not voices: it can't hear tone of voice, laughter or a noisy line, and a flat transcript reads neutral.

## Against rc1

On turns of support calls, read with the agent's line before each, rc1 in brackets:

| | rc2 (rc1) |
|---|---|
| Sarcasm F1, support calls (3,752 turns, 35 domains) | 0.959 (0.903) |
| … after a plain agent line, with no news in it | 0.968 (0.909) |
| … in the three domains held out of training | 0.960 (0.906) |
| Sarcastic turns read as happy | 2.1% (18.7%) |
| Praise or thanks after a line it's meant after and one that makes it mean the opposite: pairs read right on both (365 pairs) | 94.0% (49.0%) |
| … at the 0.7 Pipecat's guidance acts at | 93.4% (35.6%) |
| The same turn alone called sarcastic | 0.3% (20.3%) |

On rc1's own test sets, its emotion and move are a little lower and none of it beyond the noise of a paired bootstrap: emotion macro-F1 0.661 (0.669) on the main test and 0.684 (0.724) on player speech, move 0.818 (0.831). Sarcasm F1 on the held-out sets is 0.848 (0.854), and it calls 1.0% (1.7%) of natural sincere lines sarcastic. On calls with more feeling, telling an answered question from an open one is lower beyond noise: 0.772 (0.803).

## Training

A 24.2M-parameter MiniLM-L6 encoder (`sentence-transformers/all-MiniLM-L6-v2`), the r3c recipe's run `tonetext-r3c-student-minilm-l6-seed2`, trained on 1,111,319 lines with the soft labels of rc1's five Ettin-1B teachers, then calibrated. Its lines are rc1's, and two sets written for rc2 by an LLM:

- **Support calls:** 30,000 turns of calls in 35 support domains. In each call, one line of the agent's (an update, a question, a next step, a kind word, bad news or good news) is answered twice, once sarcastically and once sincerely.
- **Contrast pairs:** 2,000 turns of praise or thanks, each after two of the agent's lines: one it is meant after, and one after which the same words mean the opposite.

A committee of three large language models labelled both under the standard's perception. In training, a line whose sarcasm only the agent's line carries never loses that line. `data/reader/` in `pipecat-ai/tone-models` holds what training reads, without third-party text, and a manifest of rc1's datasets, each under its own licence; `lab.reader.pipeline.fetch` downloads them and rebuilds the rest.

## Files

`reader/` in `pipecat-ai/tone-models` at `reader-rc2`:

| File | Bytes | sha256 |
|---|---|---|
| `config.json` | 1,291 | `844e73c5d914631642f99f577b5146c9023d1bf9a223d30550de0f5b96ec01c7` |
| `tokenizer.json` | 711,649 | `1b19820b21bee2ab9a5d3f2ac461f9517c4589536c7d751d60d38877993c5e8b` |
| `model_fp16.onnx` | 48,247,715 | `29d2e3839d806ab91fe64e9de14d13e7ca7251ec2fd44bcd37251200370a163c` |
| `model.opt.onnx` | 96,298,762 | `d96f05db2e1cc7b4559dec0d723fd68ec51e468f04da1d6b3a9ebecce4e0d8a9` |

The 16-bit file's emotion and move each differ from full precision's on one of 8,207 test lines, and its sarcasm on none. It runs at rc1's speed, on the same graph: timed in turns with rc1 on the same Apple M1 Max, it took the same time per turn on 1 thread and on 4. rc1's p95 per turn on 4 threads there: 5.0 ms with the turn before, 3.7 ms without; 11.9 ms on 1 thread. Licence: BSD-2-Clause.
