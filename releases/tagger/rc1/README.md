# Tone tagger rc1

Voices an LLM's plain reply. For each sentence as it streams in, it chooses a delivery, slow spans for details, stressed words and sounds, and writes them as Pipecat's expression markup in the call's TTS names. Asked to, it breathes as people do: in about as many replies, and as a light breath, a deep one or now and then an exhale. An Ettin encoder with six heads, on the CPU, at 17M (the default) and 32M.

```python
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat_tone.tagger import BREATHE, ToneTaggerParams, ToneTaggerProcessor

guidance = [ExpressionGuidance.MATCH_STYLE, ExpressionGuidance.SLOW_FOR_DETAILS, BREATHE]
policy = ExpressionPolicy(guidance=guidance)
# size="32m" for the larger
ToneTaggerProcessor(context=context, policy=policy, params=ToneTaggerParams(revision="tagger-rc1"))
```

## Labels

Deliveries: calm, sympathetic, apologetic, warm, excited, surprised, sad, grateful, curious, confident, slow and fast. Sounds: laugh, chuckle, sigh and breath. `labels.yaml` names each for Cartesia, ElevenLabs, Fish and MiniMax; another TTS's labels are matched on the names it declares, and a `vocabulary` can name them in your own words. The policy narrows what's offered, as it does for an LLM.

Breath is offered only with the `BREATHE` preset in the policy's guidance, on a TTS that has one. Each breath is drawn as a light breath, a deep breath or an exhale, in the shares of the breaths ICSI's speakers took to speak: 59%, 36% and 5% (a deep one lasts over 0.6 s). The draw is seeded by the call so far, so a call replays alike. `labels.yaml` names the kinds for each TTS:

| | Light | Deep | Exhale |
|---|---|---|---|
| ElevenLabs | `quick breath` | `deep breath` | `exhales softly` |
| MiniMax | `breath` | `inhale` | `exhale` |
| Fish | `inhaling` | `deep breath` | `exhaling` |

Cartesia has no breath. ElevenLabs' names were picked by ear from twelve candidates on Eleven v4 and v4 Turbo, sent through Text-to-Dialogue as Pipecat's v4 service sends them; MiniMax's and Fish's aren't checked by ear.

## Training

Corpora v0 to v3: 4,997 synthetic support calls and 241,086 training sentences. Claude writes each scenario, a deployment LLM holds the call through a real Pipecat pipeline, a Claude teacher labels every sentence without changing a word, and another vendor's model reviews each reply. v1 adds contrast cases, where the obvious rule is wrong; v2 adds calls that ask to breathe; v3 gives every call a moment for a rare delivery. Each sentence is read with the one before it, the caller's last line, the policy's guidance and the labels of the agent's last reply.

Which replies breathe is drawn at people's rate for their length, from the breaths the ICSI Meeting Corpus's transcribers marked (statistics only); the teacher chooses only the sentence each breath starts. Breath has its own head, trained only where breath is offered. The config caps a reply at one breath (`breaths_per_reply`), shifts the breath head by sentence length (`sound_bias`) and gives the kinds' shares (`breath_kinds`).

Each size averages runs that started from the same weights: eight at 17M, which weigh rare deliveries up in training, and six at 32M. On dev, the 17M's runs score 0.711 to 0.721 alone and 0.721 averaged; the 32M's score 0.721 to 0.731 alone and 0.735 averaged.

## Breaths

On the held-out test domains of corpora v0 to v2, in replies that offer breath:

| | Replies that breathe | Replies of 6–15 words | 16–30 | 31–60 | Over 60 | A breathing reply right after another, against chance |
|---|---|---|---|---|---|---|
| **17M** | **22%** | 6% | 15% | 25% | 44% | 1.13 times |
| **32M** | **19%** | 7% | 12% | 24% | 35% | 1.18 times |
| People (ICSI), turns that long | 22% | 7% | 15% | 26% | 41% | 1.17 times |

## Scores

Held-out test domains, combined score against the teacher's labels, without breaths. "All" pools the 439 test calls.

| | v0 domains | v1 domains | v2 domains | v3 domains | All | p50 per sentence, 1 thread |
|---|---|---|---|---|---|---|
| **17M** | 0.751 | 0.759 | 0.748 | 0.742 | **0.753** | 13.8 ms |
| **32M** | 0.757 | 0.763 | 0.783 | 0.789 | **0.767** | 37.1 ms |

The rules alone (`rules=True`) score 0.38 and 0.37 on v0's and v1's domains. Delivery macro F1 over all test calls is 0.640 at 17M and 0.646 at 32M. Laughs, chuckles and sighs score 0.596 at 17M, which puts in a tenth fewer than the teacher, and 0.661 at 32M. v3's test calls are 50, with 5 to 17 lines of each rare delivery, so its column moves with a handful of sentences. Times are on an Apple Silicon Mac, in one session.

tone-bench, phonellm-mini, judged by Claude Opus 5.5, without breaths:

| | Headline | Expression | Recall | Precision | Passes |
|---|---|---|---|---|---|
| **With the 17M** | **90.9** | **97.1** | 100.0 | 94.4 | 2 |
| With the 32M | 91.3 | 95.6 | 98.4 | 92.9 | 2 |
| With the rules alone | 90.0 | 94.7 | 100.0 | 90.0 | 1 |
| With the teacher labelling whole replies | 89.5 | 88.7 | 88.7 | 88.7 | 1 |
| phonellm-mini alone | 57.9 | 0.0 | 0.0 | | 1 |

With the 17M, laughs come on cue every time and 92% of moods fit; with the 32M, 65% and 94%.

## Files

`tagger/` in `pipecat-ai/tone-models` at `tagger-rc1`:

| File | Bytes | sha256 |
|---|---|---|
| `17m/config.json` | 1,993 | `ff26cfab6f4002e716385e5d6443d3777cb7aa89f0de43c987567cf3f248e4ac` |
| `17m/model.int8.onnx` | 63,370,447 | `a42f5e1d930555036769d21adec4f94d4a02f33dcb1f251e66635a23608ddd55` |
| `17m/tokenizer.json` | 3,583,228 | `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30` |
| `32m/config.json` | 1,951 | `9121e3fa537955e9c51626e19a8cde3ac91fd69eb4cc56b6d3ae880fd61da2c8` |
| `32m/model.int8.onnx` | 114,707,020 | `0d98c480ed049b6e5b5e53125261a3e01f0b23a1f659a433b4537f32da00b7a5` |
| `32m/tokenizer.json` | 3,583,228 | `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30` |

Its training data is `data/tagger/v0` to `v3` in the same repo. Licence: BSD-2-Clause.
