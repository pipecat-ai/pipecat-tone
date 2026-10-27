# Tone tagger: changelog

Trained here (`src/lab/`) on synthetic support calls: Claude writes each scenario, a deployment LLM holds the call through a real Pipecat pipeline, a Claude teacher labels every sentence without changing a word, and another vendor's model reviews each reply. Scores are on held-out test domains, against the teacher's labels, and on tone-bench.

## rc1 (2026-10-27)

Ettin encoders at 17M and 32M with six heads (delivery, sound before, sound after, slow spans, stress and breath), int8, on the CPU. Trained on corpora v0 to v3: 4,997 calls and 241,086 training sentences.

- **Corpora:** v0, 999 calls; v1, 2,000 with contrast cases, where the obvious rule is wrong (a code mentioned again in passing, a weak guess at frustration, sarcasm, a joke, bad news, a complaint repeated after an apology); v2, 1,000, half asking to breathe; v3, 1,000, each with a moment for a rare delivery (good news, heartfelt thanks, a bereavement, a firm reassurance, an odd problem, a surprise, a rush or a long read-out). In `pipecat-ai/tone-models` at `data/tagger/v0` to `v3`.
- **Context:** each sentence is read with the one before it, the caller's last line, the policy's guidance and the labels of the agent's last reply.
- **Breaths,** with the tagger's `BREATHE` preset: which replies breathe is drawn at people's rate for their length, from the breaths the ICSI Meeting Corpus's transcribers marked (statistics only), and the teacher picks only the sentence. A reply breathes at most once (`breaths_per_reply`), and a bias by sentence length holds the model to people's rate (`sound_bias`). On the held-out domains the 17M breathes in 22% of replies, as people do in turns that long, and a breathing reply follows a breathing one 1.13 times as often as chance (people: 1.17).
- **Breath kinds:** each breath is drawn as a light breath, a deep breath or an exhale at people's shares (59%, 36% and 5%; `breath_kinds`), seeded by the call so far, and named per TTS in `labels.yaml`: ElevenLabs `quick breath`, `deep breath` and `exhales softly`, picked by ear on Eleven v4 and v4 Turbo; MiniMax `breath`, `inhale` and `exhale`; Fish `inhaling`, `deep breath` and `exhaling`.
- **Averaged runs:** each size averages runs that start from the same weights (`--init-seed`), weight by weight (`lab.train.soup`): eight at 17M, which weigh rare deliveries up (`--delivery-balance`), and six at 32M. One run alone moves held-out scores by about 1.5 points.
- **Held-out test domains** (no breaths): the 17M scores 0.751, 0.759, 0.748 and 0.742 on v0's to v3's, and 0.753 over all 439 test calls; the 32M 0.757, 0.763, 0.783 and 0.789, and 0.767. Delivery macro F1: 0.640 and 0.646. The rules alone: 0.38 and 0.37 on v0's and v1's.
- **tone-bench** (phonellm-mini, judged by Claude Opus 5.5, no breaths): with the 17M, headline 90.9 and expression 97.1 (recall 100.0, precision 94.4) over two passes; with the 32M, 91.3 and 95.6. With the rules alone, 90.0 and 94.7; phonellm-mini alone, 57.9.
- **Speed:** 13.8 ms p50 per sentence on one Apple Silicon thread at 17M, and 37.1 ms at 32M; the int8 files are 63 MB and 115 MB.

Card: [rc1/README.md](rc1/README.md).
