# Tone reader: changelog

The tone reader was Pipecat Tone, published as `pipecat-ai/tone`. Its runtime and training now live here: `src/pipecat_tone/reader/` and `src/lab/reader/`.

## rc1 (2026-10-16)

Pipecat Tone rc1, moved here unchanged: the files in `reader/` at `reader-rc1` are rc1's, byte for byte. A 24.2M-parameter MiniLM-L6 encoder, the r2 recipe's run `tonetext-r2-student-minilm-l6-seed1`, trained on 1,083,920 lines with the soft labels of five Ettin-1B teachers, then calibrated.

- **Real calls** (1,500 turns from held-out conversations of nine openly licensed support and assistant datasets): emotion macro-F1 0.940 over neutral and happy, move macro-F1 0.843 over asks, answers, closes and other, and 1 of 1,496 sincere turns called sarcastic.
- **Calls with more feeling** (897 written turns from 299 calls): emotion macro-F1 0.598 over all nine, move 0.800, sarcasm F1 0.887, and 1.0% of sincere turns called sarcastic.
- **Speed:** p95 5.0 ms per turn with the turn before on 4 threads (Apple M1 Max), 11.9 ms on 1.
- **The 16-bit file:** half the download, the same tags on all but 2 of 8,207 test lines.
- **Runtime:** `ToneReaderProcessor`, ported from Pipecat's `ToneProcessor`, which Pipecat drops before #6319 merges. It reads exactly as Pipecat's runtime did, in both precisions, and loads its model after the pipeline starts.
- **Training:** the pipeline, ported to `src/lab/reader/`, reproduces rc1: the export rebuilds its files byte for byte, and the scorecard in `results/reader/` matches the original value for value.

Card: [rc1/README.md](rc1/README.md).
