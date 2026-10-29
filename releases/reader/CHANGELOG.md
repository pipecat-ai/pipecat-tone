# Tone reader: changelog

The tone reader was Pipecat Tone, published as `pipecat-ai/tone`. Its runtime and training now live here: `src/pipecat_tone/reader/` and `src/lab/reader/`.

## rc2 (2026-10-28)

Reads sarcasm with the agent's line as context, as the Realtime Context Standard does. rc1 read a turn sarcastic on its own words as sincere after a plain agent line; rc2 doesn't, and it keeps reading praise as sarcastic after bad news and as sincere after good news. The r3c recipe's run `tonetext-r3c-student-minilm-l6-seed2`, the same MiniLM-L6 encoder as rc1, trained on 1,111,319 lines with the soft labels of rc1's five teachers, then calibrated. `reader-rc2` is the runtime's default now; `revision="reader-rc1"` keeps rc1.

- **Support calls** (3,752 turns in 35 domains, read with the agent's line): sarcasm F1 0.959 (rc1 0.903), and 2.1% of sarcastic turns read as happy (rc1 18.7%).
- **Contrast pairs** (365 pairs of a praise or thanks turn after a line it's meant after and one that makes it mean the opposite): both read right in 94.0% of pairs (rc1 49.0%), 93.4% at Pipecat's 0.7 (rc1 35.6%).
- **The cases that showed rc1's misread:** 10 of 10 right (rc1 8), and 35 of 35 hand-written praise cases (rc1 30).
- **Real calls:** emotion macro-F1 0.936 over neutral and happy, move macro-F1 0.846, and 1 of 1,496 sincere turns called sarcastic.
- **Calls with more feeling:** emotion macro-F1 0.677 over all nine, move 0.774, sarcasm F1 0.903, and 0.9% of sincere turns called sarcastic.
- **A little lower:** on rc1's own test sets, emotion and move by up to 0.04, none of it beyond noise; on calls with more feeling, telling an answered question from an open one, 0.772 (rc1 0.803), beyond noise.
- **Training data:** two sets written for rc2, 30,000 turns of support calls in which one agent line is answered once sarcastically and once sincerely, and 2,000 contrast pairs, labelled by the committee under the standard's perception. Both are in `data/reader/`, with rc2's run.
- **The 16-bit file:** half the download; its emotion and move each differ from full precision's on one of 8,207 test lines.
- **Speed:** rc1's. It is the same graph, and timed in turns with rc1 on the same Mac it took the same time per turn.

Card: [rc2/README.md](rc2/README.md).

## rc1 (2026-10-16)

Pipecat Tone rc1, moved here unchanged: the files in `reader/` at `reader-rc1` are rc1's, byte for byte. A 24.2M-parameter MiniLM-L6 encoder, the r2 recipe's run `tonetext-r2-student-minilm-l6-seed1`, trained on 1,083,920 lines with the soft labels of five Ettin-1B teachers, then calibrated.

- **Real calls** (1,500 turns from held-out conversations of nine openly licensed support and assistant datasets): emotion macro-F1 0.940 over neutral and happy, move macro-F1 0.843 over asks, answers, closes and other, and 1 of 1,496 sincere turns called sarcastic.
- **Calls with more feeling** (897 written turns from 299 calls): emotion macro-F1 0.598 over all nine, move 0.800, sarcasm F1 0.887, and 1.0% of sincere turns called sarcastic.
- **Speed:** p95 5.0 ms per turn with the turn before on 4 threads (Apple M1 Max), 11.9 ms on 1.
- **The 16-bit file:** half the download, the same tags on all but 2 of 8,207 test lines.
- **Runtime:** `ToneReaderProcessor`, ported from Pipecat's `ToneProcessor`, which Pipecat drops before #6319 merges. It reads exactly as Pipecat's runtime did, in both precisions, and loads its model after the pipeline starts.
- **Training:** the pipeline, ported to `src/lab/reader/`, reproduces rc1: the export rebuilds its files byte for byte, and the scorecard in `results/reader/` matches the original value for value.

Card: [rc1/README.md](rc1/README.md).
