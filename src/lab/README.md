# lab

How the tone models are trained and measured. Not shipped: the package is `src/pipecat_tone/`.

| Path | What it holds |
|---|---|
| `reader/` | The reader's pipeline: open sets, speech twins, committee labels, teachers, students, calibration, export and scoring |
| `corpus/` | The tagger's corpus: specs, scenario writer, call builder, teacher, reviewer, rows |
| `train/` | The tagger's model inputs and targets, fit loop, ONNX export and scoring |
| `bench/` | tone-bench runs with a tagger, without changing tone-bench |
| `jobs/` | The tagger's Modal jobs |
| `release.py` | Releases to Hugging Face, and checks of them in a fresh cache |

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

Keys go in `.env`, and so do model ids, which are read from the environment and never written down: `LABELLER_B_MODEL` is the Claude model that writes scenarios and teaches, `LABELLER_D_MODEL` the reviewer's, and the reader's labelling committee uses both with `LABELLER_A_MODEL`.

tone-bench, Pipecat's benchmark for expressive conversation, is private. Building the tagger's corpus, benchmark runs (`bench`, `jobs.bench`, `jobs.offline`, `jobs.rejudge`) and the contamination check import it read-only from `../tone-bench` (or `$TONE_BENCH`). Everything else works without it: the runtime, the reader's pipeline, and training the tagger on the published rows (put `data/tagger/` from Hugging Face on the corpus volume and pass `--skip-export`); the few tests that need it are skipped. Pipecat is pinned to the commit tone-bench pins; move both together.

Jobs run on Modal in the daily workspace's `tone-tagger` environment: pass `MODAL_PROFILE=daily` on each command rather than switching your profile, and `--detach` after `modal run` to keep a long job going. Keys come from the `tone-tagger-keys` secret; refresh it after changing `.env`:

```bash
MODAL_PROFILE=daily uv run modal secret create tone-tagger-keys --from-dotenv .env --env tone-tagger --force
```

## The reader

Data lives on the `tone-reader-data` volume, each file at its key. Hugging Face holds what rc1's training and checks read, under `data/reader/` and without third-party text: the files that are ours, and stand-ins for those that mix our lines with others'. `pull` copies that folder to the volume, then `fetch` downloads the original sources at their pins and rebuilds the rest, the open sets included, writing each file only if its sha256 is the one in `rebuild/manifest.json`:

```bash
MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.pull
MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.fetch
```

`lab.reader.rebuild` defines the stand-ins, and `lab.reader.publish_data` makes the published folder from a full copy of the data. A stage runs one step per entrypoint, and its docstring lists them:

```bash
MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.<stage>[::<step>]
```

- **r1,** the round before rc1's, whose own data isn't published: `voice_generate` and `voice_label` (its generated voice-agent test set), `sarcasm_teacher`, then `train`'s r1 steps.
- **r2, rc1's recipe:** `open_ingest` (14 open sets: 13 to train on, and TopicalChat's test split), `speech` (noisy twins), `r2_select` and `r2_label` (committee labels), `teacher` (five Ettin-1B teachers), then `train`'s r2 steps.
- **r3, a candidate after rc1:** r2 with `support_generate` (the support-call set below), `support_label` (its committee labels), `teacher --recipe r3` (five teachers on r2's committee labels and the set's), then `train::spawn_r2 --recipe r3`.
- **r3c, r3 with contrast pairs:** r3's mix with the contrast pairs below, read with r2's teachers: `r3c_chain::run` writes, labels and trains it in one detached job.
- **Every run:** `export` (calibration and ONNX), `score`, `compare`, `precision` (the 16-bit file) and `latency` (timing on this machine).

rc1 is `train::spawn_r2 --seed 1`, exported with `export --run-id tonetext-r2-student-minilm-l6-seed1 --recipe r2 --sarcasm-cap 0.015`. This pipeline reproduces it exactly: the export rebuilds its files byte for byte, and `results/reader/` holds its scorecard. Counts and scores, never line text, go to `results/reader/<stage>/`.

rc2 is r3c's seed 2, `tonetext-r3c-student-minilm-l6-seed2` (`r3c_chain::run`, exported with `export --recipe r3c --sarcasm-cap 0.015`): of the two seeds, it calls the fewest sincere lines sarcastic. `results/reader/` holds its comparisons with rc1 and r3, its scorecard, its 16-bit file's measures and its timing.

**The support-call set (r3).** rc1 reads a turn with the agent's line before it, as it was trained, and that line is how it catches sarcasm aimed at news: "Oh, wonderful." after "Your delivery is delayed again." But after a plain line it reads a turn that is sarcastic on its own words as sincere: "Well, that's just great, isn't it? It's gonna be late." after "It's still processing, so it hasn't shipped yet." reads happy 0.82, sarcasm 0.07, against angry 0.53, sarcasm 0.65 on its own. In its training, sarcasm nearly always answered news. So the set's calls, written by the voice-agent set's teacher over 35 support domains, each have a moment where one line of the agent's (a plain update, a routine question, a next step, a kind word, bad news or good news) is answered twice, once sarcastically and once sincerely, and both answers are kept. Its committee is asked under the Real-time Context Standard's perception (`committee/guides/perception-guide-0.1.md`): the emotion the turn's words show, and whether it is sarcastic, with the agent's line as context, not the turn. Lines are split by call, three domains held out whole for test, and no training line repeats an evaluation line's text or pair or one of the cases above (`support.PROBES`). Each line keeps the call's turns before it (`meta.jsonl`), for a reader that reads more of the call. `compare` reads the set's test split, by the committee and as written, and the cases, beside r2's sets. r3 is calibrated as rc1 is: `export --recipe r3` fits on the dev lines r2 fits on. `r3_chain::run` runs every step after the calls in one detached job, and `r3_chain::no_move` a student whose move head is kept, so Pipecat's runtime loads it, but never trained (`spawn_r2 --no-move`), its epoch chosen on emotion and sarcasm: the standard reads only those, so it tests whether the move costs them anything.

**The contrast pairs (r3c).** r3 fixed that misread but taught two cues of its own. In the support-call set's training lines a turn opening on a praise word is sarcastic four times in five, even after good news, so r3 reads a bare "Oh, wonderful." as sarcastic on its own. And thanks with "helpful" is sincere nine times in ten, so it reads "Thanks, that's really helpful." after "I'm sorry, I can't help with that." as sincere. Training also dropped the agent's line from a fifth of the lines and kept their labels, so a line sarcastic only after news taught that its words alone were sarcastic. Each contrast pair is one turn of praise or thanks, written to say nothing of how it is meant, after two of the agent's lines: one it is meant after (good news, the thing done, the answer, a favour) and one after which the same words mean the opposite (a refusal, bad news, being passed on, no answer). Both lines are kept, so only the agent's line tells them apart. The committee labels them as it labels the support-call set (`support_label --set-id tonetext-support-contrast-0.1`). In training, a line written to be sarcastic only after the agent's line never loses that line (`data.NEEDS_PREVIOUS`, `model.kept_previous`). The pairs keep out of the cases and the praise cases (`support.PRAISE_PROBES`), and their dev and test pairs keep out of the support set's training turns. `compare` reads their test split: pairs right on both lines, by each reader's call and at the 0.7 Pipecat's demo acts at, and each pair's turn read alone. r3c reads r2's teachers' labels, as r3's best student did.

## The tagger

**Corpus.** For each seeded spec, Claude writes a scenario, a deployment LLM holds the call plain, the teacher labels every sentence, and another vendor's model reviews each reply. A call costs about $0.15. After a teacher or rubric change, bump its version and run `reteach_corpus` or `rereview_corpus`; `reteach_corpus` also redoes calls whose teacher failed. v3 (seeds 4000 on) plants in every call a situation that calls for a delivery the teacher seldom uses, such as heartfelt thanks, a loss or firm reassurance, as rare deliveries are where the tagger is weakest.

**Breaths.** v2 (seeds 3000 on) asks half its calls to breathe with the `BREATHE` preset. `breathe_corpus` gives every call taught without it a breath variant: the same call with `BREATHE`. Its breathing replies are drawn at people's rate for their length (ICSI's, below), the teacher chooses only the sentence each breath starts, and the call's verdicts stand. Asked for a rate instead, the teacher breathed by its own sense of where breaths belong, and the reviewer failed a third of the drawn breaths as unneeded, both in long replies far more than people do. A call that asks to breathe takes the pass's breaths in place of its teacher's. So every breath in the corpus is drawn the same way, and 22% of replies breathe, by length as people's do. Export writes a variant's rows beside the call's own, so the model learns what the preset changes and that nothing else does. `--trial` writes the pass's calls under `trials/` and leaves the batch as it is.

```bash
MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.jobs.corpus::build_corpus --batch v2 --start 3000 --n 1000
```

```bash
MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.jobs.corpus::breathe_corpus --batch v0
```

**Breaths in real speech.** `lab.corpus.speech_breaths` measures where people breathe from the ICSI Meeting Corpus's transcripts (CC BY 4.0), whose transcribers marked the breaths they heard on close-talking microphones: 34,933 turns, used for statistics only (`results/breaths/icsi.json`). People breathe before 4 to 6% of short sentences, and before or inside 17 to 19% of those of 21 words or more. The teacher never breathes before short ones, and breathes before 35% of long ones. `lab.train.calibrate --speech --per-reply 1` caps a reply at one breath and holds the teacher's breaths to people's rate for each length of sentence, without forcing the ones the teacher never takes: fitted to people's rate there, the model put them in sign-offs ("Take care"). So calibrated, rc1's 17M breathes in 22% of test replies, as people do in turns that long. `lab.train.evaluate.breaths` runs a tagger as calls run and summarises it the same way. An audio classifier found no breaths in HarperValleyBank's 8 kHz phone audio. A breath is light, deep or exhaled: 59%, 36% and 5% of the breaths people take to speak (`speech_breaths.breath_kinds`), which don't depend on how long the speech after them is. So a model chooses only breath, and with `calibrate --kinds` in its config the tagger draws each breath's kind at those shares, among the names `labels.yaml` gives the call's TTS.

```bash
uv run python -m lab.corpus.speech_breaths
```

```bash
uv run python -m lab.train.calibrate models/<folder> <dev.jsonl> ... --speech --per-reply 1 --kinds
```

**Training.** Exports the batches to rows, trains on an L4, keeps the best dev epoch and scores the ONNX export through the runtime. The training is spawned, so with `--detach` it outlives the client. `--cut-call-first` cuts the call rather than the label space when a pair is over budget, `--seed` another seed, `--layout '{"previous_tags": true}'` reads the agent's last reply's labels, `--speech-breaths --breaths-per-reply 1` calibrates breaths to real speech, `--delivery-balance` weighs rare deliveries up, and `::export_batches` exports without training. Breath has a head of its own, as the corpus draws which replies breathe, and checkpoints are chosen without breaths. Seeds vary by about 1.5 points, so `--init-seed` starts seeds from the same weights and `lab.train.soup` averages them into one model; seeds whose heads started apart average into a broken one.

rc1's 17M averages eight such runs, `--seed 0` to `7`; its 32M averages six on `jhu-clsp/ettin-encoder-32m`, without `--delivery-balance`. Each average is then calibrated as above.

```bash
MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.jobs.train::train_tagger --batch v0,v1,v2,v3 --backbone jhu-clsp/ettin-encoder-17m --name ettin17-v3-bal-s0 --skip-export --seed 0 --init-seed 0 --layout '{"previous_tags": true}' --speech-breaths --breaths-per-reply 1 --delivery-balance
```

```bash
uv run --group train python -m lab.train.soup models/ettin17-v3-bal-soup models/ettin17-v3-bal-s0 ... models/ettin17-v3-bal-s7
```

**Benchmark.** `--tagger` takes `noop`, `rules`, `model` or `module:Class`. Runs land in `results/runs/` and are logged in `results/runs.jsonl` with every commit involved. `lab.jobs.offline` runs the teacher offline, and `lab.jobs.rejudge` grades finished runs with another vendor's model. `python -m lab.bench.leaderboard` ranks the newest run of each tagger beside tone-bench's models in `results/leaderboard.md`, leaving tone-bench as it is; that file isn't kept in git, as it holds tone-bench's own results.

```bash
MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.jobs.bench --model phonellm-mini --tagger model --tagger-model ettin17-v3-bal-soup --passes 1
```

**Releases.** `python -m lab.release model tagger-rc1 tagger/17m=<folder> ...` commits a release with its manifest and tags it, leaving folders it isn't given as they were; `python -m lab.release check tagger-rc1` downloads it into a fresh cache through the runtime and checks every file.

Guards against tuning to tone-bench:

- The domain families tone-bench tests are left out of the corpus, and rows are split by domain. Checkpoints are chosen on dev.
- The scenario writer never sees tone-bench; `corpus.contamination` reports each batch's closest calls (about 0.5% 5-gram overlap on v0).
- The reviewer is another vendor's model, with a rubric from Pipecat's guidance.
- `bench/agent.py` copies tone-bench's `run_call` with the tagger added, `bench.results.score_call` builds turns as its `run_pass` does, and `jobs/rejudge.py` builds checks as its judge does; keep them in step.

Known gaps:

- **Teacher v2 was benchmark-influenced**: slowing every mention of a detail came from reading baseline E's misses. It's a fair reading of `SLOW_FOR_DETAILS`, but it helps tone-bench's every-mention check. Teacher v3's changes came from the reviewer.
- **Read-backs**: the teacher leaves a detail plain when the caller has just read it back, and the reviewer fails those replies, so they drop out of training.
- **Pace**: once "slow down" has slowed the TTS, the tagger still slows details on top. ElevenLabs' Dialogue service has no speed setting, so pace requests are ignored there.
- **Readings**: the reader's nine emotions are folded into the corpus's four tones before the tagger reads them.
- **Truncation**: an over-budget pair (about 4% of sentences, the long ones) loses the start of its longer text, often the label space, which decoding enforces anyway. Cutting the call first instead (`Layout.cut_call_first`) scored no better: 0.640 against 0.654 on the 701 over-budget test sentences.
- The turn pass runs when the LLM starts its reply, not when the caller's turn ends.
- **Context**: the tagger reads the caller's last line, the sentence before and state hints, and with `previous_tags` its own last reply's labels, not the words of its earlier replies. Without those labels, a breathing reply follows a breathing one 1.4 times as often as chance would have it on the test domains. With them, rc1's 17M is at 1.13, near ICSI's speakers (1.17), but a single seed has stayed at 1.4: the labels help, though not reliably. Reading the last three messages and two earlier sentences brings it to chance at 17M and at 32M, but takes 1.8 times as long at 17M. A mood carries over only through the hints.
- **Calibration's context**: calibration reads the teacher's last reply, as training rows do, while a call reads the model's own. So run as calls, a model breathes about 5% more than it was calibrated to.
- **Breaths inside sentences**: 38% of people's breaths fall inside a sentence, at a clause; the tagger breathes only before one. ICSI's transcribers marked breaths within an utterance only some of the time, so its rates are a floor.
