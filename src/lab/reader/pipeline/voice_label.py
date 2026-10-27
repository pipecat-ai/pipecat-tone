"""The committee's labels on r1's voice-agent test set, which is never trained on; local results keep counts, scores
and costs only.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.voice_label::pilot
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.voice_label::label
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.voice_label::resume --batches <json> --deadline <unix time>
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.voice_label::collect
"""

import json
import random
import time

from lab.reader import sources, storage
from lab.reader.app import DATA, app, job
from lab.reader.committee import label as committee_label

STAGE = "voice_label"
LINES_KEY = sources.voice_key("lines.jsonl")
# About 40 lines with their labels.
SAMPLES_KEY = sources.voice_key("samples.json")
SETS = dict(sources.VOICE_GENERATED)
PILOT_SETS = {q: f"{sources.VOICE_ID}-pilot-{q}" for q in SETS}
WORK = committee_label.Work(f"{sources.VOICE_ID}-work")
KEYS = {"lines_key": LINES_KEY}


def load() -> tuple[list, list]:
    """Every line of the set, and (custom id, input) for each."""
    from lab.reader.voice import CorpusLine

    return committee_label.load_lines(LINES_KEY, CorpusLine)


def pilot_ids(lines: list) -> list[int]:
    """30 lines spread over what the set holds: sarcasm and look-alike seeds, marked moves, other lines of the person
    (half speech-to-text), the agent's lines (one opener)."""
    rng = random.Random(502)
    want = [
        ("sarcasm_seed", 6),
        ("lookalike_seed", 5),
        ("move_focus", 3),
        ("user_asr", 3),
        ("user_clean", 3),
        ("agent_opener", 1),
        ("agent", 9),
    ]

    def kind(ln) -> str:
        if "sarcasm_seed" in ln.slices:
            return "sarcasm_seed"
        if "lookalike_seed" in ln.slices:
            return "lookalike_seed"
        if (ln.generator or "").startswith("move_focus"):
            return "move_focus"
        if ln.input.speaker_kind == "player":
            return "user_asr" if "asr_style" in ln.slices else "user_clean"
        return "agent_opener" if ln.input.previous is None else "agent"

    by: dict[str, list[int]] = {}
    for i, ln in enumerate(lines):
        by.setdefault(kind(ln), []).append(i)
    out = []
    for k, n in want:
        pool = by.get(k, [])
        out += rng.sample(pool, min(n, len(pool)))
    return sorted(out)


def samples(lines: list, act_l: list, xch_l: list, sar_l: list, n: int | str = 40) -> list[dict]:
    """About `n` lines, or "all", with their three labels and every vote: sarcasm and look-alike seeds, other lines of
    the person, the agent's lines."""
    if n == "all":
        picked = list(range(len(lines)))
    else:
        rng = random.Random(5021)
        groups = {"sarcasm_seed": 10, "lookalike_seed": 8, "user": 10, "agent": 12}
        by: dict[str, list[int]] = {}
        for i, ln in enumerate(lines):
            k = (
                "sarcasm_seed"
                if "sarcasm_seed" in ln.slices
                else "lookalike_seed"
                if "lookalike_seed" in ln.slices
                else "user"
                if ln.input.speaker_kind == "player"
                else "agent"
            )
            by.setdefault(k, []).append(i)
        picked = sorted(
            i
            for k, m in groups.items()
            for i in rng.sample(by.get(k, []), min(m * n // 40, len(by.get(k, []))))
        )
    out = []
    for i in picked:
        ln = lines[i]
        e, x, s = act_l[i], xch_l[i], sar_l[i]
        act = next(iter(e["acts"].values()))
        out.append(
            {
                "line_id": ln.line_id,
                "slices": ln.slices,
                "generator": ln.generator,
                "previous": ln.input.previous,
                "speaker": ln.input.speaker,
                "to": ln.input.to,
                "text": ln.input.text,
                "emotion": e["emotion"]["label"],
                "intensity": e["emotion"]["intensity"],
                "emotion_votes": e["emotion"]["votes"],
                "backchannel": e["backchannel"]["label"],
                "act": act["label"],
                "act_votes": act["votes"],
                "move": x["exchange"]["label"],
                "move_votes": x["exchange"]["votes"],
                "sarcastic": s["sarcasm"]["label"],
                "sarcasm_votes": s["sarcasm"]["votes"],
                "d_sarcasm_probability": s["sarcasm"]["d_probability"],
            }
        )
    return out


@job(DATA, cpu=2.0, memory=8192, timeout=2 * 3600)
def label_pilot(systems: dict, meta: dict) -> dict:
    """About 30 lines through all three questions synchronously, into the pilot's own sets; every line's labels back."""
    storage.reload()
    lines, items = load()
    idx = pilot_ids(lines)
    lines, items = [lines[i] for i in idx], [items[i] for i in idx]
    answers = committee_label.ask_all(items, systems, committee_label.models())
    return committee_label.write_sets(
        PILOT_SETS,
        lines,
        items,
        answers,
        {},
        meta,
        sources.tonetext_labels_key(PILOT_SETS["committee"], "pilot_lines.json"),
        lambda ls, a, x, s: samples(ls, a, x, s, "all"),
    )


@job(DATA, cpu=2.0, memory=8192, timeout=8 * 3600, nonpreemptible=True)
def label_all(systems: dict, meta: dict, wait_s: float = 3600.0) -> dict:
    """Every line through all three questions: A and B by batch for act and emotion and the move, each batch sent
    once; everything else synchronously."""
    storage.reload()
    t0 = time.time()
    lines, items = load()
    models = committee_label.models()
    sent = WORK.send_once(items, systems, models, ("b", "a"), wait_s, SETS)
    batches = {q: {m: sent[m][q][0] for m in ("b", "a")} for q in committee_label.BATCHED}
    answers = committee_label.ask_all(
        items, systems, models, batches, max(0.0, sent["deadline"] - time.time())
    )
    meta = {**meta, "seconds": round(time.time() - t0)}
    return committee_label.write_sets(
        SETS, lines, items, answers, {"a": True, "b": True}, meta, SAMPLES_KEY, samples
    )


@job(DATA, cpu=2.0, memory=8192, timeout=8 * 3600, nonpreemptible=True)
def label_resume(systems: dict, meta: dict, batches: dict, deadline: float) -> dict:
    """Every line through all three questions from batches already sent, never new ones: D and sarcasm asked now, the
    batches collected at `deadline`, the rest asked synchronously; every synchronous answer checkpointed."""
    storage.reload()
    t0 = time.time()
    lines, items = load()
    runners = committee_label.runners(systems, committee_label.models())
    batched = [(q, m) for q in committee_label.BATCHED for m in ("a", "b")]
    done, from_batch = WORK.answer_all(
        items, runners, batched, lambda: committee_label.collect_sent(batches, items, deadline)
    )
    ids = {q: {"a": batches[f"a_{q}"], "b": "identified by answers"} for q in committee_label.BATCHED}
    meta = {
        **meta,
        "seconds": round(time.time() - t0),
        "resumed": True,
        "answers_from_batches": {f"{q}-{m}": len(from_batch[(q, m)]) for q, m in batched},
    }
    answers = committee_label.answers(done, from_batch, ids)
    return committee_label.write_sets(
        SETS, lines, items, answers, {"a": True, "b": True}, meta, SAMPLES_KEY, samples
    )


def show(rows: list[dict]) -> None:
    """Labelled lines on the terminal, to read; never saved in the repo."""
    for r in rows:
        print(
            f"[{r['generator']}] {r['previous']}\n    -> {r['speaker']}: {r['text']}\n    emotion {r['emotion']} "
            f"{r['emotion_votes']} | move {r['move']} {r['move_votes']} | sarcastic {r['sarcastic']} {r['sarcasm_votes']}"
        )


def digest(res: dict) -> dict:
    """What the terminal shows of a summary."""
    return {q: {k: res[q].get(k) for k in ("errors", "cost_usd", "a_b")} for q in SETS}


@app.local_entrypoint()
def pilot():
    """About 30 lines, all three questions, synchronously."""
    committee_label.pilot(STAGE, label_pilot, KEYS, show, digest)


@app.local_entrypoint()
def label(wait_s: float = 3600.0):
    """Every line in one spawned call: a batch not finished after `wait_s` is cancelled, what it finished kept and the
    rest asked synchronously, once."""
    committee_label.spawn(STAGE, "label", label_all, KEYS, wait_s=wait_s)


@app.local_entrypoint()
def resume(batches: str, deadline: float):
    """Finishes from batches already sent, {"a_committee", "a_exchange", "b": [two ids]}, waiting for them until
    `deadline`."""
    committee_label.spawn(STAGE, "resume", label_resume, KEYS, batches=json.loads(batches), deadline=deadline)


@app.local_entrypoint()
def collect():
    committee_label.collect(STAGE, SETS, SAMPLES_KEY, show, digest)
