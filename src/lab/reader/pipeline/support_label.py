"""The committee's labels on r3's support-call set (``lab.reader.pipeline.support_generate``), asked under the Realtime
Context Standard's perception (``committee.perception_emotion`` and ``committee.perception_sarcasm``): emotion with its
intensity, the act toward the agent and backchannel, and sarcasm; the move is left to the teachers. Local results keep
counts, scores and costs only.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.support_label::pilot
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.support_label::label
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.support_label::collect
"""

import json
import random
import time
from collections import Counter

from lab.reader import sources, storage
from lab.reader.app import DATA, app, job, save
from lab.reader.committee import label as committee_label
from lab.reader.committee import perception_emotion, perception_sarcasm

STAGE = "support_label"
LINES_KEY = sources.support_key("lines.jsonl")
META_KEY = sources.support_key("meta.jsonl")
QUESTIONS = {"committee": perception_emotion, "sarcasm": perception_sarcasm}
BATCHED = ("committee", "sarcasm")  # A answers both by batch
SETS = dict(sources.SUPPORT_SETS)
PILOT_SETS = {q: f"{sources.SUPPORT_ID}-pilot-{q}" for q in SETS}
SAMPLES_KEY = sources.tonetext_labels_key(SETS["committee"], "samples.json")
WORK = committee_label.Work(f"{sources.SUPPORT_ID}-work", QUESTIONS, BATCHED)
KEYS = {"lines_key": LINES_KEY}
A_CHUNK = 3000  # lines per A batch
B_CONCURRENCY = 12  # B's two questions at once: 24 calls in flight
PILOT_PER_KIND = 3


def load() -> tuple[list, list]:
    """Every line of the set, and (custom id, input) for each."""
    from lab.reader.voice import CorpusLine

    return committee_label.load_lines(LINES_KEY, CorpusLine)


def pilot_ids(lines: list) -> list[int]:
    """PILOT_PER_KIND lines of each moment and answer kind, and of the person's other turns, from dev and train only."""
    rng = random.Random(611)
    by: dict[str, list[int]] = {}
    for i, ln in enumerate(lines):
        if ln.split != "test":
            by.setdefault(ln.generator, []).append(i)
    return sorted(i for g in sorted(by) for i in rng.sample(by[g], min(PILOT_PER_KIND, len(by[g]))))


def samples(lines: list, act_l: list, _xch_l, sar_l: list, n: int | str = 60) -> list[dict]:
    """About `n` lines, or "all", with their labels, every vote and what each was written to be."""
    meta = {m["line_id"]: m for m in storage.jsonl(META_KEY)}
    idx = list(range(len(lines)))
    picked = idx if n == "all" else sorted(random.Random(6111).sample(idx, min(int(n), len(idx))))
    out = []
    for i in picked:
        ln, e, s = lines[i], act_l[i], sar_l[i]
        act = next(iter(e["acts"].values()))
        out.append(
            {
                "line_id": ln.line_id,
                "split": ln.split,
                "generator": ln.generator,
                "by_construction": meta.get(ln.line_id, {}).get("sarcastic_by_construction"),
                "previous": ln.input.previous,
                "text": ln.input.text,
                "emotion": e["emotion"]["label"],
                "intensity": e["emotion"]["intensity"],
                "emotion_votes": e["emotion"]["votes"],
                "act": act["label"],
                "backchannel": e["backchannel"]["label"],
                "sarcastic": s["sarcasm"]["label"],
                "sarcasm_votes": s["sarcasm"]["votes"],
                "d_sarcasm_probability": s["sarcasm"]["d_probability"],
            }
        )
    return out


def construction_agreement(sar_l: list) -> dict:
    """The committee's sarcasm against what each line was written to be, by moment and answer kind: how many of each
    kind it calls sarcastic, sincere, or gives no majority."""
    meta = {m["line_id"]: m for m in storage.jsonl(META_KEY)}
    out: dict[str, Counter] = {}
    for r in sar_l:
        m = meta.get(r["line"]["line_id"], {})
        g = r["line"].get("generator") or ""
        lab = r["sarcasm"]["label"]
        out.setdefault(g, Counter())[
            "sarcastic" if lab is True else "sincere" if lab is False else "no_majority"
        ] += 1
        if m.get("sarcastic_by_construction") is not None and lab is not None:
            out.setdefault("_all", Counter())[
                "agree" if lab == m["sarcastic_by_construction"] else "disagree"
            ] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def summaries() -> dict:
    """The voice-agent set's summaries: every line is the person's."""
    return {
        "committee": lambda rows: committee_label.act_emotion_summary(rows),
        "sarcasm": lambda rows: {
            **committee_label.sarcasm_summary(rows),
            "by_construction": committee_label.guarded(construction_agreement, rows),
        },
    }


@job(DATA, cpu=2.0, memory=8192, timeout=2 * 3600)
def label_pilot(systems: dict, meta: dict) -> dict:
    """PILOT_PER_KIND lines of each kind through both questions synchronously, into the pilot's own sets; every pilot
    line's labels beside them."""
    storage.reload()
    lines, items = load()
    idx = pilot_ids(lines)
    lines, items = [lines[i] for i in idx], [items[i] for i in idx]
    answers = committee_label.ask_all(items, systems, committee_label.models(), questions=QUESTIONS)
    return committee_label.write_sets(
        PILOT_SETS,
        lines,
        items,
        answers,
        {},
        meta,
        sources.tonetext_labels_key(PILOT_SETS["committee"], "pilot_lines.json"),
        lambda ls, a, x, s: samples(ls, a, x, s, "all"),
        summaries(),
        batched=BATCHED,
    )


@job(DATA, cpu=4.0, memory=16384, timeout=24 * 3600, nonpreemptible=True)
def label_all(systems: dict, meta: dict, wait_s: float = 6 * 3600.0) -> dict:
    """Every line through both questions, checkpointed from the start: A's batches, sent once and waited for until the
    first sent plus `wait_s`, then what they left; B and D synchronously; then one more pass for any line a member left
    unanswered."""
    storage.reload()
    t0 = time.time()
    lines, items = load()
    models = committee_label.models()
    sent = WORK.send_once(items, systems, models, ("a",), wait_s, SETS, A_CHUNK)
    runners = committee_label.runners(systems, models, B_CONCURRENCY, QUESTIONS)
    batched = [(q, "a") for q in BATCHED]
    done, from_batch = WORK.answer_all(items, runners, batched, lambda: WORK.collect_batches(sent, items))
    retried = {}
    for k in runners:
        missing = sum(1 for cid, _ in items if done[k].get(cid, (None,))[0] is None)
        if missing:
            done[k], asked = WORK.ask_chunked(*k, items, runners[k], have=dict(done[k]))
            retried[f"{k[0]}-{k[1]}"] = {
                "missing": missing,
                "answered_now": sum(1 for r in asked.values() if r[0] is not None),
            }
    meta = {
        **meta,
        "seconds": round(time.time() - t0),
        "answers_from_a_batches": {q: len(from_batch[(q, "a")]) for q in BATCHED},
        "retried": retried,
        "a_batches": sent["a"],
    }
    ids = {q: {"a": sent["a"][q], "b": "synchronous"} for q in BATCHED}
    return committee_label.write_sets(
        SETS,
        lines,
        items,
        committee_label.answers(done, from_batch, ids, QUESTIONS),
        {"a": True},
        meta,
        SAMPLES_KEY,
        samples,
        summaries(),
        batched=BATCHED,
    )


def show(rows: list[dict]) -> None:
    """Labelled lines on the terminal, to read; never saved in the repo."""
    for r in rows:
        print(
            f"[{r['split']} {r['generator']} built {r['by_construction']}] {r['previous']}\n    -> {r['text']}\n"
            f"    emotion {r['emotion']}/{r['intensity']} {r['emotion_votes']} | act {r['act']} | backchannel "
            f"{r['backchannel']} | sarcastic {r['sarcastic']} {r['sarcasm_votes']} d={r['d_sarcasm_probability']}"
        )


def digest(res: dict) -> dict:
    """What the terminal shows of a summary."""
    return {
        **{k: res.get(k) for k in ("cost_usd_total", "seconds", "retried", "answers_from_a_batches")},
        **{
            q: {
                "errors": res[q]["errors"],
                "cost_usd": res[q]["cost_usd"],
                "a_b": res[q].get("a_b"),
                **({"by_construction": res[q].get("by_construction")} if q == "sarcasm" else {}),
            }
            for q in SETS
        },
    }


@app.local_entrypoint()
def pilot():
    """PILOT_PER_KIND lines of each kind, both questions, synchronously."""
    systems, meta = committee_label.prepare("pilot", QUESTIONS, **KEYS)
    res = label_pilot.remote(systems, meta)
    save(STAGE, "pilot.json", res)
    print(json.dumps(digest(res), indent=1, default=str))
    show(json.loads(storage.get(res["samples_key"])))


@app.local_entrypoint()
def label(wait_s: float = 6 * 3600.0):
    """Every line in one spawned call."""
    systems, meta = committee_label.prepare("label", QUESTIONS, **KEYS)
    call = label_all.spawn(systems, meta, wait_s=wait_s)
    save(STAGE, "label-spawned.json", {"call": call.object_id, **meta, "wait_s": wait_s})
    print("spawned", call.object_id)


@app.local_entrypoint()
def collect():
    committee_label.collect(STAGE, SETS, SAMPLES_KEY, show, digest)
