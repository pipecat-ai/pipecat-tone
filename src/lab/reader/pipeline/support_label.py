"""The committee's labels on r3's support-call set (``lab.reader.pipeline.support_generate``), and on r3c's contrast
pairs, asked under the Realtime Context Standard's perception (``committee.perception_emotion`` and
``committee.perception_sarcasm``): emotion with its intensity, the act toward the agent and backchannel, and sarcasm;
the move is left to the teachers. Every step takes the set's id, the support set's by default. Local results keep
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
QUESTIONS = {"committee": perception_emotion, "sarcasm": perception_sarcasm}
BATCHED = ("committee", "sarcasm")  # A answers both by batch
A_CHUNK = 3000  # lines per A batch
B_CONCURRENCY = 12  # B's two questions at once: 24 calls in flight
# Lines of each kind (each moment and answer kind, or each contrast side and turn kind) in a set's pilot.
PILOT_PER_KIND = {sources.SUPPORT_ID: 3, sources.CONTRAST_ID: 8}


def sets_of(set_id: str) -> dict:
    return {q: f"{set_id}-{q}" for q in QUESTIONS}


def pilot_sets_of(set_id: str) -> dict:
    return {q: f"{set_id}-pilot-{q}" for q in QUESTIONS}


def lines_key_of(set_id: str) -> str:
    return sources.support_key("lines.jsonl", set_id)


def meta_key_of(set_id: str) -> str:
    return sources.support_key("meta.jsonl", set_id)


def samples_key_of(set_id: str) -> str:
    return sources.tonetext_labels_key(sets_of(set_id)["committee"], "samples.json")


def keys_of(set_id: str) -> dict:
    return {"lines_key": lines_key_of(set_id)}


def work_of(set_id: str):
    return committee_label.Work(f"{set_id}-work", QUESTIONS, BATCHED)


LINES_KEY = lines_key_of(sources.SUPPORT_ID)
META_KEY = meta_key_of(sources.SUPPORT_ID)
SETS = sets_of(sources.SUPPORT_ID)
PILOT_SETS = pilot_sets_of(sources.SUPPORT_ID)
SAMPLES_KEY = samples_key_of(sources.SUPPORT_ID)
KEYS = keys_of(sources.SUPPORT_ID)


def load(set_id: str = sources.SUPPORT_ID) -> tuple[list, list]:
    """Every line of the set, and (custom id, input) for each."""
    from lab.reader.voice import CorpusLine

    return committee_label.load_lines(lines_key_of(set_id), CorpusLine)


def pilot_ids(lines: list, per_kind: int = PILOT_PER_KIND[sources.SUPPORT_ID]) -> list[int]:
    """`per_kind` lines of each kind the set's lines were written as, from dev and train only."""
    rng = random.Random(611)
    by: dict[str, list[int]] = {}
    for i, ln in enumerate(lines):
        if ln.split != "test":
            by.setdefault(ln.generator, []).append(i)
    return sorted(i for g in sorted(by) for i in rng.sample(by[g], min(per_kind, len(by[g]))))


def samples(
    lines: list, act_l: list, _xch_l, sar_l: list, n: int | str = 60, meta_key: str = META_KEY
) -> list[dict]:
    """About `n` lines, or "all", with their labels, every vote and what each was written to be."""
    meta = {m["line_id"]: m for m in storage.jsonl(meta_key)}
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


def construction_agreement(sar_l: list, meta_key: str = META_KEY) -> dict:
    """The committee's sarcasm against what each line was written to be, by moment and answer kind: how many of each
    kind it calls sarcastic, sincere, or gives no majority."""
    meta = {m["line_id"]: m for m in storage.jsonl(meta_key)}
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


def summaries(meta_key: str = META_KEY) -> dict:
    """The voice-agent set's summaries: every line is the person's."""
    return {
        "committee": lambda rows: committee_label.act_emotion_summary(rows),
        "sarcasm": lambda rows: {
            **committee_label.sarcasm_summary(rows),
            "by_construction": committee_label.guarded(construction_agreement, rows, meta_key),
        },
    }


@job(DATA, cpu=2.0, memory=8192, timeout=2 * 3600)
def label_pilot(systems: dict, meta: dict, set_id: str = sources.SUPPORT_ID) -> dict:
    """PILOT_PER_KIND lines of each kind through both questions synchronously, into the pilot's own sets; every pilot
    line's labels beside them."""
    storage.reload()
    lines, items = load(set_id)
    idx = pilot_ids(lines, PILOT_PER_KIND[set_id])
    lines, items = [lines[i] for i in idx], [items[i] for i in idx]
    answers = committee_label.ask_all(items, systems, committee_label.models(), questions=QUESTIONS)
    pilot_sets, meta_key = pilot_sets_of(set_id), meta_key_of(set_id)
    return committee_label.write_sets(
        pilot_sets,
        lines,
        items,
        answers,
        {},
        meta,
        sources.tonetext_labels_key(pilot_sets["committee"], "pilot_lines.json"),
        lambda ls, a, x, s: samples(ls, a, x, s, "all", meta_key),
        summaries(meta_key),
        batched=BATCHED,
    )


@job(DATA, cpu=4.0, memory=16384, timeout=24 * 3600, nonpreemptible=True)
def label_all(
    systems: dict, meta: dict, wait_s: float = 6 * 3600.0, set_id: str = sources.SUPPORT_ID
) -> dict:
    """Every line through both questions, checkpointed from the start: A's batches, sent once and waited for until the
    first sent plus `wait_s`, then what they left; B and D synchronously; then one more pass for any line a member left
    unanswered."""
    storage.reload()
    t0 = time.time()
    lines, items = load(set_id)
    models = committee_label.models()
    work, sets, meta_key = work_of(set_id), sets_of(set_id), meta_key_of(set_id)
    sent = work.send_once(items, systems, models, ("a",), wait_s, sets, A_CHUNK)
    runners = committee_label.runners(systems, models, B_CONCURRENCY, QUESTIONS)
    batched = [(q, "a") for q in BATCHED]
    done, from_batch = work.answer_all(items, runners, batched, lambda: work.collect_batches(sent, items))
    retried = {}
    for k in runners:
        missing = sum(1 for cid, _ in items if done[k].get(cid, (None,))[0] is None)
        if missing:
            done[k], asked = work.ask_chunked(*k, items, runners[k], have=dict(done[k]))
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
        sets,
        lines,
        items,
        committee_label.answers(done, from_batch, ids, QUESTIONS),
        {"a": True},
        meta,
        samples_key_of(set_id),
        lambda ls, a, x, s: samples(ls, a, x, s, meta_key=meta_key),
        summaries(meta_key),
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
            for q in QUESTIONS
        },
    }


def stage_name(set_id: str, name: str) -> str:
    """A results file's name: the support set's as it was, another set's with its id."""
    return name if set_id == sources.SUPPORT_ID else f"{set_id}-{name}"


@app.local_entrypoint()
def pilot(set_id: str = sources.SUPPORT_ID):
    """PILOT_PER_KIND lines of each kind, both questions, synchronously."""
    systems, meta = committee_label.prepare("pilot", QUESTIONS, **keys_of(set_id))
    res = label_pilot.remote(systems, meta, set_id)
    save(STAGE, stage_name(set_id, "pilot.json"), res)
    print(json.dumps(digest(res), indent=1, default=str))
    show(json.loads(storage.get(res["samples_key"])))


@app.local_entrypoint()
def label(wait_s: float = 6 * 3600.0, set_id: str = sources.SUPPORT_ID):
    """Every line in one spawned call."""
    systems, meta = committee_label.prepare("label", QUESTIONS, **keys_of(set_id))
    call = label_all.spawn(systems, meta, wait_s=wait_s, set_id=set_id)
    save(STAGE, stage_name(set_id, "label-spawned.json"), {"call": call.object_id, **meta, "wait_s": wait_s})
    print("spawned", call.object_id)


@app.local_entrypoint()
def collect(set_id: str = sources.SUPPORT_ID):
    committee_label.collect(STAGE, sets_of(set_id), samples_key_of(set_id), show, digest)
