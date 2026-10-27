"""r2's committee labels on the turns ``lab.reader.pipeline.r2_select`` chose; local results keep counts, scores and
costs only.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.r2_label::pilot
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.r2_label::label
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.r2_label::collect
"""

import time

from lab.reader import r2_labels, sources, storage
from lab.reader.app import DATA, app, job
from lab.reader.committee import label as committee_label

STAGE = "r2_label"
LINES_KEY = sources.tonetext_labels_key(sources.SELECTION, "lines.jsonl")
CHOSEN_KEY = sources.tonetext_labels_key(sources.SELECTION, "chosen.jsonl")
SETS = dict(sources.R2_SETS)
PILOT_SETS = {q: s.replace("tonetext-r2-", "tonetext-r2-pilot-") for q, s in SETS.items()}
# About 40 lines with their labels.
SAMPLES_KEY = sources.tonetext_labels_key(SETS["committee"], "samples.json")
WORK = committee_label.Work("tonetext-r2-work")
KEYS = {"lines_key": LINES_KEY, "chosen_key": CHOSEN_KEY}
# Lines per A batch.
A_CHUNK = 3000
# B's three questions run at once: 24 calls in flight.
B_CONCURRENCY = 8


def load() -> tuple[list, list]:
    """Every chosen line as an open line, so the open sources' names and slices pass through, and (custom id, input)
    for each."""
    from lab.reader.open_sources import OpenLine

    return committee_label.load_lines(LINES_KEY, OpenLine)


@job(DATA, cpu=2.0, memory=8192, timeout=2 * 3600)
def label_pilot(systems: dict, meta: dict) -> dict:
    """30 chosen lines (dev and training, no test line) through all three questions synchronously, into the pilot's
    own sets; every pilot line's labels beside them."""
    storage.reload()
    lines, items = load()
    chosen = storage.jsonl(CHOSEN_KEY)
    idx = r2_labels.pilot_ids(chosen)
    lines, items = [lines[i] for i in idx], [items[i] for i in idx]
    by_id = {r["line_id"]: r for r in chosen}
    answers = committee_label.ask_all(items, systems, committee_label.models())
    return committee_label.write_sets(
        PILOT_SETS,
        lines,
        items,
        answers,
        {},
        meta,
        sources.tonetext_labels_key(PILOT_SETS["committee"], "pilot_lines.json"),
        lambda ls, a, x, s: r2_labels.view(ls, by_id, a, x, s, list(range(len(ls)))),
        r2_labels.summaries({lid: r["group"] for lid, r in by_id.items()}),
    )


# Non-preemptible CPU and memory bill at three times the list rate: 4 GB holds 9,000 lines and every answer.
@job(DATA, cpu=2.0, memory=4096, timeout=20 * 3600, nonpreemptible=True)
def label_all(systems: dict, meta: dict, wait_s: float = 5 * 3600.0) -> dict:
    """Every chosen line through all three questions, checkpointed from the start: A's batches for act and emotion
    and the move, sent once and waited for until the first sent plus `wait_s`, then what they left; B (its batches got
    no prompt-cache reads), D and sarcasm synchronously; then one more pass for any line a member left unanswered."""
    storage.reload()
    t0 = time.time()
    lines, items = load()
    models = committee_label.models()
    sent = WORK.send_once(items, systems, models, ("a",), wait_s, SETS, A_CHUNK)
    runners = committee_label.runners(systems, models, B_CONCURRENCY)
    batched = [(q, "a") for q in committee_label.BATCHED]
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
    chosen = {r["line_id"]: r for r in storage.jsonl(CHOSEN_KEY)}
    meta = {
        **meta,
        "seconds": round(time.time() - t0),
        "answers_from_a_batches": {q: len(from_batch[(q, "a")]) for q in committee_label.BATCHED},
        "retried": retried,
        "a_batches": sent["a"],
    }
    ids = {q: {"a": sent["a"][q], "b": "synchronous"} for q in committee_label.BATCHED}
    return committee_label.write_sets(
        SETS,
        lines,
        items,
        committee_label.answers(done, from_batch, ids),
        {"a": True},
        meta,
        SAMPLES_KEY,
        lambda ls, a, x, s: r2_labels.samples(ls, chosen, a, x, s),
        r2_labels.summaries({k: r["group"] for k, r in chosen.items()}),
    )


def show(rows: list[dict]) -> None:
    """Labelled lines on the terminal, to read; never saved in the repo."""
    for r in rows:
        why = r.get("why") or {}
        print(
            f"[{r['set']}/{r['group']} {r['source']} {r['role']} {r['modality']} {why.get('criterion') or ''}] "
            f"{r['previous']}\n    -> {r['speaker']} (to {r['to']}): {r['text']}\n"
            f"    emotion {r['emotion']}/{r['intensity']} {r['emotion_votes']} | act {r['act']} {r['act_votes']} | "
            f"backchannel {r['backchannel']} | move {r['move']} {r['move_votes']} | sarcastic {r['sarcastic']} "
            f"{r['sarcasm_votes']} d={r['d_sarcasm_probability']}"
        )


def digest(res: dict) -> dict:
    """What the terminal shows of a summary."""
    return {
        **{k: res.get(k) for k in ("cost_usd_total", "seconds", "retried", "answers_from_a_batches")},
        **{
            q: {
                "errors": res[q]["errors"],
                "cost_usd": res[q]["cost_usd"],
                "a_b": res[q].get("all", {}).get("a_b"),
            }
            for q in SETS
        },
    }


@app.local_entrypoint()
def pilot():
    """30 chosen lines, all three questions, synchronously."""
    committee_label.pilot(STAGE, label_pilot, KEYS, show, digest)


@app.local_entrypoint()
def label(wait_s: float = 5 * 3600.0):
    """Every chosen line in one spawned call."""
    committee_label.spawn(STAGE, "label", label_all, KEYS, wait_s=wait_s)


@app.local_entrypoint()
def collect():
    committee_label.collect(STAGE, SETS, SAMPLES_KEY, show, digest)
