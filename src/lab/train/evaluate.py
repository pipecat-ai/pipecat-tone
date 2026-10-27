"""Scores any tagger against the teacher's labels on rows, so rules and models compare like for like.

Delivery: accuracy, F1 for tagging a sentence at all, and macro F1 over the deliveries the teacher
used. Slow spans and stressed words: F1 over exact character ranges. Sounds other than breath: F1 over
(position, sound). Breaths apart, as which replies breathe is drawn at random: their F1 against the teacher,
and how a tagger's breaths fall across calls, in the terms real speech is measured in.
"""

import asyncio
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from lab.corpus.rows import Row
from lab.corpus.speech_breaths import Turn, summary
from pipecat_tone.tagger import SentenceLabels, Tagger, ToneTaggerModel
from pipecat_tone.tagger.labels import is_breath
from pipecat_tone.tagger.markup import keep
from pipecat_tone.tagger.model import words


def score(rows: list[Row], predict: Callable[[Row], SentenceLabels]) -> dict:
    counts: Counter = Counter()
    per_class: Counter = Counter()
    for row in rows:
        gold, pred = row.labels, predict(row)
        g, p = gold.delivery, pred.delivery
        counts["n"] += 1
        counts["delivery_correct"] += g == p
        counts["tag_tp"] += bool(g) and bool(p)
        counts["tag_fp"] += not g and bool(p)
        counts["tag_fn"] += bool(g) and not p
        for label in {g, p} - {None}:
            per_class[(label, "tp")] += g == p == label
            per_class[(label, "fp")] += p == label and g != label
            per_class[(label, "fn")] += g == label and p != label
        _count(counts, "span", {(s.start, s.end) for s in gold.spans}, {(s.start, s.end) for s in pred.spans})
        _count(counts, "emphasis", set(gold.emphasis), set(pred.emphasis))
        _count(counts, "sound", _sounds(gold), _sounds(pred))
        _count(counts, "breath", _breath(gold), _breath(pred))
    classes = sorted({label for label, _ in per_class})
    class_f1 = {c: _f1(per_class[(c, "tp")], per_class[(c, "fp")], per_class[(c, "fn")]) for c in classes}
    used = [c for c in classes if per_class[(c, "tp")] + per_class[(c, "fn")]]
    result = {
        "rows": counts["n"],
        "delivery_accuracy": counts["delivery_correct"] / max(counts["n"], 1),
        "tagged_f1": _f1(counts["tag_tp"], counts["tag_fp"], counts["tag_fn"]),
        "tagged_precision": counts["tag_tp"] / max(counts["tag_tp"] + counts["tag_fp"], 1),
        "tagged_recall": counts["tag_tp"] / max(counts["tag_tp"] + counts["tag_fn"], 1),
        "delivery_macro_f1": sum(class_f1[c] for c in used) / max(len(used), 1),
        "span_f1": _f1(counts["span_tp"], counts["span_fp"], counts["span_fn"]),
        "emphasis_f1": _f1(counts["emphasis_tp"], counts["emphasis_fp"], counts["emphasis_fn"]),
        "sound_f1": _f1(counts["sound_tp"], counts["sound_fp"], counts["sound_fn"]),
        "breath_f1": _f1(counts["breath_tp"], counts["breath_fp"], counts["breath_fn"]),
        "breath_precision": counts["breath_tp"] / max(counts["breath_tp"] + counts["breath_fp"], 1),
        "breath_recall": counts["breath_tp"] / max(counts["breath_tp"] + counts["breath_fn"], 1),
        "breaths_gold": counts["breath_tp"] + counts["breath_fn"],
        "breaths_predicted": counts["breath_tp"] + counts["breath_fp"],
        "per_delivery_f1": {c: round(class_f1[c], 3) for c in used},
    }
    # One number for choosing a checkpoint: deliveries and spans matter most, and drawn breaths stay out.
    result["combined"] = (
        0.4 * result["delivery_macro_f1"]
        + 0.2 * result["tagged_f1"]
        + 0.3 * result["span_f1"]
        + 0.1 * result["sound_f1"]
    )
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}


def predictor(tagger: Tagger) -> Callable[[Row], SentenceLabels]:
    """A tagger as a predictor over rows in call order: its turn pass when a reply starts, then its sentence pass, as
    in a call. A new call starts whenever the rows' call changes, so nothing carries over between calls."""
    at: list = [None, None]

    async def label(row: Row) -> SentenceLabels:
        if row.call != at[0]:
            tagger.start_call()
        if [row.call, row.turn] != at:
            await tagger.start_turn(row.turn_context())
            at[:] = [row.call, row.turn]
        return await tagger.tag_sentence(row.sentence, row.earlier)

    return lambda row: keep(row.sentence, asyncio.run(label(row)), row.space)


def breaths(rows: list[Row], tagger: Tagger) -> dict:
    """A tagger's breaths on the rows offering breath, each call's replies in order and each after the tagger's own
    last reply, as in a call: F1 against the teacher's on trainable rows, and both summarised as real speech is.
    Rows should hold whole replies, trainable or not."""
    calls: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "breath" in r.space.sounds:
            calls[r.call][r.turn].append(r)
    pairs, mine, teachers = [], [], []
    for call, turns in calls.items():
        tagger.start_call()
        for t in sorted(turns):
            reply = sorted(turns[t], key=lambda r: r.n)
            marks = asyncio.run(_breathes(tagger, reply))
            gold = [r.labels.sound_before == "breath" for r in reply]
            mine.append(Turn(call, tuple((words(r.sentence), b) for r, b in zip(reply, marks))))
            teachers.append(Turn(call, tuple((words(r.sentence), g) for r, g in zip(reply, gold))))
            pairs += [(b, g) for r, b, g in zip(reply, marks, gold) if r.trainable]
    tp = sum(b and g for b, g in pairs)
    return {
        "breath_f1": round(
            _f1(tp, sum(b and not g for b, g in pairs), sum(g and not b for b, g in pairs)), 4
        ),
        "breaths_predicted": sum(b for b, _ in pairs),
        "breaths_gold": sum(g for _, g in pairs),
        "tagger": summary(mine),
        "teacher": summary(teachers),
    }


async def _breathes(tagger: Tagger, reply: list[Row]) -> list[bool]:
    await tagger.start_turn(replace(reply[0].turn_context(), previous=None))
    return [is_breath((await tagger.tag_sentence(r.sentence, r.earlier)).sound_before) for r in reply]


def benchmark(folder: Path, rows: list[Row], n: int = 300, threads: int = 1) -> dict:
    """The sentence pass's time on this machine's CPU, in milliseconds."""
    tagger = ToneTaggerModel(folder, threads=threads)
    inputs = [(row.turn_context(), row.earlier, row.sentence) for row in rows[:n]]
    for args in inputs[:20]:
        tagger.predict(*args)
    times = []
    for args in inputs:
        started = time.perf_counter()
        tagger.predict(*args)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {
        "n": len(times),
        "threads": threads,
        "p50_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[int(len(times) * 0.95)], 2),
        "max_ms": round(times[-1], 2),
    }


def _sounds(labels: SentenceLabels) -> set[tuple[str, str]]:
    return {
        (at, sound)
        for at, sound in (("before", labels.sound_before), ("after", labels.sound_after))
        if sound and not is_breath(sound)
    }


def _breath(labels: SentenceLabels) -> set[str]:
    return {"before"} if is_breath(labels.sound_before) else set()


def _count(counts: Counter, name: str, gold: set, pred: set):
    counts[f"{name}_tp"] += len(gold & pred)
    counts[f"{name}_fp"] += len(pred - gold)
    counts[f"{name}_fn"] += len(gold - pred)


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)
