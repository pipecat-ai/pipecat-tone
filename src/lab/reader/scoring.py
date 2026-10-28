"""The line classifier's scorecard, which the reader is scored on: metrics with 95% intervals by slice, read from a
runtime's tags in each input mode, and paired comparisons of two taggers on the same lines."""

import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lab.reader import sources
from lab.reader.calibration import ece
from lab.reader.data import previous_words
from lab.reader.schemas import ACTS, EMOTIONS, EXCHANGES, MOVE_INDEX, MOVES, STILL_OPEN, is_hostile

NATURAL = ("mix", "handwritten")
STRATIFIED = ("generated", "civil_comments", "goemotions", "gutenberg")
# The sarcasm test sets: F1 is read on the hard ones, sincere lines called sarcastic on the natural one.
HARD_SARCASM_SETS = sources.HARD_SARCASM_TEST
NATURAL_SARCASM_SET = sources.NATURAL_SARCASM[1]
# A difference counts on 30 lines or more and by 0.1 points or more.
FLOOR_LINES, FLOOR_POINTS = 30, 0.001
# On the calm real voice-agent sets, a class counts in macro-F1 only with this many gold lines.
FLOOR = 30


# --- test lines ----------------------------------------------------------------------------------------------------------


def clean_test(line: dict) -> bool:
    """A test line that is not a noisy twin."""
    return line["split"] == "test" and not line.get("twin_of")


def main_test(line: dict) -> bool:
    """The main test set: clean test lines outside the slices scored apart."""
    slices = line.get("slices") or []
    return clean_test(line) and not any(s in slices for s in sources.HELD_SLICES)


def in_slice(line: dict, name: str) -> bool:
    return clean_test(line) and name in (line.get("slices") or [])


def move_test(line: dict) -> bool:
    """The move's test set: clean test lines outside the unseen settings."""
    return clean_test(line) and "unseen_genre" not in (line.get("slices") or [])


def noisy_twin(line: dict) -> bool:
    return line["split"] == "test" and bool(line.get("twin_of"))


# --- metrics -----------------------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """The 95% Wilson interval for k of n, to 4 places."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def interval(values, places: int = 4) -> list[float]:
    """The middle 95% of resampled values."""
    v = np.sort(values)
    return [round(float(v[int(0.025 * len(v))]), places), round(float(v[int(0.975 * len(v)) - 1]), places)]


def _round(x: float, places: int | None) -> float:
    return x if places is None else round(x, places)


def per_class_f1(
    gold: list[str], pred: list[str], classes: tuple[str, ...], places: int | None = 3, rates: bool = False
) -> dict[str, dict]:
    """Per class that occurs: F1 to `places` (None: unrounded), gold lines and predictions; with `rates`, precision
    and recall too."""
    out = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        row = {"f1": _round(2 * tp / (2 * tp + fp + fn), places), "support": tp + fn, "predicted": tp + fp}
        if rates:
            row["precision"] = _round(tp / (tp + fp), places) if tp + fp else None
            row["recall"] = _round(tp / (tp + fn), places) if tp + fn else None
        out[c] = row
    return out


def macro_f1(gold: list[str], pred: list[str], classes: tuple[str, ...], places: int | None = 3) -> float:
    """The mean of per-class F1 over the classes that occur, each rounded to `places` first."""
    f = per_class_f1(gold, pred, classes, places)
    return sum(v["f1"] for v in f.values()) / len(f) if f else 0.0


def floor_classes(gold: list[str], classes: tuple[str, ...], floor: int = FLOOR) -> tuple[str, ...]:
    """The classes with at least `floor` gold lines."""
    n = Counter(gold)
    return tuple(c for c in classes if n[c] >= floor)


def macro_f1_floor(gold: list[str], pred: list[str], classes: tuple[str, ...], floor: int = FLOOR) -> float:
    """Macro-F1 over the classes with at least `floor` gold lines (a prediction of any other class is simply wrong)."""
    kept = floor_classes(gold, classes, floor)
    return macro_f1(gold, pred, kept) if kept else 0.0


def bootstrap(units: list, metric, n: int = 1000, seed: int = 0, places: int = 4) -> list[float]:
    """The middle 95% of `metric` over n resamples of the units (lines)."""
    if not units:
        return [0.0, 0.0]
    rng = random.Random(seed)
    return interval([metric([rng.choice(units) for _ in units]) for _ in range(n)], places)


def bootstrap_macro_f1(groups: list[list[tuple[str, str]]], classes: tuple[str, ...]) -> list[float]:
    """95% interval for macro-F1, resampling lines (each group holds one line's gold and predicted pairs)."""

    def metric(sample):
        return macro_f1([x for grp in sample for x, _ in grp], [y for grp in sample for _, y in grp], classes)

    return bootstrap(groups, metric, n=500, places=3)


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two labellers on the same lines."""
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# --- the move's scorecard: per-class F1 rounds to 4 places here, as it always has -------------------------------------


def side(label: str | None) -> str | None:
    """Was the question answered, or is it still open?"""
    if label == "answers":
        return "answered"
    if label in STILL_OPEN:
        return "open"
    return None


def answered_vs_open(gold: list[str], pred: list[str]) -> dict:
    """On lines whose gold label is answers, doesnt_know or withholds, the share the prediction puts on the same side.
    `crossed` is the costly error: a side swapped."""
    pairs = [(g, p) for g, p in zip(gold, pred) if side(g)]
    right = sum(side(g) == side(p) for g, p in pairs)
    crossed = sum(side(p) is not None and side(p) != side(g) for g, p in pairs)
    return {
        "lines": len(pairs),
        "right": right,
        "rate": round(right / max(len(pairs), 1), 4),
        "ci": wilson(right, len(pairs)),
        "crossed": crossed,
        "crossed_rate": round(crossed / max(len(pairs), 1), 4),
        "crossed_ci": wilson(crossed, len(pairs)),
        "answers_lines": sum(g == "answers" for g, _ in pairs),
        "open_lines": sum(side(g) == "open" for g, _ in pairs),
    }


def asks_vs_rest(gold: list[str], pred: list[str]) -> dict:
    """asks against every other move, as a two-way decision on every line."""
    n = len(gold)
    right = sum((g == "asks") == (p == "asks") for g, p in zip(gold, pred))
    tp = sum(g == "asks" and p == "asks" for g, p in zip(gold, pred))
    gp = sum(g == "asks" for g in gold)
    pp = sum(p == "asks" for p in pred)
    return {
        "lines": n,
        "right": right,
        "rate": round(right / max(n, 1), 4),
        "ci": wilson(right, n),
        "asks_recall": round(tp / max(gp, 1), 4),
        "asks_precision": round(tp / max(pp, 1), 4),
        "asks_lines": gp,
    }


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """Rows: gold (or the first labeller); columns: prediction (or the second)."""
    m = {g: {p: 0 for p in EXCHANGES} for g in EXCHANGES}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def move_scores(gold: list[str], pred: list[str]) -> dict:
    """Everything the scorecard reads for one set of lines (gold labels without a majority already removed)."""
    right = sum(g == p for g, p in zip(gold, pred))
    return {
        "lines": len(gold),
        "accuracy": round(right / max(len(gold), 1), 4),
        "accuracy_ci": wilson(right, len(gold)),
        "macro_f1": round(macro_f1(gold, pred, EXCHANGES, 4), 4),
        "per_class": per_class_f1(gold, pred, EXCHANGES, 4, rates=True),
        "answered_vs_open": answered_vs_open(gold, pred),
        "asks_vs_rest": asks_vs_rest(gold, pred),
        "confusion": confusion(gold, pred),
        "gold_classes": dict(Counter(gold).most_common()),
        "macro_f1_ci": bootstrap(list(zip(gold, pred)), lambda s: macro_f1(*zip(*s), EXCHANGES, 4)),
    }


def hostility(gold_hostile: list[bool], pred_hostile: list[bool]) -> dict:
    """False hostility over gold non-hostile pairs; recall over gold hostile pairs."""
    neg = [bool(p) for g, p in zip(gold_hostile, pred_hostile) if not g]
    pos = [bool(p) for g, p in zip(gold_hostile, pred_hostile) if g]
    fh, rec = int(sum(neg)), int(sum(pos))
    return {
        "false_hostile": fh,
        "non_hostile_pairs": len(neg),
        "false_hostility_rate": round(fh / max(len(neg), 1), 4),
        "false_hostility_ci": wilson(fh, len(neg)),
        "hostile_found": rec,
        "hostile_pairs": len(pos),
        "hostile_recall": round(rec / max(len(pos), 1), 4),
        "hostile_recall_ci": wilson(rec, len(pos)),
    }


def act_scores(groups: list[list[tuple[str, str]]], hostile_groups: list[list[tuple[bool, bool]]]) -> dict:
    gold = [g for grp in groups for g, _ in grp]
    pred = [p for grp in groups for _, p in grp]
    gh = [g for grp in hostile_groups for g, _ in grp]
    ph = [p for grp in hostile_groups for _, p in grp]
    return {
        "pairs": len(gold),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1), 4),
        "macro_f1": round(macro_f1(gold, pred, ACTS), 4),
        "macro_f1_ci": bootstrap_macro_f1(groups, ACTS),
        "per_class": per_class_f1(gold, pred, ACTS),
        **hostility(gh, ph),
        "gold_classes": dict(Counter(gold).most_common()),
    }


def emotion_scores(gold: list[str], pred: list[str]) -> dict:
    groups = [[(g, p)] for g, p in zip(gold, pred)]
    return {
        "lines": len(gold),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1), 4),
        "macro_f1": round(macro_f1(gold, pred, EMOTIONS), 4),
        "macro_f1_ci": bootstrap_macro_f1(groups, EMOTIONS),
        "per_class": per_class_f1(gold, pred, EMOTIONS),
    }


def f1_sarcastic(gold: list[int], pred: list[int]) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def threshold_on(probs: list[float], gold: list[int], cap: float = 0.02, grid=None) -> float:
    """The lowest threshold whose share of sincere lines called sarcastic is at most `cap`."""
    grid = grid or [round(x / 100, 2) for x in range(5, 100)]
    neg = [p for p, g in zip(probs, gold) if g == 0]
    for t in grid:
        if sum(1 for p in neg if p >= t) <= cap * max(1, len(neg)):
            return t
    return grid[-1]


def sarcasm_scores(probs: list[float], gold: list[int], t: float) -> dict:
    """At threshold t: F1 of the sarcastic class, precision, recall, and the share of sincere lines called
    sarcastic."""
    pred = [int(p >= t) for p in probs]
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    neg = sum(1 for g in gold if g == 0)
    rng, boot = random.Random(0), []  # F1's 95% interval: 1,000 bootstrap resamples of the lines
    for _ in range(1000 if gold else 0):
        idx = [rng.randrange(len(gold)) for _ in gold]
        boot.append(f1_sarcastic([gold[i] for i in idx], [pred[i] for i in idx]))
    boot.sort()
    return {
        "lines": len(gold),
        "sarcastic": sum(gold),
        "f1": round(f1_sarcastic(gold, pred), 4),
        "f1_ci": [round(boot[24], 4), round(boot[974], 4)] if boot else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_sarcasm": fp,
        "sincere": neg,
        "false_sarcasm_rate": round(fp / neg, 4) if neg else None,
        "false_sarcasm_ci": wilson(fp, neg) if neg else None,
    }


# --- labelled rows as items ----------------------------------------------------------------------------------------------


@dataclass
class Pair:
    target: str
    act: str | None  # None: no majority
    intensity: str | None


@dataclass
class LineEx:
    line_id: str
    source: str
    slices: list[str]
    emotion: str | None  # None: no majority
    pairs: list[Pair]
    twin_of: str | None


def line_examples(rows: list[dict]) -> list[LineEx]:
    """Act and emotion rows: one emotion per line, one act per (line, person) pair."""
    return [
        LineEx(
            r["line"]["line_id"],
            r["line"]["source"],
            r["line"]["slices"],
            r["emotion"]["label"],
            [Pair(t, h["label"], h["intensity"]) for t, h in r["acts"].items()],
            r["line"].get("twin_of"),
        )
        for r in rows
    ]


def hostile_label(act: str | None, intensity: str | None) -> bool | None:
    return None if act is None else is_hostile(act, intensity or "low")


def move_examples(rows: list[dict]) -> list[dict]:
    """Move rows as items; a line without a majority keeps label -100."""
    out = []
    for r in rows:
        ln, lab = r["line"], r["exchange"].get("label")
        out.append(
            {
                "line_id": ln["line_id"],
                "y": MOVE_INDEX[lab] if lab else -100,
                "split": ln["split"],
                "twin_of": ln.get("twin_of"),
                "slices": ln.get("slices") or [],
                "source": ln["source"],
                "speaker_kind": ln["input"]["speaker_kind"],
            }
        )
    return out


# --- every scorecard item in every input mode ----------------------------------------------------------------------------
# A caller may or may not send who a line is said to (`to`) and the line before (`previous`): every item is read with
# the line as given, without `to`, without the line before and without either, on the same lines and labels. Lines
# whose labels depend on a removed input are kept, counted and scored apart.

MODES = ("as_given", "no_to", "no_previous", "neither")
LABELS = {
    "as_given": "as given",
    "no_to": "without `to`",
    "no_previous": "without the line before",
    "neither": "without either",
}
SECOND_PERSON = re.compile(r"\b(you|your|you're|yours|yourself|yourselves|ya|u|ur)\b", re.I)


def apply(inp: dict, mode: str) -> dict:
    """The line's input with the mode's optional inputs removed. Labels are never touched."""
    if mode not in MODES:
        raise ValueError(f"unknown input mode {mode}")
    out = dict(inp)
    if mode in ("no_to", "neither"):
        out["to"] = None
    if mode in ("no_previous", "neither"):
        out["previous"] = None
    return out


def addressee_bound(inp: dict, gold_acts: dict[str, str | None]) -> bool:
    """Two or more people, said to one of them, second person in the words, and the gold act toward the addressee
    differs from the gold act toward someone else."""
    to = inp.get("to")
    if not to or len(inp["targets"]) < 2 or not SECOND_PERSON.search(inp["text"]):
        return False
    mine = gold_acts.get(to)
    return mine is not None and any(a is not None and a != mine for t, a in gold_acts.items() if t != to)


def context_bound(inp: dict, slices: list[str] | tuple = ()) -> bool:
    """It has a line before, and it is at most three words long or in the sarcasm slice ("Yes.", "oh great")."""
    return bool(inp.get("previous")) and (len(inp["text"].split()) <= 3 or "sarcasm" in slices)


def act_emotion_scorecard(rows: list[dict], tags: list[dict]) -> dict:
    """The act and emotion items from the runtime's tags (LineTags dicts), one per labelled row."""
    lex = line_examples(rows)

    def where(cond) -> list[int]:
        return [i for i, r in enumerate(rows) if cond(r["line"])]

    def pairs(li):
        t = tags[li]
        for p in lex[li].pairs:
            if p.act is not None:
                yield p, t["acts"][p.target]

    def block(lines: list[int]) -> dict:
        groups, hgroups, eg, ep = [], [], [], []
        for li in lines:
            g = [(p.act, a["act"]) for p, a in pairs(li)]
            h = [
                (bool(hostile_label(p.act, p.intensity)), is_hostile(a["act"], a["intensity"]))
                for p, a in pairs(li)
            ]
            if g:
                groups.append(g)
                hgroups.append(h)
            if lex[li].emotion:
                eg.append(lex[li].emotion)
                ep.append(tags[li]["emotion"])
        return {
            "lines": len(lines),
            "acts": act_scores(groups, hgroups) if groups else {},
            "emotion": emotion_scores(eg, ep) if eg else {},
        }

    def host(lines: list[int]) -> dict:
        gold, pred = [], []
        for li in lines:
            for p, a in pairs(li):
                gold.append(bool(hostile_label(p.act, p.intensity)))
                pred.append(is_hostile(a["act"], a["intensity"]))
        return hostility(gold, pred)

    def calib(lines: list[int]) -> dict:
        a_conf, a_ok, e_conf, e_ok = [], [], [], []
        for li in lines:
            for p, a in pairs(li):
                if a.get("confidence") is None:
                    continue
                hostile = is_hostile(a["act"], a["intensity"])
                a_conf.append(a["confidence"])
                a_ok.append(bool(hostile_label(p.act, p.intensity)) if hostile else a["act"] == p.act)
            if lex[li].emotion and tags[li].get("emotion_confidence") is not None:
                e_conf.append(tags[li]["emotion_confidence"])
                e_ok.append(tags[li]["emotion"] == lex[li].emotion)
        return {"act": ece(a_conf, a_ok)["ece"], "emotion": ece(e_conf, e_ok)["ece"]}

    main = where(main_test)
    speech = where(lambda ln: in_slice(ln, "player_speech"))
    unseen = where(lambda ln: in_slice(ln, "unseen_genre"))
    twins = where(noisy_twin)
    twin_src = {lex[i].twin_of for i in twins}
    clean_tw = [i for i in main if lex[i].line_id in twin_src]
    every = main + speech + unseen
    gold_acts = [{p.target: p.act for p in ex.pairs} for ex in lex]
    inputs = [r["line"]["input"] for r in rows]
    addr = [i for i in every if addressee_bound(inputs[i], gold_acts[i])]
    ctx = [i for i in every if context_bound(inputs[i], lex[i].slices)]
    return {
        "main_test": {**block(main), "calibration": calib(main)},
        "natural_test_false_hostility": host([i for i in main if lex[i].source in NATURAL]),
        "stratified_test_recall": host([i for i in main if lex[i].source in STRATIFIED]),
        "player_speech": {**block(speech), "hostility": host(speech), "calibration": calib(speech)},
        "unseen_genre": {**block(unseen), "hostility": host(unseen)},
        "group_lines": block([i for i in main if len(lex[i].pairs) > 1]),
        "noisy_twins": block(twins),
        "clean_lines_with_twins": block(clean_tw),
        "addressee_bound": {**block(addr), "hostility": host(addr)},
        "context_bound": {**block(ctx), "hostility": host(ctx)},
    }


def exchange_scorecard(rows: list[dict], tags: list[dict]) -> dict:
    """The move's items from the runtime's tags: the test lines with a majority, its slices, the noisy twins, the lines
    without a majority (predicted classes only), the context-bound lines, and the calibration of the reported
    confidence."""
    exs = move_examples(rows)
    pred = [t["exchange"]["label"] for t in tags]

    def where(cond) -> list[int]:
        return [i for i, e in enumerate(exs) if e["y"] >= 0 and cond(e)]

    def block(idx: list[int]) -> dict | None:
        return move_scores([EXCHANGES[exs[i]["y"]] for i in idx], [pred[i] for i in idx]) if idx else None

    main = where(move_test)
    clean = {e["twin_of"] for e in exs if noisy_twin(e)}
    out = {
        "test": block(main),
        "player_lines": block(where(lambda e: move_test(e) and e["speaker_kind"] == "player")),
        "player_speech": block(where(lambda e: move_test(e) and "player_speech" in e["slices"])),
        "npc_lines": block(where(lambda e: move_test(e) and e["speaker_kind"] == "npc")),
        "npc_to_npc": block(where(lambda e: move_test(e) and "npc_to_npc" in e["slices"])),
        "our_lines": block(where(lambda e: move_test(e) and e["source"] == "mix")),
        "generated_lines": block(where(lambda e: move_test(e) and e["source"] == "generated")),
        "noisy_twins": block(where(noisy_twin)),
        "unseen_genre": block(where(lambda e: in_slice(e, "unseen_genre"))),
        "clean_lines_with_twins": block(where(lambda e: clean_test(e) and e["line_id"] in clean)),
    }
    out = {k: v for k, v in out.items() if v is not None}
    nm = [i for i, e in enumerate(exs) if clean_test(e) and e["y"] < 0]
    out["no_majority"] = {"lines": len(nm), "predicted": dict(Counter(pred[i] for i in nm).most_common())}
    ctx = [i for i in main if context_bound(rows[i]["line"]["input"], exs[i]["slices"])]
    if ctx:
        out["context_bound"] = block(ctx)
    conf = [tags[i]["exchange"]["confidence"] for i in main]
    right = [pred[i] == EXCHANGES[exs[i]["y"]] for i in main]
    out["calibration_ece"] = ece(conf, right)["ece"]
    return out


def sarcasm_scorecard(rows: list[dict], tags: list[dict]) -> dict:
    """Rows carry {"line", "sarcasm": {"label"}, "set"}; tags are LineTags03 dicts. F1 on the sarcasm and chain
    tests, sincere lines called sarcastic on the natural test, calibration."""
    keep = [(r, t) for r, t in zip(rows, tags) if r["sarcasm"]["label"] is not None and t.get("sarcasm")]

    def read(cond):
        sel = [(r, t) for r, t in keep if cond(r)]
        return sarcasm_scores(
            [1.0 if t["sarcasm"]["sarcastic"] else 0.0 for _, t in sel],
            [int(r["sarcasm"]["label"]) for r, _ in sel],
            0.5,
        )

    natural = lambda r: r["set"] == NATURAL_SARCASM_SET  # noqa: E731
    return {
        "sarcasm_test": read(lambda r: r["set"] in HARD_SARCASM_SETS),
        "natural_test": read(natural),
        "player_speech_test": read(lambda r: natural(r) and "player_speech" in r["line"]["slices"]),
        "game_dialogue_test": read(lambda r: natural(r) and r["line"]["source"] in NATURAL),
        "calibration": ece(
            [t["sarcasm"]["probability"] for _, t in keep], [int(r["sarcasm"]["label"]) for r, _ in keep]
        ),
    }


def score_modes(
    tag: Callable[[dict], dict],
    act_rows: list[dict],
    exchange_rows: list[dict] | None = None,
    modes: tuple[str, ...] = MODES,
    sarcasm_rows: list[dict] | None = None,
) -> dict:
    """Every item in every mode. `tag` maps a LineInput dict to the runtime's tags as a dict."""
    out = {"modes": list(modes), "labels": {m: LABELS[m] for m in modes}}
    for mode in modes:
        res = {}
        act_tags = [tag(apply(r["line"]["input"], mode)) for r in act_rows]
        res["act_emotion"] = act_emotion_scorecard(act_rows, act_tags)
        if exchange_rows:
            ex_tags = [tag(apply(r["line"]["input"], mode)) for r in exchange_rows]
            res["exchange"] = exchange_scorecard(exchange_rows, ex_tags)
        if sarcasm_rows:
            res["sarcasm"] = sarcasm_scorecard(
                sarcasm_rows, [tag(apply(r["line"]["input"], mode)) for r in sarcasm_rows]
            )
        out[mode] = res
    return out


def test_rows(jsonl) -> dict:
    """The scorecard's test lines (`jsonl` reads one file): the line classifier's act and emotion test lines, each line once; the move
    test; the sarcasm tests, each row with its set; the generated voice-agent set; r2's real voice-agent test set."""
    act, seen = [], set()
    for lid in sources.ACT_TEST_SETS:
        for r in jsonl(sources.labels_key(lid)):
            if r["line"]["split"] == "test" and r["line"]["line_id"] not in seen:
                seen.add(r["line"]["line_id"])
                act.append(r)
    real = {
        "emotion": sources.R2_SETS["committee"],
        "move": sources.R2_SETS["exchange"],
        "sarcasm": sources.R2_SETS["sarcasm"],
    }
    return {
        "act": act,
        "exchange": [
            r for r in jsonl(sources.labels_key(sources.EXCHANGE_SET)) if r["line"]["split"] == "test"
        ],
        "sarcasm": [
            {**r, "set": s}
            for s in (*HARD_SARCASM_SETS, NATURAL_SARCASM_SET)
            for r in jsonl(sources.sarcasm_key(s))
        ],
        "voice": {k: jsonl(sources.tonetext_labels_key(v)) for k, v in sources.VOICE_SETS.items()},
        "voice_real": {
            k: [r for r in jsonl(sources.tonetext_labels_key(v)) if r["line"]["split"] == "test"]
            for k, v in real.items()
        },
    }


def support_rows(jsonl, set_id: str = sources.SUPPORT_ID) -> dict:
    """r3's support-call set's test lines (or r3c's contrast pairs', by `set_id`), each with the committee's labels
    (emotion and sarcasm), and every test line's sarcasm as it was written to be ("sarcasm_built", from the set's
    meta)."""
    rows = {
        q: [r for r in jsonl(sources.tonetext_labels_key(f"{set_id}-{name}")) if r["line"]["split"] == "test"]
        for q, name in (("emotion", "committee"), ("sarcasm", "sarcasm"))
    }
    built = {
        m["line_id"]: m["sarcastic_by_construction"] for m in jsonl(sources.support_key("meta.jsonl", set_id))
    }
    rows["sarcasm_built"] = [
        {**r, "sarcasm": {"label": built[r["line"]["line_id"]]}}
        for r in rows["sarcasm"]
        if built.get(r["line"]["line_id"]) is not None
    ]
    return rows


# --- paired comparisons --------------------------------------------------------------------------------------------------


def class_counts(pairs: list[tuple[str, str]], classes: tuple[str, ...]):
    """Per class [tp, fp, fn] for one line's (gold, predicted) pairs."""
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(classes), 3), dtype=np.int64)
    for g, p in pairs:
        if g == p:
            out[idx[g], 0] += 1
        else:
            out[idx[p], 1] += 1
            out[idx[g], 2] += 1
    return out


def summed_macro_f1(counts):
    """counts [..., K, 3] summed over lines: the mean F1 over classes that occur."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1.sum(-1) / np.maximum((den > 0).sum(-1), 1)


def resamples(lines: int, n: int, seed: int):
    """n bootstrap resamples of the lines, each as how many times it draws every line: [n, lines]."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, lines, lines), minlength=lines) for _ in range(n)])


def paired_f1(
    ref_lines: list,
    var_lines: list,
    classes: tuple[str, ...],
    kept: tuple[str, ...] | None = None,
    n: int = 1000,
    seed: int = 0,
) -> dict:
    """Macro-F1 of var minus ref on the same lines, with a 95% interval resampling lines (the same draws for both).
    ref_lines and var_lines: per line, its list of (gold, predicted) pairs. With `kept`, the mean is over those classes
    only, every class still counted in the confusion (`macro_f1_floor`'s reading), and each side's own interval comes
    back too."""
    R = np.stack([class_counts(x, classes) for x in ref_lines])
    V = np.stack([class_counts(x, classes) for x in var_lines])
    if kept is not None:
        idx = [classes.index(c) for c in kept]
        R, V = R[:, idx, :], V[:, idx, :]
    W = resamples(len(R), n, seed)
    r, v = summed_macro_f1(np.tensordot(W, R, axes=1)), summed_macro_f1(np.tensordot(W, V, axes=1))
    ref, var = float(summed_macro_f1(R.sum(0))), float(summed_macro_f1(V.sum(0)))
    out = {"ref": round(ref, 4), "var": round(var, 4), "diff": round(var - ref, 4), "ci": interval(v - r)}
    if kept is None:
        return {**out, "lines": len(R)}
    return {**out, "ref_ci": interval(r), "var_ci": interval(v), "lines": len(R), "classes": list(kept)}


def paired_rate(
    ref_lines: list[tuple[int, int]], var_lines: list[tuple[int, int]], n: int = 1000, seed: int = 0
) -> dict:
    """A rate (numerator over denominator, summed over lines) of var minus ref, with a 95% interval resampling
    lines."""
    R, V = np.asarray(ref_lines, dtype=np.float64), np.asarray(var_lines, dtype=np.float64)
    W = resamples(len(R), n, seed)
    rr, vv = W @ R, W @ V
    r, v = rr[:, 0] / np.maximum(rr[:, 1], 1), vv[:, 0] / np.maximum(vv[:, 1], 1)
    ref, var = R[:, 0].sum() / max(R[:, 1].sum(), 1), V[:, 0].sum() / max(V[:, 1].sum(), 1)
    return {
        "ref": round(float(ref), 5),
        "var": round(float(var), 5),
        "diff": round(float(var - ref), 5),
        "ci": interval(v - r, 5),
        "lines": len(R),
        "ref_count": [int(R[:, 0].sum()), int(R[:, 1].sum())],
        "var_count": [int(V[:, 0].sum()), int(V[:, 1].sum())],
    }


def paired_bootstrap(units: list, metric, n: int = 1000, seed: int = 0) -> dict:
    """metric(units, side) -> float for side "ref" or "var". The paired difference var - ref on all units, and its 95%
    interval resampling units (lines) with the same indices for both sides."""
    rng = random.Random(seed)
    ref, var = metric(units, "ref"), metric(units, "var")
    diffs = []
    for _ in range(n):
        s = [units[rng.randrange(len(units))] for _ in units]
        diffs.append(metric(s, "var") - metric(s, "ref"))
    return {
        "ref": round(ref, 4),
        "var": round(var, 4),
        "diff": round(var - ref, 4),
        "ci": interval(diffs),
        "units": len(units),
    }


def paired_sarcasm_f1(gold: list[int], ref: list[int], var: list[int]) -> dict:
    units = list(zip(gold, ref, var))
    return paired_bootstrap(
        units,
        lambda u, side: f1_sarcastic([g for g, _, _ in u], [r if side == "ref" else v for _, r, v in u]),
    )


def by_class(pairs: list[tuple[str, str, str]], classes: tuple[str, ...]) -> dict:
    """Per class on (gold, ref, var) triples: gold lines, and for each side F1, recall and how often it was said."""
    out = {}
    for c in classes:
        n = sum(g == c for g, _, _ in pairs)
        row = {"gold": n}
        for side, k in (("ref", 1), ("var", 2)):
            tp = sum(t[0] == c and t[k] == c for t in pairs)
            said = sum(t[k] == c for t in pairs)
            row[side] = {
                "f1": round(2 * tp / max(1, n + said), 4),
                "recall": round(tp / n, 4) if n else None,
                "said": said,
            }
        if n or row["ref"]["said"] or row["var"]["said"]:
            out[c] = row
    return out


def worse_beyond_noise(d: dict, higher_is_better: bool = True) -> bool:
    """The interval wholly on the worse side of zero, on at least FLOOR_LINES lines, by at least FLOOR_POINTS."""
    lines = d.get("lines", d.get("units", 0))
    lo, hi = d["ci"]
    beyond = (hi < 0) if higher_is_better else (lo > 0)
    return bool(beyond and lines >= FLOOR_LINES and abs(d["diff"]) >= FLOOR_POINTS)


def counts_mistakes(item: str) -> bool:
    """An item lower is better on: sincere lines called sarcastic, or sarcasm taken at face value."""
    return item.endswith("sincere_called_sarcastic") or item.startswith("face_value_")


def turn(row: dict, keep: bool) -> tuple[str, str | None]:
    """What the reader reads of a labelled row: its text, and the words of the line before when `keep`."""
    inp = row["line"]["input"]
    return inp["text"], previous_words(inp.get("previous")) if keep else None


class Paired:
    """Two taggers on the same lines, each input read once by both."""

    def __init__(self, ref, var):
        self.ref, self.var = ref, var
        self.tagged: dict[tuple[str, str | None], tuple[dict, dict]] = {}

    def tag(self, text: str, previous: str | None) -> tuple[dict, dict]:
        key = (text, previous)
        if key not in self.tagged:
            self.tagged[key] = (self.ref.tag(text, previous), self.var.tag(text, previous))
        return self.tagged[key]

    def column(self, keep: bool) -> "Column":
        return Column(self, keep)


class Column:
    """The items of one column, with or without the line before: each a paired difference, var minus ref, kept in the
    order read, with each class's reading beside the F1 items."""

    def __init__(self, paired: Paired, keep: bool):
        self.paired, self.keep = paired, keep
        self.items: dict[str, dict] = {}
        self.classes: dict[str, dict] = {}

    def tags(self, row: dict) -> tuple[dict, dict]:
        return self.paired.tag(*turn(row, self.keep))

    def emotion(self, name: str, rows: list[dict], floor: bool = False) -> None:
        rows = [r for r in rows if (r.get("emotion") or {}).get("label")]
        trip = [(r["emotion"]["label"], *(t["emotion"]["label"] for t in self.tags(r))) for r in rows]
        self._f1(name, trip, EMOTIONS, floor)

    def move(self, name: str, rows: list[dict], floor: bool = False) -> None:
        rows = [r for r in rows if (r.get("exchange") or {}).get("label")]
        trip = [(r["exchange"]["label"], *(t["move"]["label"] for t in self.tags(r))) for r in rows]
        self._f1(f"{name}_macro_f1", trip, MOVES, floor)
        ans = [(g, x, y) for g, x, y in trip if side(g)]
        self.items[f"{name}_answered_or_open"] = paired_rate(
            [(int(side(g) == side(x)), 1) for g, x, _ in ans],
            [(int(side(g) == side(y)), 1) for g, _, y in ans],
        )
        self.items[f"{name}_asks_or_not"] = paired_rate(
            [(int((g == "asks") == (x == "asks")), 1) for g, x, _ in trip],
            [(int((g == "asks") == (y == "asks")), 1) for g, _, y in trip],
        )

    def sarcasm(self, name: str, rows: list[dict], f1: bool, false_rate: bool) -> None:
        rows = [r for r in rows if (r.get("sarcasm") or {}).get("label") is not None]
        trip = [
            (int(r["sarcasm"]["label"]), *(int(t["sarcasm"]["sarcastic"]) for t in self.tags(r)))
            for r in rows
        ]
        if f1:
            self.items[f"{name}_f1"] = paired_sarcasm_f1(*map(list, zip(*trip)))
        if false_rate:
            neg = [(x, y) for g, x, y in trip if g == 0]
            self.items[f"{name}_sincere_called_sarcastic"] = paired_rate(
                [(x, 1) for x, _ in neg], [(y, 1) for _, y in neg]
            )

    def worse(self) -> list[str]:
        return [k for k, v in self.items.items() if not counts_mistakes(k) and worse_beyond_noise(v)]

    def better(self) -> list[str]:
        return [
            k
            for k, v in self.items.items()
            if not counts_mistakes(k) and worse_beyond_noise(v, higher_is_better=False)
        ]

    def rises(self) -> list[str]:
        """The items that count mistakes and rise beyond noise."""
        return [
            k
            for k, v in self.items.items()
            if counts_mistakes(k) and worse_beyond_noise(v, higher_is_better=False)
        ]

    def _f1(self, name: str, trip: list[tuple[str, str, str]], classes: tuple[str, ...], floor: bool) -> None:
        a, b = [[(g, x)] for g, x, _ in trip], [[(g, y)] for g, _, y in trip]
        self.items[name] = paired_f1(a, b, classes)
        self.classes[name] = by_class(trip, classes)
        if floor:
            self.items[f"{name}_30"] = paired_f1(
                a, b, classes, floor_classes([g for g, _, _ in trip], classes)
            )
