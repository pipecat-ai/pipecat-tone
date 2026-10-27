"""r2's committee labels: which turns of the open conversation sets the committee labels (the rules are in RULES), and
how its answers are summarised. Pure functions: ``lab.reader.pipeline.r2_select`` chooses the lines and
``lab.reader.pipeline.r2_label`` asks the committee."""

import random
import re
from collections import Counter, defaultdict

from lab.reader.data import sha_order
from lab.reader.open_sources import ROLES, modality_of, role_of

SEED = 514
VOICE_SOURCES = (
    "natcs",
    "taskmaster1",
    "taskmaster2",
    "taskmaster3",
    "abcd",
    "multiwoz22",
    "star",
    "ccpem",
    "harpervalley",
)
# Turns per source in each held set (dev, test): half the user's, half the agent's.
HELD_QUOTA = {
    "natcs": 300,
    "taskmaster1": 200,
    "taskmaster2": 200,
    "taskmaster3": 150,
    "abcd": 200,
    "multiwoz22": 150,
    "star": 100,
    "ccpem": 100,
    "harpervalley": 100,
}
TRAIN_VOICE_PER_ROLE = 2000
MULTI_QUOTA = {"ami": 400, "dipco": 100}
SARCASM_PAIRS, SARCASM_SINGLES = 150, 100  # iSarcasmEval: 300 lines in pairs, 100 single posts
DYNASENT = 300
DYNASENT_LABELS = ("positive", "negative", "neutral", "mixed")
UNSURE = 800
UNSURE_CANDIDATES = 50_000
MOVE_UNSURE = 0.5  # a move confidence under this
SARCASM_BAND = 0.15  # a raw sarcasm probability within this of r1's threshold
TEXT_CAP = 3
SPOKEN = ("spoken", "asr")
SETS = ("dev", "test", "train")

RULES = {
    "dev_voice_agent": "the voice-agent dev set: the dev split of the nine voice-agent sources, per source a fixed share, half "
    "user and half agent turns, spread over conversations, the user's turns spoken where the source has them",
    "test_voice_agent": "the real voice-agent test set: as the dev set, from the test split",
    "train_voice_agent": "training: the r2 pool draw's voice-agent turns, 2,000 user and 2,000 agent by the held sets' weights, "
    "the user's turns spoken first",
    "train_multi_party": "training: spoken meeting turns from the r2 pool draw (AMI 400, DiPCo 100), spread over meetings",
    "train_sarcasm_pair": "training: an iSarcasmEval sarcastic post with its author's sincere rephrase, both taken",
    "train_sarcasm_single": "training: a single iSarcasmEval post (no rephrase in the set)",
    "train_sentiment": "training: a DynaSent round 2 sentence, even over its four labels",
    "train_unsure": "training: a voice-agent turn r1 is unsure of, from a fixed draw of 50,000 candidates: the lowest top "
    "emotion confidence, a move confidence under 0.5, or a raw sarcasm probability within 0.15 of r1's "
    "threshold, a third each per source",
}


# --- reading a line ------------------------------------------------------------------------------------------------------


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def segment(ln: dict) -> str:
    """One key for Harper Valley's two readings of a segment (its machine transcript's ref ends in /asr)."""
    ref = ln.get("source_ref") or ln["line_id"]
    return f"{ln.get('source')}:{ref[:-4] if ref.endswith('/asr') else ref}"


def allocate(total: int, weights: dict) -> dict:
    """`total` split by `weights` in whole numbers (largest remainder, ties by key)."""
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


# --- choosing --------------------------------------------------------------------------------------------------------------


class Picker:
    """The chosen lines in the order taken, and the rules every take keeps."""

    def __init__(self):
        self.rows: list[dict] = []
        self.ids: set[str] = set()
        self.pairs: set[tuple] = set()
        self.segments: set[str] = set()
        self.texts: Counter = Counter()

    def ok(self, ln: dict, set_name: str) -> bool:
        inp = ln["input"]
        return (
            ln["line_id"] not in self.ids
            and (inp["text"], inp.get("previous")) not in self.pairs
            and segment(ln) not in self.segments
            and self.texts[(set_name, norm_text(inp["text"]))] < TEXT_CAP
        )

    def take(self, ln: dict, set_name: str, group: str, why: dict) -> None:
        inp = ln["input"]
        self.ids.add(ln["line_id"])
        self.pairs.add((inp["text"], inp.get("previous")))
        self.segments.add(segment(ln))
        self.texts[(set_name, norm_text(inp["text"]))] += 1
        self.rows.append({"line": ln, "set": set_name, "group": group, "why": why})

    def chosen(self) -> list[dict]:
        """What is saved beside the lines: ids and reasons, no text."""
        return [
            {
                "line_id": r["line"]["line_id"],
                "source": r["line"]["source"],
                "split": r["line"]["split"],
                "set": r["set"],
                "group": r["group"],
                "role": role_of(r["line"]),
                "modality": modality_of(r["line"]),
                "why": r["why"],
            }
            for r in self.rows
        ]


def spread(
    cands: list[dict], n: int, rng: random.Random, picker: Picker, set_name: str, group: str, why: dict
) -> int:
    """Up to `n` of `cands`, one conversation at a time: the conversations in a shuffled order, a shuffled line from each
    in turn, so no conversation gives a second line before every other has given one. Returns how many were taken."""
    if n <= 0:
        return 0
    by: dict[str, list[dict]] = defaultdict(list)
    for ln in sorted(cands, key=lambda x: x["line_id"]):
        by[ln["group_key"]].append(ln)
    keys = sorted(by)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by[k])
    taken = 0
    while taken < n and keys:
        nxt = []
        for k in keys:
            if taken >= n:
                break
            lst = by[k]
            while lst:
                ln = lst.pop()
                if picker.ok(ln, set_name):
                    picker.take(ln, set_name, group, {**why, "modality": modality_of(ln)})
                    taken += 1
                    break
            if lst:
                nxt.append(k)
        keys = nxt
    return taken


def role_turns(
    cands: list[dict],
    source: str,
    role: str,
    n: int,
    rng: random.Random,
    picker: Picker,
    set_name: str,
    rule: str,
) -> int:
    """`n` turns of `role`: the user's spoken or machine-transcribed turns first where the source has them (Harper Valley:
    half its machine transcript, half its human transcript; its agent turns the human transcript); then any turn of the
    role if those run short."""
    pool = [ln for ln in cands if role_of(ln) == role]
    why = {"rule": rule, "source": source, "role": role}
    got = 0
    if source == "harpervalley":
        if role == "user":
            got += spread(
                [ln for ln in pool if modality_of(ln) == "asr"],
                n // 2,
                rng,
                picker,
                set_name,
                "voice_agent",
                {**why, "preferred": "machine transcript"},
            )
        got += spread(
            [ln for ln in pool if modality_of(ln) == "spoken"],
            n - got,
            rng,
            picker,
            set_name,
            "voice_agent",
            {**why, "preferred": "human transcript"},
        )
    elif role == "user" and any(modality_of(ln) in SPOKEN for ln in pool):
        got += spread(
            [ln for ln in pool if modality_of(ln) in SPOKEN],
            n,
            rng,
            picker,
            set_name,
            "voice_agent",
            {**why, "preferred": "spoken"},
        )
    return got + spread(pool, n - got, rng, picker, set_name, "voice_agent", {**why, "preferred": "any"})


def held_set(lines_by_source: dict[str, list[dict]], split: str, picker: Picker) -> dict:
    """The voice-agent dev or test set from that split only. Returns the count per source and role."""
    counts = {}
    for src in VOICE_SOURCES:
        cands = [ln for ln in lines_by_source.get(src, []) if ln["split"] == split]
        nu = HELD_QUOTA[src] // 2
        for role, n in (("user", nu), ("agent", HELD_QUOTA[src] - nu)):
            counts[f"{src}/{role}"] = role_turns(
                cands,
                src,
                role,
                n,
                random.Random(f"{SEED}:{split}:{src}:{role}"),
                picker,
                split,
                f"{split}_voice_agent",
            )
    return counts


def train_voice(pool_by_source: dict[str, list[dict]], picker: Picker) -> dict:
    per = allocate(TRAIN_VOICE_PER_ROLE, HELD_QUOTA)
    counts = {}
    for src in VOICE_SOURCES:
        cands = [ln for ln in pool_by_source.get(src, []) if ln["split"] == "train"]
        for role in ("user", "agent"):
            counts[f"{src}/{role}"] = role_turns(
                cands,
                src,
                role,
                per[src],
                random.Random(f"{SEED}:train:{src}:{role}"),
                picker,
                "train",
                "train_voice_agent",
            )
    return counts


def train_multi(pool_by_source: dict[str, list[dict]], picker: Picker) -> dict:
    return {
        src: spread(
            [ln for ln in pool_by_source.get(src, []) if ln["split"] == "train"],
            n,
            random.Random(f"{SEED}:train:{src}"),
            picker,
            "train",
            "multi_party",
            {"rule": "train_multi_party", "source": src, "role": "participant"},
        )
        for src, n in MULTI_QUOTA.items()
    }


def _native(ln: dict):
    return (ln.get("native") or {}).get("label")


def train_sarcasm(lines: list[dict], picker: Picker) -> dict:
    """iSarcasmEval: SARCASM_PAIRS sarcastic posts each with its sincere rephrase (both lines or neither), then
    SARCASM_SINGLES posts that stand alone."""
    by: dict[str, list[dict]] = defaultdict(list)
    for ln in sorted((x for x in lines if x["split"] == "train"), key=lambda x: x["line_id"]):
        by[ln["group_key"]].append(ln)
    pairs = [g for _, g in sorted(by.items()) if len(g) == 2 and sorted(_native(x) for x in g) == [0, 1]]
    singles = [g[0] for _, g in sorted(by.items()) if len(g) == 1]
    rng = random.Random(f"{SEED}:train:isarcasmeval")
    rng.shuffle(pairs)
    rng.shuffle(singles)
    n_pairs = n_single = 0
    for g in pairs:
        if n_pairs >= SARCASM_PAIRS:
            break
        if all(picker.ok(x, "train") for x in g) and norm_text(g[0]["input"]["text"]) != norm_text(
            g[1]["input"]["text"]
        ):
            for x in sorted(g, key=lambda x: -_native(x)):  # the sarcastic post, then its rephrase
                picker.take(
                    x,
                    "train",
                    "sentiment_sarcasm",
                    {
                        "rule": "train_sarcasm_pair",
                        "source": "isarcasmeval",
                        "role": "user",
                        "native": _native(x),
                        "pair": x["group_key"],
                    },
                )
            n_pairs += 1
    for x in singles:
        if n_single >= SARCASM_SINGLES:
            break
        if picker.ok(x, "train"):
            picker.take(
                x,
                "train",
                "sentiment_sarcasm",
                {
                    "rule": "train_sarcasm_single",
                    "source": "isarcasmeval",
                    "role": "user",
                    "native": _native(x),
                },
            )
            n_single += 1
    return {
        "pairs": n_pairs,
        "pair_lines": 2 * n_pairs,
        "singles": n_single,
        "pairs_available": len(pairs),
        "singles_available": len(singles),
    }


def train_sentiment(lines: list[dict], picker: Picker) -> dict:
    """DynaSent round 2: DYNASENT sentences even over the labels its train lines carry (DYNASENT_LABELS when all are
    there); a label short of its share leaves the rest to the others. Sentences without a majority label are left out."""
    train = [x for x in lines if x["split"] == "train" and _native(x) is not None]
    labels = sorted(
        {str(_native(x)) for x in train},
        key=lambda lab: (DYNASENT_LABELS.index(lab) if lab in DYNASENT_LABELS else 99, lab),
    )
    by = {lab: [x for x in train if str(_native(x)) == lab] for lab in labels}
    want = {lab: 0 for lab in labels}
    left, open_ = DYNASENT, [lab for lab in labels if by[lab]]
    while left > 0 and open_:
        share = allocate(left, {lab: 1 for lab in open_})
        for lab in open_:
            add = min(share[lab], len(by[lab]) - want[lab])
            want[lab] += add
            left -= add
        open_ = [lab for lab in open_ if want[lab] < len(by[lab])]
    got = {
        lab: spread(
            by[lab],
            want[lab],
            random.Random(f"{SEED}:train:dynasent_r2:{lab}"),
            picker,
            "train",
            "sentiment_sarcasm",
            {"rule": "train_sentiment", "source": "dynasent_r2", "role": "user", "native": lab},
        )
        for lab in labels
    }
    return {"by_label": got, "available": {lab: len(v) for lab, v in by.items()}}


def unsure_candidates(pool_by_source: dict[str, list[dict]], picker: Picker) -> list[dict]:
    """A fixed draw of UNSURE_CANDIDATES voice-agent training turns not yet chosen, by HELD_QUOTA's weights."""
    per = allocate(UNSURE_CANDIDATES, HELD_QUOTA)
    out = []
    for src in VOICE_SOURCES:
        free = [
            ln
            for ln in pool_by_source.get(src, [])
            if ln["split"] == "train" and ln["line_id"] not in picker.ids
        ]
        out += sha_order(free, f"r2-unsure-{SEED}")[: per[src]]
    return out


def pick_unsure(cands: list[dict], scores: dict[str, dict], tau: float, picker: Picker) -> dict:
    """UNSURE lines r1 is unsure of, by HELD_QUOTA's weights. `scores`: line id -> {emotion_confidence, move_confidence,
    sarcasm_raw} from r1 (through the runtime: calibrated confidences, the raw sarcasm probability that r1's
    threshold `tau` decides on). Per source three lists (sarcasm near the threshold, the move under 0.5, the lowest top
    emotion confidence) are taken from in turn, the closest first. Returns counts per source and criterion."""
    per = allocate(UNSURE, HELD_QUOTA)
    counts: dict = {}
    for src in VOICE_SOURCES:
        mine = sorted(
            (ln for ln in cands if ln["source"] == src and ln["line_id"] in scores),
            key=lambda x: x["line_id"],
        )
        sc = {ln["line_id"]: scores[ln["line_id"]] for ln in mine}
        emo_rank = {
            lid: i for i, lid in enumerate(sorted(sc, key=lambda k: (sc[k]["emotion_confidence"], k)))
        }
        lists = {
            "sarcasm": sorted(
                (ln for ln in mine if abs(sc[ln["line_id"]]["sarcasm_raw"] - tau) <= SARCASM_BAND),
                key=lambda x: (abs(sc[x["line_id"]]["sarcasm_raw"] - tau), x["line_id"]),
            ),
            "move": sorted(
                (ln for ln in mine if sc[ln["line_id"]]["move_confidence"] < MOVE_UNSURE),
                key=lambda x: (sc[x["line_id"]]["move_confidence"], x["line_id"]),
            ),
            "emotion": sorted(mine, key=lambda x: (sc[x["line_id"]]["emotion_confidence"], x["line_id"])),
        }
        at = {k: 0 for k in lists}
        got = Counter()
        while sum(got.values()) < per[src] and any(at[k] < len(lists[k]) for k in lists):
            for k in ("sarcasm", "move", "emotion"):
                if sum(got.values()) >= per[src]:
                    break
                while at[k] < len(lists[k]):
                    ln = lists[k][at[k]]
                    at[k] += 1
                    if picker.ok(ln, "train"):
                        s = sc[ln["line_id"]]
                        picker.take(
                            ln,
                            "train",
                            "unsure",
                            {
                                "rule": "train_unsure",
                                "source": src,
                                "role": role_of(ln),
                                "modality": modality_of(ln),
                                "criterion": k,
                                "emotion_confidence": s["emotion_confidence"],
                                "emotion_rank": round(emo_rank[ln["line_id"]] / max(1, len(mine)), 4),
                                "move_confidence": s["move_confidence"],
                                "sarcasm_raw": s["sarcasm_raw"],
                                "sarcasm_threshold": tau,
                                "move_under_0.5": s["move_confidence"] < MOVE_UNSURE,
                                "sarcasm_near_threshold": abs(s["sarcasm_raw"] - tau) <= SARCASM_BAND,
                            },
                        )
                        got[k] += 1
                        break
        counts[src] = {
            "candidates": len(mine),
            "taken": dict(got),
            "eligible": {k: len(v) for k, v in lists.items() if k != "emotion"},
        }
    return counts


def choose(
    lines_by_source: dict[str, list[dict]], pool_by_source: dict[str, list[dict]], score_fn, tau: float
) -> tuple[Picker, dict]:
    """Every set in order (dev, test, then the training groups). `lines_by_source`: each voice-agent source's every line
    (for its dev and test splits); `pool_by_source`: the r2 pool draw by source; `score_fn(lines) -> {line id: scores}`
    runs r1 on the unsure candidates. Returns the picker and the counts (no text)."""
    p = Picker()
    counts = {
        "dev": held_set(lines_by_source, "dev", p),
        "test": held_set(lines_by_source, "test", p),
        "train_voice_agent": train_voice(pool_by_source, p),
        "train_multi_party": train_multi(pool_by_source, p),
        "train_sarcasm": train_sarcasm(pool_by_source.get("isarcasmeval", []), p),
        "train_sentiment": train_sentiment(pool_by_source.get("dynasent_r2", []), p),
    }
    cands = unsure_candidates(pool_by_source, p)
    scores = score_fn(cands)
    counts["train_unsure"] = {
        "candidates": len(cands),
        "tau": tau,
        "by_source": pick_unsure(cands, scores, tau, p),
    }
    return p, counts


def tally(chosen: list[dict]) -> dict:
    """Counts of the chosen lines by set, split, source, role, modality and group (no text)."""

    def c(keyf):
        return dict(sorted(Counter(keyf(r) for r in chosen).items()))

    return {
        "lines": len(chosen),
        "by_set": c(lambda r: r["set"]),
        "by_split": c(lambda r: r["split"]),
        "by_set_and_group": c(lambda r: f"{r['set']}/{r['group']}"),
        "by_set_source_role": c(lambda r: f"{r['set']}/{r['source']}/{r['role']}"),
        "by_set_role_modality": c(lambda r: f"{r['set']}/{r['role']}/{r['modality']}"),
        "by_rule": c(lambda r: r["why"]["rule"]),
        "unsure_by_criterion": dict(
            sorted(
                Counter(
                    f"{r['source']}/{r['why']['criterion']}" for r in chosen if r["group"] == "unsure"
                ).items()
            )
        ),
    }


# --- the pilot and the samples -------------------------------------------------------------------------------------------

PILOT_WANT = (
    ("dev", "voice_agent", "user", 4),
    ("dev", "voice_agent", "agent", 4),
    ("train", "voice_agent", "user", 3),
    ("train", "voice_agent", "agent", 3),
    ("train", "multi_party", None, 4),
    ("train", "sentiment_sarcasm", None, 7),
    ("train", "unsure", None, 5),
)


def pilot_ids(chosen: list[dict]) -> list[int]:
    """30 lines spread over the dev set and the training groups (no test line): positions in the chosen order."""
    rng = random.Random(f"{SEED}:pilot")
    out: list[int] = []
    for set_name, group, role, n in PILOT_WANT:
        pool = [
            i
            for i, r in enumerate(chosen)
            if r["set"] == set_name and r["group"] == group and (role is None or r["role"] == role)
        ]
        out += rng.sample(pool, min(n, len(pool)))
    want = sum(w[3] for w in PILOT_WANT)
    if len(out) < want:  # a group short of its share: the rest from any dev or training line
        rest = [i for i, r in enumerate(chosen) if r["set"] != "test" and i not in set(out)]
        out += rng.sample(rest, min(want - len(out), len(rest)))
    return sorted(out)


def samples(
    lines: list, chosen_by_id: dict, act_l: list, xch_l: list, sar_l: list, n: int = 40
) -> list[dict]:
    """About `n` labelled lines to glance at: dev and test turns of both roles, every training group, and up to five with
    a sarcastic majority."""
    rng = random.Random(f"{SEED}:samples")
    want = {
        ("dev", "user"): 6,
        ("dev", "agent"): 6,
        ("test", "user"): 4,
        ("test", "agent"): 4,
        ("train", "voice_agent"): 6,
        ("train", "multi_party"): 4,
        ("train", "sentiment_sarcasm"): 5,
        ("train", "unsure"): 5,
    }
    scale = n / 40

    def kind(i):
        c = chosen_by_id.get(lines[i].line_id) or {}
        return (
            (c.get("set"), c.get("role"))
            if c.get("set") in ("dev", "test")
            else (c.get("set"), c.get("group"))
        )

    by: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(lines)):
        by[kind(i)].append(i)
    picked = set()
    sarcastic = [i for i, s in enumerate(sar_l) if s["sarcasm"]["label"] is True]
    picked.update(rng.sample(sarcastic, min(len(sarcastic), round(5 * scale))))
    for k, m in want.items():
        pool = [i for i in by.get(k, []) if i not in picked]
        picked.update(
            rng.sample(
                pool, min(len(pool), max(0, round(m * scale) - sum(1 for i in picked if kind(i) == k)))
            )
        )
    return view(lines, chosen_by_id, act_l, xch_l, sar_l, sorted(picked))


def view(
    lines: list, chosen_by_id: dict, act_l: list, xch_l: list, sar_l: list, idx: list[int]
) -> list[dict]:
    """Lines `idx` with their three labels, every vote and why each was chosen (the volume and the terminal only, never
    the repo)."""
    out = []
    for i in idx:
        ln, e, x, s = lines[i], act_l[i], xch_l[i], sar_l[i]
        c = chosen_by_id.get(ln.line_id) or {}
        act = next(iter(e["acts"].values()))
        out.append(
            {
                "line_id": ln.line_id,
                "set": c.get("set"),
                "group": c.get("group"),
                "source": ln.source,
                "role": c.get("role"),
                "modality": c.get("modality"),
                "why": c.get("why"),
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


# --- the summaries: per split and per role, read as Tone's reports read them ---------------------------------------------


def row_role(row: dict) -> str:
    return role_of(row["line"])


def _roles(rows: list[dict]) -> tuple[str, ...]:
    have = {row_role(r) for r in rows}
    return tuple(x for x in ROLES if x in have)


def _splits(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        s: [r for r in rows if r["line"]["split"] == s]
        for s in SETS
        if any(r["line"]["split"] == s for r in rows)
    }


def summaries(group_of: dict[str, str] | None = None) -> dict:
    """The summary functions ``committee.label.write_sets`` takes, for r2's sets: every line, then each split
    with its roles (A against B per split and per role), and the majority's spread per training group (`group_of`: line id
    to group)."""
    from lab.reader.committee import label as committee_label

    group_of = group_of or {}

    def groups(rows, spread):
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[f"{r['line']['split']}/{group_of.get(r['line']['line_id'], '?')}"].append(r)
        return {g: {"lines": len(rs), **spread(rs)} for g, rs in sorted(by.items())}

    def committee(rows):
        return {
            "all": committee_label.act_emotion_summary(rows, row_role, _roles(rows)),
            "by_split": {
                s: committee_label.act_emotion_summary(rs, row_role, _roles(rs))
                for s, rs in _splits(rows).items()
            },
            "by_group": groups(
                rows,
                lambda rs: {
                    "emotion": dict(
                        Counter(r["emotion"]["label"] or "(no majority)" for r in rs).most_common()
                    )
                },
            ),
        }

    def exchange(rows, votes):
        ids = {s: {r["line"]["line_id"] for r in rs} for s, rs in _splits(rows).items()}
        return {
            "all": committee_label.exchange_summary(rows, votes, row_role, _roles(rows), split=None),
            "by_split": {
                s: committee_label.exchange_summary(
                    rs, [v for v in votes if v["line_id"] in ids[s]], row_role, _roles(rs), split=s
                )
                for s, rs in _splits(rows).items()
            },
            "by_group": groups(
                rows,
                lambda rs: {
                    "move": dict(Counter(r["exchange"]["label"] or "(no majority)" for r in rs).most_common())
                },
            ),
        }

    def sarcasm(rows):
        return {
            "all": committee_label.sarcasm_summary(
                rows, row_role, _roles(rows), seeds=(), ab_roles=_roles(rows)
            ),
            "by_split": {
                s: committee_label.sarcasm_summary(rs, row_role, _roles(rs), seeds=(), ab_roles=_roles(rs))
                for s, rs in _splits(rows).items()
            },
            "by_group": groups(
                rows, lambda rs: {"sarcasm": dict(Counter(str(r["sarcasm"]["label"]) for r in rs))}
            ),
        }

    return {"committee": committee, "exchange": exchange, "sarcasm": sarcasm}
