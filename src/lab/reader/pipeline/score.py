"""The scorecard: the reader through its shipped runtime on the scorecard's test lines, with and without the line
before, plus reads the scorecard lacks. Its tags are written in the line-tags shape with no act toward anyone.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.score --run-id tonetext-r2-student-minilm-l6-seed1
"""

import json

from lab import ROOT
from lab.reader import storage
from lab.reader.app import TORCH, app, job, save

# Topical-Chat's writers labelled each turn with one of these; ours where one matches ("Curious to dive deeper" has none)
TOPICAL_TO_OURS = {
    "angry": "angry",
    "disgusted": "disgusted",
    "fearful": "afraid",
    "sad": "sad",
    "happy": "happy",
    "surprised": "surprised",
    "neutral": "neutral",
}


def as_line_tags(t: dict, targets: list[str]) -> dict:
    """The reader's tags in the line-tags shape (LineTags03 as a dict), with no act toward anyone."""
    return {
        "emotion": t["emotion"]["label"],
        "emotion_intensity": t["emotion"]["intensity"],
        "emotion_confidence": t["emotion"]["confidence"],
        "acts": {x: {"act": "none", "intensity": "low", "confidence": None} for x in targets},
        "backchannel": t["backchannel"]["is_backchannel"],
        "exchange": {"label": t["move"]["label"], "confidence": t["move"]["confidence"]},
        "sarcasm": {"sarcastic": t["sarcasm"]["sarcastic"], "probability": t["sarcasm"]["probability"]},
    }


def cached(fn):
    """A tagger that answers each distinct input once (the scorecard asks for the same line in several reads)."""
    memo = {}

    def tag(inp: dict) -> dict:
        k = json.dumps(inp, sort_keys=True)
        if k not in memo:
            memo[k] = fn(inp)
        return memo[k]

    return tag


def extra_reads(tag, act_rows: list[dict], chain_rows: list[dict], mode: str) -> dict:
    """What the scorecard lacks: sarcasm on the chain test alone, backchannel F1 and emotion intensity accuracy on the
    main test lines that carry a label."""
    from lab.reader import scoring

    chain = [
        (r, tag(scoring.apply(r["line"]["input"], mode)))
        for r in chain_rows
        if r["sarcasm"]["label"] is not None
    ]
    main = [r for r in act_rows if scoring.main_test(r["line"])]
    bgold, bpred, igold, ipred = [], [], [], []
    for r in main:
        t = tag(scoring.apply(r["line"]["input"], mode))
        b = {"true": 1, "false": 0}.get((r.get("backchannel") or {}).get("label"))
        if b is not None:
            bgold.append(b)
            bpred.append(int(bool(t["backchannel"])))
        emo, inten = (r.get("emotion") or {}).get("label"), (r.get("emotion") or {}).get("intensity")
        if emo and emo != "neutral" and inten:
            igold.append(inten)
            ipred.append(t["emotion_intensity"])
    return {
        "chain_test": scoring.sarcasm_scores(
            [1.0 if t["sarcasm"]["sarcastic"] else 0.0 for _, t in chain],
            [int(r["sarcasm"]["label"]) for r, _ in chain],
            0.5,
        ),
        "backchannel_main_test": {
            "lines": len(bgold),
            "backchannels": sum(bgold),
            "f1": round(scoring.f1_sarcastic(bgold, bpred), 4),
            "false_rate": round(sum(p for p, g in zip(bpred, bgold) if g == 0) / max(1, bgold.count(0)), 4),
        },
        "intensity_main_test": {
            "lines": len(igold),
            "accuracy": round(sum(a == b for a, b in zip(igold, ipred)) / max(1, len(igold)), 4),
        },
    }


def voice_reads(tag, rows: dict[str, list[dict]], mode: str) -> dict:
    """A voice-agent test set: emotion, the move and sarcasm on all lines, user lines and agent lines."""
    from lab.reader import scoring, sources
    from lab.reader.calibration import ece
    from lab.reader.committee.label import speaker_role
    from lab.reader.schemas import EMOTIONS, MOVES

    def plain(r: dict) -> dict:
        """Every voice line counts in the scorecard's main block, whatever slices it carries."""
        slices = [s for s in (r["line"].get("slices") or []) if s not in sources.HELD_SLICES]
        return {**r, "line": {**r["line"], "twin_of": None, "slices": slices}}

    rows = {k: [plain(r) for r in v] for k, v in rows.items()}

    def part(cond) -> dict:
        out = {}
        er = [r for r in rows["emotion"] if cond(r)]
        if er:
            et = [tag(scoring.apply(r["line"]["input"], mode)) for r in er]
            a = scoring.act_emotion_scorecard(er, et)["main_test"]
            out["emotion"] = {
                k: a["emotion"].get(k) for k in ("lines", "macro_f1", "macro_f1_ci", "accuracy")
            }
            out["emotion"]["calibration_ece"] = a["calibration"]["emotion"]
            ge = [
                (r["emotion"]["label"], t["emotion"])
                for r, t in zip(er, et)
                if (r.get("emotion") or {}).get("label")
            ]
            out["emotion"]["macro_f1_30"] = round(
                scoring.macro_f1_floor([g for g, _ in ge], [p for _, p in ge], EMOTIONS), 4
            )
        mr = [r for r in rows["move"] if cond(r)]
        if mr:
            mt = [tag(scoring.apply(r["line"]["input"], mode)) for r in mr]
            x = scoring.exchange_scorecard(mr, mt)
            t = x.get("test") or {}
            out["move"] = {
                "lines": t.get("lines"),
                "macro_f1": t.get("macro_f1"),
                "macro_f1_ci": t.get("macro_f1_ci"),
                "answered_vs_open": (t.get("answered_vs_open") or {}).get("rate"),
                "asks_vs_rest": (t.get("asks_vs_rest") or {}).get("rate"),
                "calibration_ece": x.get("calibration_ece"),
            }
            gm = [
                (r["exchange"]["label"], t_["exchange"]["label"])
                for r, t_ in zip(mr, mt)
                if (r.get("exchange") or {}).get("label")
            ]
            out["move"]["macro_f1_30"] = round(
                scoring.macro_f1_floor([g for g, _ in gm], [p for _, p in gm], MOVES), 4
            )
        sr = [r for r in rows["sarcasm"] if cond(r) and r["sarcasm"]["label"] is not None]
        if sr:
            ts = [tag(scoring.apply(r["line"]["input"], mode)) for r in sr]
            gold = [int(r["sarcasm"]["label"]) for r in sr]
            out["sarcasm"] = {
                **scoring.sarcasm_scores([1.0 if t["sarcasm"]["sarcastic"] else 0.0 for t in ts], gold, 0.5),
                "calibration_ece": ece([t["sarcasm"]["probability"] for t in ts], gold)["ece"],
            }
        return out

    return {
        "all": part(lambda r: True),
        "user_lines": part(lambda r: speaker_role(r) == "user"),
        "agent_lines": part(lambda r: speaker_role(r) == "agent"),
        "asr_style_user_lines": part(
            lambda r: speaker_role(r) == "user" and "asr_style" in (r["line"].get("slices") or [])
        ),
    }


def topical_reads(tag, rows: list[dict], mode: str) -> dict:
    """Emotion against Topical-Chat's writers' own labels (test only): macro-F1 over the seven labels that map to ours,
    accuracy, and the spread of what the model said."""
    from collections import Counter

    from lab.reader import scoring

    pairs = []
    for r in rows:
        lab = str((r.get("native") or {}).get("label") or "").strip().lower()
        ours = TOPICAL_TO_OURS.get(lab)
        if ours:
            pairs.append((ours, tag(scoring.apply(r["input"], mode))["emotion"]))
    classes = tuple(sorted(set(TOPICAL_TO_OURS.values())))
    gold, pred = [g for g, _ in pairs], [p for _, p in pairs]
    return {
        "lines": len(pairs),
        "macro_f1": round(scoring.macro_f1(gold, pred, classes), 4) if pairs else None,
        "accuracy": round(sum(g == p for g, p in pairs) / max(1, len(pairs)), 4),
        "gold": dict(Counter(gold).most_common()),
        "predicted": dict(Counter(pred).most_common()),
    }


@job(TORCH, cpu=8.0, memory=16384, timeout=3 * 3600)
def score(run_id: str, prefix: str = "export", source: str = "") -> dict:
    """`source` scores the files in any folder on the volume instead of a run's export."""
    import time

    from lab.reader import runtime, scoring, sources
    from lab.reader.data import previous_words

    storage.reload()
    t0 = time.time()
    tt = runtime.load(runtime.fetch_files(source or sources.run_key(run_id, prefix)), threads=8)
    tt_tag = cached(
        lambda inp: as_line_tags(tt.tag(inp["text"], previous_words(inp.get("previous"))), inp["targets"])
    )

    rows = scoring.test_rows(storage.jsonl)
    act_rows, ex_rows, sar_rows = rows["act"], rows["exchange"], rows["sarcasm"]
    chain_rows = storage.jsonl(sources.sarcasm_key(sources.CHAIN_TEST))
    tt_modes = ("as_given", "no_previous")
    res = {
        "run_id": run_id,
        "source": source or None,
        "act_label_ids": list(sources.ACT_TEST_SETS),
        "act_test_lines": len(act_rows),
        "exchange_test_lines": len(ex_rows),
        "sarcasm_test_lines": len(sar_rows),
        "tonetext": scoring.score_modes(tt_tag, act_rows, ex_rows, tt_modes, sarcasm_rows=sar_rows),
    }
    res["tonetext_extra"] = {m: extra_reads(tt_tag, act_rows, chain_rows, m) for m in tt_modes}
    for name in ("voice", "voice_real"):
        vrows = rows[name]
        res[name] = {
            "lines": {k: len(v) for k, v in vrows.items()},
            "tonetext": {m: voice_reads(tt_tag, vrows, m) for m in tt_modes},
        }
    tc = [
        ln
        for ln in storage.jsonl(sources.open_key("topicalchat_test"))
        if (ln.get("native") or {}).get("label")
    ]
    res["topicalchat"] = {"tonetext": {m: topical_reads(tt_tag, tc, m) for m in tt_modes}}
    res["seconds"] = round(time.time() - t0)
    return res


def table(res: dict) -> list[str]:
    """The scorecard's items, with and without the line before."""

    def emo(r, m):
        return r[m]["act_emotion"]["main_test"]["emotion"]["macro_f1"]

    rows = [
        ("emotion macro-F1, main test", lambda r, m: emo(r, m)),
        (
            "emotion macro-F1, player speech",
            lambda r, m: r[m]["act_emotion"]["player_speech"]["emotion"]["macro_f1"],
        ),
        (
            "emotion macro-F1, unseen settings",
            lambda r, m: r[m]["act_emotion"]["unseen_genre"]["emotion"].get("macro_f1"),
        ),
        ("move macro-F1", lambda r, m: r[m]["exchange"]["test"]["macro_f1"]),
        ("answered or still open", lambda r, m: r[m]["exchange"]["test"]["answered_vs_open"]["rate"]),
        ("asks or not", lambda r, m: r[m]["exchange"]["test"]["asks_vs_rest"]["rate"]),
        ("sarcasm F1, held-out tests", lambda r, m: r[m]["sarcasm"]["sarcasm_test"]["f1"]),
        (
            "sincere lines called sarcastic",
            lambda r, m: r[m]["sarcasm"]["natural_test"]["false_sarcasm_rate"],
        ),
        ("calibration: emotion", lambda r, m: r[m]["act_emotion"]["main_test"]["calibration"]["emotion"]),
        ("calibration: move", lambda r, m: r[m]["exchange"]["calibration_ece"]),
        ("calibration: sarcasm", lambda r, m: r[m]["sarcasm"]["calibration"]["ece"]),
    ]
    cols = [("tonetext", "as_given"), ("tonetext", "no_previous")]
    out = ["item | " + " | ".join(f"{a} {b}" for a, b in cols)]
    for name, f in rows:
        out.append(f"{name} | " + " | ".join(str(f(res[a], b)) for a, b in cols))
    return out


@app.local_entrypoint()
def main(run_id: str = "", prefix: str = "export", source: str = ""):
    """`--source <folder on the volume>` scores those files; `--run-id` then only names them."""
    if not (run_id or source):
        raise SystemExit("give --run-id, or --source with a folder on the volume")
    res = score.remote(run_id, prefix, source)
    path = save("score", f"results-{source.rstrip('/').split('/')[-1] if source else run_id}.json", res)
    print(
        f"{res['act_test_lines']} act test lines, {res['exchange_test_lines']} move test lines, "
        f"{res['sarcasm_test_lines']} sarcasm test lines, {res['seconds']} s"
    )
    print("\n".join(table(res)))
    print("wrote", path.relative_to(ROOT))
