"""The turns r2's committee labels, chosen from the open sets by ``lab.reader.r2_labels``'s rules, with r1 run over
the unsure candidates; the selection's summary keeps counts only.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.r2_select
"""

import json

from lab import ROOT
from lab.reader import sources, storage
from lab.reader.app import TORCH, app, job, save

R1_RUN = "tonetext-r1-student-minilm-l6-seed0"
KEYS = {
    name: sources.tonetext_labels_key(sources.SELECTION, f"{name}.jsonl")
    for name in ("lines", "chosen", "unsure_scores")
}


@job(TORCH, cpu=8.0, memory=32768, timeout=2 * 3600)
def select() -> dict:
    import hashlib
    import time
    from collections import Counter, defaultdict

    from lab.reader import data, open_sources, r2_labels, runtime
    from lab.reader.calibration import softmax
    from lab.reader.schemas import LineInput

    storage.reload()
    t0 = time.time()
    cache: dict[str, list[dict]] = {}

    # Each open set read once, for the pool draw and the dev and test splits alike.
    def jsonl(key: str) -> list[dict]:
        if key not in cache:
            cache[key] = storage.jsonl(key)
        return cache[key]

    pool = data.r2_open_lines(jsonl)
    pool_by: dict[str, list[dict]] = defaultdict(list)
    for ln in pool:
        pool_by[ln["source"]].append(ln)
    every = {src: jsonl(sources.open_key(src)) for src in r2_labels.VOICE_SOURCES}
    t_read = time.time() - t0

    clf = runtime.fetch(R1_RUN)
    tau = float(clf.config.get("sarcasm_tau_raw", 0.5))
    scored: dict[str, dict] = {}
    timing: dict[str, float] = {}

    def score(lines: list[dict]) -> dict[str, dict]:
        """r1 on each candidate as the scorecard reads a line (the line before's words, names cut): the calibrated top
        emotion and move confidences, and the raw sarcasm probability its threshold decides on."""
        t = time.time()
        for ln in lines:
            text, previous = ln["input"]["text"], data.previous_words(ln["input"].get("previous"))
            raw = clf.outputs(text, previous)
            tags = clf.tags_from(raw)
            scored[ln["line_id"]] = {
                "emotion_confidence": tags["emotion"]["confidence"],
                "emotion": tags["emotion"]["label"],
                "move_confidence": tags["move"]["confidence"],
                "move": tags["move"]["label"],
                "sarcasm_raw": round(float(softmax(raw["sarcasm"])[1]), 4),
            }
        timing["score"] = round(time.time() - t)
        return scored

    p, counts = r2_labels.choose(every, pool_by, score, tau)
    chosen = p.chosen()

    # the rules, checked again on the result
    bad = []
    for r in p.rows:
        try:
            open_sources.OpenLine.model_validate(r["line"])
            LineInput.model_validate(r["line"]["input"])
        except Exception as exc:
            bad.append(f"{r['line']['line_id']}: {type(exc).__name__}")
    pool_ids = {ln["line_id"] for ln in pool}
    checks = {
        "invalid": len(bad),
        "line_ids_unique": len({c["line_id"] for c in chosen}) == len(chosen),
        "pairs_unique": len(
            {(r["line"]["input"]["text"], r["line"]["input"].get("previous")) for r in p.rows}
        )
        == len(chosen),
        "set_is_split": all(c["set"] == c["split"] for c in chosen),
        "train_in_pool_draw": all(c["line_id"] in pool_ids for c in chosen if c["set"] == "train"),
        "held_from_voice_sources": all(
            c["source"] in r2_labels.VOICE_SOURCES for c in chosen if c["set"] != "train"
        ),
        "max_same_text_per_set": max(
            Counter((r["set"], r2_labels.norm_text(r["line"]["input"]["text"])) for r in p.rows).values()
        ),
    }
    if bad or not all(v for k, v in checks.items() if k not in ("invalid", "max_same_text_per_set")):
        raise RuntimeError(f"selection checks failed: {checks} {bad[:5]}")

    lines_body = "".join(json.dumps(r["line"]) + "\n" for r in p.rows).encode()
    storage.put(KEYS["lines"], lines_body)
    storage.put_jsonl(KEYS["chosen"], chosen)
    cand_src = {
        ln["line_id"]: (ln["source"], open_sources.role_of(ln)) for ln in pool if ln["line_id"] in scored
    }
    storage.put_jsonl(
        KEYS["unsure_scores"],
        [
            {"line_id": k, "source": cand_src[k][0], "role": cand_src[k][1], **v}
            for k, v in sorted(scored.items())
        ],
    )

    def ids_sha(lines: list[dict]) -> str:
        return hashlib.sha256("\n".join(sorted(ln["line_id"] for ln in lines)).encode()).hexdigest()

    unsure_rows = [r for r in p.rows if r["group"] == "unsure"]
    summary = {
        "selection": sources.SELECTION,
        "seed": r2_labels.SEED,
        "rules": r2_labels.RULES,
        "quotas": {
            "held_per_set": r2_labels.HELD_QUOTA,
            "train_voice_per_role": r2_labels.TRAIN_VOICE_PER_ROLE,
            "multi_party": r2_labels.MULTI_QUOTA,
            "isarcasmeval_pairs": r2_labels.SARCASM_PAIRS,
            "isarcasmeval_singles": r2_labels.SARCASM_SINGLES,
            "dynasent_r2": r2_labels.DYNASENT,
            "unsure": r2_labels.UNSURE,
            "unsure_candidates": r2_labels.UNSURE_CANDIDATES,
            "text_cap_per_set": r2_labels.TEXT_CAP,
        },
        "keys": KEYS,
        "lines_sha256": hashlib.sha256(lines_body).hexdigest(),
        "pool_draw": {
            "caps": dict(sources.OPEN_CAPS),
            "lines": len(pool),
            "by_source": {
                src: {"lines": len(v), "ids_sha256": ids_sha(v)} for src, v in sorted(pool_by.items())
            },
        },
        "held_split_lines": {src: dict(Counter(ln["split"] for ln in v)) for src, v in every.items()},
        "r1": {
            "run_id": R1_RUN,
            "export": sources.run_key(R1_RUN, "export"),
            "sarcasm_tau_raw": tau,
            "candidates_scored": len(scored),
            "score_seconds": timing.get("score"),
            "unsure_taken": len(unsure_rows),
            "taken_emotion_confidence_max": max(
                (r["why"]["emotion_confidence"] for r in unsure_rows if r["why"]["criterion"] == "emotion"),
                default=None,
            ),
            "taken_move_confidence_max": max(
                (r["why"]["move_confidence"] for r in unsure_rows if r["why"]["criterion"] == "move"),
                default=None,
            ),
            "candidates_move_under_0.5": sum(
                v["move_confidence"] < r2_labels.MOVE_UNSURE for v in scored.values()
            ),
            "candidates_sarcasm_near_threshold": sum(
                abs(v["sarcasm_raw"] - tau) <= r2_labels.SARCASM_BAND for v in scored.values()
            ),
        },
        "counts": counts,
        "tally": r2_labels.tally(chosen),
        "checks": checks,
        "seconds": {"read": round(t_read), "total": round(time.time() - t0)},
    }
    storage.put_json(sources.tonetext_labels_key(sources.SELECTION, "summary.json"), summary)
    return summary


@app.local_entrypoint()
def main():
    res = select.remote()
    path = save("r2_select", "selection.json", res)
    t = res["tally"]
    print(
        json.dumps(
            {k: t[k] for k in ("lines", "by_set", "by_set_and_group", "by_set_role_modality")}, indent=1
        )
    )
    print(json.dumps({"checks": res["checks"], "r1": res["r1"], "seconds": res["seconds"]}, indent=1))
    print("wrote", path.relative_to(ROOT))
