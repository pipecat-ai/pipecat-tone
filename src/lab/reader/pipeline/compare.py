"""Two exports compared line by line on every test set through the shipped runtime, with and without the line before:
each scorecard item as the paired difference, var minus ref, with its 95% bootstrap interval.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.compare --var tonetext-r2-student-minilm-l6-seed1 --ref tonetext-r1-student-minilm-l6-seed0
"""

from lab import ROOT
from lab.reader import storage
from lab.reader.app import RESULTS, TORCH, app, job, save

# The items on Tone's own test sets, which r2 may not make worse than r1 beyond noise.
TONE_ITEMS = (
    "emotion_main",
    "emotion_player_speech",
    "emotion_unseen",
    "move_macro_f1",
    "move_answered_or_open",
    "move_asks_or_not",
    "sarcasm_held_out_f1",
)
# The support-call set's moments whose agent line carries no news, as its lines' slices name them.
PLAIN = ("moment_status", "moment_question", "moment_procedure", "moment_kind")


def probe_reads(ref, var) -> dict:
    """The cases that showed rc1's misread (``support.PROBES``), read by both as they stand: each side's two likeliest
    emotions, its sarcasm probability and call, and whether the call is the one the case should get. Keyed by case id,
    without the lines' text."""
    from lab.reader.support import PROBES

    def read(model, text, previous):
        t = model.tag(text, previous)
        top = sorted(t["emotion"]["probabilities"].items(), key=lambda p: -p[1])[:2]
        return {
            "emotion": [[k, round(v, 2)] for k, v in top],
            "sarcasm": round(t["sarcasm"]["probability"], 2),
            "sarcastic": bool(t["sarcasm"]["sarcastic"]),
        }

    out = {}
    for pid, text, previous, want in PROBES:
        sides = {"ref": read(ref, text, previous), "var": read(var, text, previous)}
        for v in sides.values():
            v["right"] = v["sarcastic"] == (want == "sarcastic")
        out[pid] = {"should_read": want, "with_line_before": previous is not None, **sides}
    out["_right"] = {side: sum(v[side]["right"] for v in out.values()) for side in ("ref", "var")}
    out["_cases"] = len(PROBES)
    return out


@job(TORCH, cpu=8.0, memory=16384, timeout=3 * 3600)
def compare(var_run: str, ref_run: str, var_prefix: str = "export", ref_prefix: str = "export") -> dict:
    import time

    from lab.reader import runtime, scoring, sources
    from lab.reader.committee.label import speaker_role

    storage.reload()
    t0 = time.time()
    ref, var = runtime.fetch(ref_run, ref_prefix), runtime.fetch(var_run, var_prefix)
    rows = scoring.test_rows(storage.jsonl)
    act_rows, ex_rows, sar_rows, voice, real = (
        rows["act"],
        rows["exchange"],
        rows["sarcasm"],
        rows["voice"],
        rows["voice_real"],
    )

    # Sarcastic lines that carry a committee emotion: Tone's act test lines a sarcasm set calls sarcastic (or with the
    # sarcasm slice), and the generated voice-agent lines its sarcasm labels call sarcastic.
    is_sarc = lambda r: (r.get("sarcasm") or {}).get("label") is not None and int(r["sarcasm"]["label"]) == 1  # noqa: E731
    sarc_ids = {r["line"]["line_id"] for r in sar_rows if is_sarc(r)}
    voice_sarc_ids = {r["line"]["line_id"] for r in voice["sarcasm"] if is_sarc(r)}
    sarcastic_tone = [
        r
        for r in act_rows
        if scoring.clean_test(r["line"])
        and (r["line"]["line_id"] in sarc_ids or "sarcasm" in (r["line"].get("slices") or []))
    ]
    sarcastic_voice = [r for r in voice["emotion"] if r["line"]["line_id"] in voice_sarc_ids]
    support = None
    if storage.exists(sources.tonetext_labels_key(sources.SUPPORT_SETS["sarcasm"])):
        support = scoring.support_rows(storage.jsonl)
        support_sarc_ids = {r["line"]["line_id"] for r in support["sarcasm"] if is_sarc(r)}
        sarcastic_support = [r for r in support["emotion"] if r["line"]["line_id"] in support_sarc_ids]
    plain = lambda r: any(x in (r["line"].get("slices") or []) for x in PLAIN)  # noqa: E731
    held = lambda r: "held_domain" in (r["line"].get("slices") or [])  # noqa: E731
    moment = lambda r: "reply_ordinary" not in (r["line"].get("slices") or [])  # noqa: E731

    keep_cfg = ("run_id", "sarcasm_tau_raw", "sarcasm_tau_rule", "calibration")
    out = {
        "var_run": var_run,
        "ref_run": ref_run,
        "var_prefix": var_prefix,
        "ref_prefix": ref_prefix,
        "var_config": {k: var.config.get(k) for k in keep_cfg},
        "ref_config": {k: ref.config.get(k) for k in keep_cfg},
        "columns": {},
    }
    paired = scoring.Paired(ref, var)
    for col, keep in (("with_previous", True), ("without_previous", False)):
        c = paired.column(keep)
        c.emotion("emotion_main", [r for r in act_rows if scoring.main_test(r["line"])])
        c.emotion(
            "emotion_player_speech", [r for r in act_rows if scoring.in_slice(r["line"], "player_speech")]
        )
        c.emotion("emotion_unseen", [r for r in act_rows if scoring.in_slice(r["line"], "unseen_genre")])
        c.emotion("emotion_noisy_twins", [r for r in act_rows if scoring.noisy_twin(r["line"])])
        c.emotion("emotion_voice", voice["emotion"])
        c.emotion("emotion_voice_agent", [r for r in voice["emotion"] if speaker_role(r) == "agent"])
        c.emotion("emotion_voice_real", real["emotion"], floor=True)
        c.emotion(
            "emotion_voice_real_agent", [r for r in real["emotion"] if speaker_role(r) == "agent"], floor=True
        )
        c.move("move", [r for r in ex_rows if scoring.move_test(r["line"])])
        c.move("move_voice", voice["move"])
        c.move("move_voice_real", real["move"], floor=True)
        c.sarcasm(
            "sarcasm_held_out", [r for r in sar_rows if r["set"] in scoring.HARD_SARCASM_SETS], True, False
        )
        c.sarcasm(
            "sarcasm_natural", [r for r in sar_rows if r["set"] == scoring.NATURAL_SARCASM_SET], False, True
        )
        c.sarcasm("sarcasm_voice", voice["sarcasm"], True, True)
        c.sarcasm("sarcasm_voice_real", real["sarcasm"], True, True)
        sarcastic_sets = [("sarcastic_tone", sarcastic_tone), ("sarcastic_voice", sarcastic_voice)]
        if support:
            sr, built = support["sarcasm"], support["sarcasm_built"]
            c.emotion("emotion_support", support["emotion"])
            c.sarcasm("sarcasm_support", sr, True, True)
            c.sarcasm("sarcasm_support_after_plain", [r for r in sr if plain(r)], True, True)
            c.sarcasm("sarcasm_support_after_news", [r for r in sr if moment(r) and not plain(r)], True, True)
            c.sarcasm("sarcasm_support_held_domains", [r for r in sr if held(r)], True, True)
            c.sarcasm("sarcasm_support_built", built, True, True)
            c.sarcasm("sarcasm_support_built_after_plain", [r for r in built if plain(r)], True, True)
            c.sarcasm("sarcasm_support_built_after_news", [r for r in built if not plain(r)], True, True)
            sarcastic_sets.append(("sarcastic_support", sarcastic_support))
        for name, sarcastic in sarcastic_sets:
            sarcastic = [r for r in sarcastic if (r.get("emotion") or {}).get("label")]
            if not sarcastic:
                continue
            c.emotion(f"emotion_{name}", sarcastic)
            # Taken at face value: called happy where the committee's emotion is not happy.
            neg = [r for r in sarcastic if r["emotion"]["label"] != "happy"]
            said = [tuple(int(t["emotion"]["label"] == "happy") for t in c.tags(r)) for r in neg]
            c.items[f"face_value_{name}"] = scoring.paired_rate(
                [(x, 1) for x, _ in said], [(y, 1) for _, y in said]
            )
        worse = c.worse()
        out["columns"][col] = {
            "items": c.items,
            "by_class": c.classes,
            "worse_beyond_noise": worse,
            "better_beyond_noise": c.better(),
            "false_sarcasm_or_face_value_rises": c.rises(),
            "tone_items_worse": [k for k in worse if k in TONE_ITEMS],
        }
    out["probes"] = probe_reads(ref, var)
    out["lines_tagged"] = len(paired.tagged)
    out["seconds"] = round(time.time() - t0)
    return out


# The table's rows: (group, label, item, column), the item as `compare` names it.
TABLE = (
    ("Voice-agent sets", "Emotion macro-F1, generated calls (9 classes)", "emotion_voice", "with_previous"),
    ("Voice-agent sets", "Emotion macro-F1, real calls", "emotion_voice_real", "with_previous"),
    ("Voice-agent sets", "Move macro-F1, real calls", "move_voice_real_macro_f1", "with_previous"),
    ("Voice-agent sets", "Sarcasm F1, generated calls", "sarcasm_voice_f1", "with_previous"),
    (
        "Voice-agent sets",
        "Sincere turns called sarcastic, real calls",
        "sarcasm_voice_real_sincere_called_sarcastic",
        "with_previous",
    ),
    ("Sarcasm sets", "Sarcasm F1, held-out tests", "sarcasm_held_out_f1", "with_previous"),
    (
        "Sarcasm sets",
        "Sincere lines called sarcastic, natural test",
        "sarcasm_natural_sincere_called_sarcastic",
        "with_previous",
    ),
    (
        "Sarcasm sets",
        "Sarcastic lines read happy, Tone's tests",
        "face_value_sarcastic_tone",
        "with_previous",
    ),
    ("Tone's sets", "Emotion macro-F1, main test", "emotion_main", "with_previous"),
    ("Tone's sets", "Emotion macro-F1, player speech", "emotion_player_speech", "with_previous"),
    ("Tone's sets", "Move macro-F1", "move_macro_f1", "with_previous"),
    ("Support-call test split", "Sarcasm F1, committee labels", "sarcasm_support_f1", "with_previous"),
    (
        "Support-call test split",
        "Sarcasm F1 after a plain agent line",
        "sarcasm_support_after_plain_f1",
        "with_previous",
    ),
    ("Support-call test split", "Sarcasm F1 after news", "sarcasm_support_after_news_f1", "with_previous"),
    (
        "Support-call test split",
        "Sarcasm F1, held-out domains",
        "sarcasm_support_held_domains_f1",
        "with_previous",
    ),
    (
        "Support-call test split",
        "Sincere turns called sarcastic",
        "sarcasm_support_sincere_called_sarcastic",
        "with_previous",
    ),
    ("Support-call test split", "Sarcasm F1, as written", "sarcasm_support_built_f1", "with_previous"),
    (
        "Support-call test split",
        "Sarcasm F1 after a plain line, as written",
        "sarcasm_support_built_after_plain_f1",
        "with_previous",
    ),
    (
        "Support-call test split",
        "Sarcastic turns read happy",
        "face_value_sarcastic_support",
        "with_previous",
    ),
    ("Support-call test split", "Emotion macro-F1", "emotion_support", "with_previous"),
    ("Support-call test split", "Sarcasm F1, turn alone", "sarcasm_support_f1", "without_previous"),
)


def markdown(res: dict, ref_name: str = "ref", var_name: str = "var") -> str:
    """The comparison as a markdown table: each item read with the line before as Pipecat reads it (some also on the
    turn alone), the difference with its 95% interval, then the probes. Scores only, no line text."""
    rows = [
        f"| | Item | {ref_name} | {var_name} | Difference (95% interval) | Lines |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group, label, item, col in TABLE:
        v = res["columns"][col]["items"].get(item)
        if v is None:
            continue
        lines = v.get("lines", v.get("units"))
        rows.append(
            f"| {group} | {label} | {v['ref']} | {v['var']} | {v['diff']:+} ({v['ci'][0]:+} to {v['ci'][1]:+}) | "
            f"{lines} |"
        )
    probes = res["probes"]
    rows += [
        "",
        f"Cases read right: {ref_name} {probes['_right']['ref']}, {var_name} {probes['_right']['var']} of "
        f"{probes['_cases']}.",
        "",
        f"| Case | Should read | {ref_name}: emotion, sarcasm | {var_name}: emotion, sarcasm |",
        "| --- | --- | --- | --- |",
    ]
    for pid, v in probes.items():
        if pid.startswith("_"):
            continue
        cells = [
            f"{v[side]['emotion'][0][0]} {v[side]['emotion'][0][1]}, {v[side]['sarcasm']}"
            + ("" if v[side]["right"] else " ✗")
            for side in ("ref", "var")
        ]
        rows.append(f"| {pid} | {v['should_read']} | {cells[0]} | {cells[1]} |")
    return "\n".join(rows) + "\n"


@app.local_entrypoint()
def main(var: str, ref: str, var_prefix: str = "export", ref_prefix: str = "export"):
    res = compare.remote(var, ref, var_prefix, ref_prefix)
    path = save("compare", f"results-compare-{var}-vs-{ref}.json", res)
    table = RESULTS / "compare" / f"table-{var}-vs-{ref}.md"
    table.write_text(markdown(res, ref, var))
    for col, c in res["columns"].items():
        print(
            f"{col}: worse beyond noise {c['worse_beyond_noise'] or 'none'}; better {c['better_beyond_noise'] or 'none'}; "
            f"false sarcasm or face value rises {c['false_sarcasm_or_face_value_rises'] or 'none'}"
        )
        for k, v in c["items"].items():
            print(
                f"  {k}: {v['ref']} -> {v['var']} ({v['diff']:+}, {v['ci'][0]:+} to {v['ci'][1]:+}, "
                f"{v.get('lines', v.get('units'))} lines)"
            )
    probes = res["probes"]
    print(f"probes right: ref {probes['_right']['ref']}, var {probes['_right']['var']} of {probes['_cases']}")
    for pid, v in probes.items():
        if not pid.startswith("_"):
            print(f"  {pid} (should read {v['should_read']}): ref {v['ref']}, var {v['var']}")
    print(f"{res['lines_tagged']} inputs tagged, {res['seconds']} s; wrote", path.relative_to(ROOT))
    print("table:", table.relative_to(ROOT))
