"""r3c from its contrast pairs to its comparisons, in one job on Modal that no laptop has to watch: the pairs written
(``support.build_contrast_prompts``, one H100) and selected, the committee's pilot and r3's gate on it, every line's
labels, two students on r3's mix with the pairs added and r2's teachers' labels (``train.student_r2 --recipe r3c``),
their exports, and each compared with rc1 and with r3's student on r2's teachers. Each step is skipped when its output
is already on the volume, so a rerun picks up where one stopped. Its records go to runs/tonetext/r3c-chain/ on the
volume; `fetch` copies them, counts and scores only, to results/reader/.

    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.r3c_chain::run
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.r3c_chain::fetch
"""

import json
import time

from lab.jobs.common import app
from lab.reader import sources, storage, support
from lab.reader.app import DATA, RESULTS, job, save
from lab.reader.committee import label as committee_label

# The stages' jobs, without their entrypoints, whose names repeat from stage to stage: this job runs them in turn. Its
# own job has a name of its own, as r3's chain is on the same app.
_local_entrypoint = app.local_entrypoint
app.local_entrypoint = lambda *a, **k: lambda f: f
try:
    from lab.reader.pipeline import compare, export, r3_chain, support_generate, support_label, train
finally:
    app.local_entrypoint = _local_entrypoint

SET = sources.CONTRAST_ID
TRY = 1
STUDENTS = ({"seed": 1}, {"seed": 2})
R3 = r3_chain.student_id(r3_chain.STUDENTS[2])  # r3's student on r2's teachers, the best of r3
REFS = {r3_chain.REF: "rc1", R3: "r3 on r2's teachers"}
RECORDS = "runs/tonetext/r3c-chain"


def record_key(name: str) -> str:
    return f"{RECORDS}/{name}"


def student_id(s: dict) -> str:
    return sources.run_id("r3c", "student", train.ENCODER, s["seed"], s.get("tag", ""))


def labels_summary_key(sets: dict) -> str:
    return sources.tonetext_labels_key(sets["committee"], "combined_summary.json")


@job(DATA, cpu=1.0, memory=4096, timeout=24 * 3600, nonpreemptible=True)
def contrast_chain(skip_gate: bool = False) -> dict:
    """Every step in turn, each recorded as it ends in runs/tonetext/r3c-chain/progress.json; a step whose output is on
    the volume is not run again. Stops after the pilot when it fails r3's gate, unless `skip_gate`."""
    t0 = time.time()
    progress: dict = r3_chain.summary_of(record_key("progress.json")) or {}

    def note(step: str, res: dict) -> None:
        progress[step] = {"at_seconds": round(time.time() - t0), **res}
        storage.put_json(record_key("progress.json"), progress)

    try:
        written = r3_chain.summary_of(support_generate.contrast_raw_key(TRY, "result.json"))
        if written is None:
            written = support_generate.write_contrast.remote(
                support_generate.CONTRAST_PER_DOMAIN, support.CONTRAST_SEED, TRY
            )
        note(
            "generate",
            {k: written.get(k) for k in ("prompts", "moments", "usable_pairs", "problems", "seconds")},
        )

        summary = r3_chain.summary_of(sources.contrast_key("summary.json"))
        if summary is None:
            summary = support_generate.select_contrast_lines.remote(support_generate.CONTRAST_LINES, TRY)[
                "summary"
            ]
        note(
            "select",
            {
                k: summary.get(k)
                for k in ("lines", "pairs", "by_split", "dropped", "trained_elsewhere", "sha256")
            },
        )

        pilot = r3_chain.summary_of(labels_summary_key(support_label.pilot_sets_of(SET)))
        if pilot is None:
            systems, meta = committee_label.prepare(
                "pilot", support_label.QUESTIONS, **support_label.keys_of(SET)
            )
            pilot = support_label.label_pilot.remote(systems, meta, SET)
        g = r3_chain.gate(pilot)
        note("pilot", {"lines": pilot["lines"], "cost_usd": pilot["cost_usd_total"], "gate": g})
        if not g["pass"] and not skip_gate:
            note("stopped", {"why": "the pilot failed the gate"})
            return progress

        labelled = r3_chain.summary_of(labels_summary_key(support_label.sets_of(SET)))
        if labelled is None:
            systems, meta = committee_label.prepare(
                "label", support_label.QUESTIONS, **support_label.keys_of(SET)
            )
            labelled = support_label.label_all.remote(systems, meta, set_id=SET)
        s = labelled["sarcasm"]
        note(
            "label",
            {
                "lines": labelled["lines"],
                "cost_usd": labelled["cost_usd_total"],
                "sarcasm_majority": s.get("majority"),
                "a_b_sarcasm": (s.get("a_b") or {}).get("committee_level"),
                "by_construction": (s.get("by_construction") or {}).get("_all"),
            },
        )

        calls = {}
        for st in STUDENTS:
            rid = student_id(st)
            if not storage.exists(sources.run_key(rid, "result.json")):
                calls[rid] = train.student_r2.spawn(st["seed"], recipe="r3c")
        for rid, call in calls.items():
            call.get()
            note(f"trained {rid}", {})

        for st in STUDENTS:
            rid = student_id(st)
            if not storage.exists(sources.run_key(rid, "export", "export_result.json")):
                export.export.remote(rid, recipe="r3c", sarcasm_cap=0.015)
            note(f"exported {rid}", {})

        compares = {
            (student_id(st), ref): compare.compare.spawn(student_id(st), ref)
            for st in STUDENTS
            for ref in REFS
        }
        for (rid, ref), call in compares.items():
            res = call.get()
            name = f"{rid}-vs-{ref}"
            storage.put_json(record_key(f"compare-{name}.json"), res)
            storage.put(record_key(f"table-{name}.md"), compare.markdown(res, REFS[ref], rid).encode())
            c = res["columns"]["with_previous"]
            note(
                f"compared {rid} with {ref}",
                {
                    "worse_beyond_noise": c["worse_beyond_noise"],
                    "better_beyond_noise": c["better_beyond_noise"],
                    "false_sarcasm_or_face_value_rises": c["false_sarcasm_or_face_value_rises"],
                    "probes_right": res["probes"]["_right"],
                    "praise_right": {
                        k: {side: v[side]["right"] for side in ("ref", "var")}
                        for k, v in res["praise"]["groups"].items()
                    },
                },
            )
        note("done", {})
    except Exception as exc:
        note("failed", {"error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        raise
    return progress


@app.local_entrypoint()
def run(skip_gate: bool = False):
    """The chain, spawned to outlive this client under `modal run --detach`."""
    call = contrast_chain.spawn(skip_gate)
    save("r3c_chain", "spawned.json", {"call": call.object_id, "skip_gate": skip_gate})
    print("spawned", call.object_id)


@app.local_entrypoint()
def fetch():
    """What the chain has recorded so far, saved under results/reader/ as each stage saves its own: counts and scores,
    never line text."""
    progress = r3_chain.summary_of(record_key("progress.json")) or {}
    save("r3c_chain", "progress.json", progress)
    print(json.dumps(progress, indent=1))
    found = {
        ("support_generate", f"contrast-result-try{TRY}.json"): support_generate.contrast_raw_key(
            TRY, "result.json"
        ),
        ("support_generate", f"summary-{SET}-try{TRY}.json"): sources.contrast_key("summary.json"),
        ("support_label", f"{SET}-pilot.json"): labels_summary_key(support_label.pilot_sets_of(SET)),
        ("support_label", f"{SET}-summary.json"): labels_summary_key(support_label.sets_of(SET)),
    }
    for st in STUDENTS:
        rid = student_id(st)
        found[("train", f"result-{rid}.json")] = sources.run_key(rid, "result.json")
        found[("export", f"results-{rid}.json")] = sources.run_key(rid, "export", "export_result.json")
        for ref in REFS:
            found[("compare", f"results-compare-{rid}-vs-{ref}.json")] = record_key(
                f"compare-{rid}-vs-{ref}.json"
            )
    for (folder, name), key in found.items():
        obj = r3_chain.summary_of(key)
        if obj is not None:
            path = save(folder, name, obj)
            print("wrote", path.relative_to(RESULTS.parent.parent))
    for st in STUDENTS:
        for ref in REFS:
            name = f"{student_id(st)}-vs-{ref}"
            key = record_key(f"table-{name}.md")
            if storage.exists(key):
                path = RESULTS / "compare" / f"table-{name}.md"
                path.write_bytes(storage.get(key))
                print("wrote", path.relative_to(RESULTS.parent.parent))
