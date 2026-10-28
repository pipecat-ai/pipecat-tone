"""r3 from the support-call set's calls to its comparison with rc1, in one job on Modal that no laptop has to watch:
the set's lines, the committee's pilot and a gate on it, every line's labels, five teachers, three students (r3 at
rc1's seed and one more, and r3's labels with r2's teachers), their exports and each one compared with rc1. Each step
is skipped when its output is already on the volume, so a rerun picks up where one stopped. `no_move` adds a fourth
student beside it: r3 with the move left untrained, compared with rc1 and with r3 at the same seed. Its records go to
runs/tonetext/r3-chain/ on the volume; `fetch` copies them, counts and scores only, to results/reader/.

    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.r3_chain::run
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.r3_chain::no_move
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.r3_chain::fetch
"""

import json
import time

from lab.jobs.common import app
from lab.reader import data, sources, storage
from lab.reader.app import DATA, RESULTS, job, save
from lab.reader.committee import label as committee_label

# The stages' jobs, without their entrypoints, whose names repeat from stage to stage: this job runs them in turn.
_local_entrypoint = app.local_entrypoint
app.local_entrypoint = lambda *a, **k: lambda f: f
try:
    from lab.reader.pipeline import compare, export, support_generate, support_label, teacher, train
finally:
    app.local_entrypoint = _local_entrypoint

REF = "tonetext-r2-student-minilm-l6-seed1"  # rc1
TRY = 1
TEACHER_SEEDS = range(5)
# r3 at rc1's seed and one more, then r3's labels read with r2's teachers.
STUDENTS = (
    {"seed": 1},
    {"seed": 2},
    {"seed": 1, "soft_id": sources.R2_SOFT_ID, "tag": "r2teachers"},
)
# r3 at rc1's seed with its move head kept but never trained, and its epoch chosen on emotion and sarcasm.
NO_MOVE = {"seed": 1, "tag": "nomove"}
# The pilot's gate before every line is paid for: member answers lost, A and B's sarcasm agreement, and the committee's
# sarcasm against what each line was written to be.
GATE = {"errors_per_answer": 0.05, "a_b_sarcasm_agreement": 0.7, "by_construction_agreement": 0.7}
RECORDS = "runs/tonetext/r3-chain"


def record_key(name: str) -> str:
    return f"{RECORDS}/{name}"


def student_id(s: dict) -> str:
    return sources.run_id("r3", "student", train.ENCODER, s["seed"], s.get("tag", ""))


def wait(keys: list[str], hours: float) -> None:
    """Until every key is on the volume, reloading it to see other jobs' commits."""
    end = time.time() + hours * 3600
    while True:
        storage.reload()
        missing = [k for k in keys if not storage.exists(k)]
        if not missing:
            return
        if time.time() > end:
            raise RuntimeError(f"not there after {hours} h: {missing}")
        time.sleep(60)


def gate(pilot: dict) -> dict:
    """Whether the pilot's labels pass GATE, and the readings it was judged on."""
    s = pilot["sarcasm"]
    answers = 3 * max(1, pilot["lines"])
    errors = {q: round(sum(pilot[q]["errors"].values()) / answers, 4) for q in support_label.SETS}
    ab = ((s.get("a_b") or {}).get("committee_level") or {}).get("agreement") or 0.0
    built = (s.get("by_construction") or {}).get("_all") or {}
    agree = built.get("agree", 0) / max(1, built.get("agree", 0) + built.get("disagree", 0))
    ok = (
        all(e <= GATE["errors_per_answer"] for e in errors.values())
        and ab >= GATE["a_b_sarcasm_agreement"]
        and agree >= GATE["by_construction_agreement"]
    )
    return {
        "pass": ok,
        "errors_per_answer": errors,
        "a_b_sarcasm_agreement": ab,
        "by_construction_agreement": round(agree, 4),
        "thresholds": GATE,
    }


def summary_of(key: str) -> dict | None:
    return json.loads(storage.get(key)) if storage.exists(key) else None


@job(DATA, cpu=1.0, memory=4096, timeout=24 * 3600, nonpreemptible=True)
def chain(skip_gate: bool = False) -> dict:
    """Every step in turn, each recorded as it ends in runs/tonetext/r3-chain/progress.json; a step whose output is on
    the volume is not run again. Stops after the pilot when it fails the gate, unless `skip_gate`."""
    t0 = time.time()
    progress: dict = summary_of(record_key("progress.json")) or {}

    def note(step: str, res: dict) -> None:
        progress[step] = {"at_seconds": round(time.time() - t0), **res}
        storage.put_json(record_key("progress.json"), progress)

    try:
        # the calls, written by support_generate::generate
        shards = support_generate.SHARDS
        wait([support_generate.raw_key(TRY, f"result-{k}.json") for k in range(shards)], 4)
        note("generate", {"shards": shards})

        summary = summary_of(sources.support_key("summary.json"))
        if summary is None:
            summary = support_generate.select_lines.remote(support_generate.MAX_LINES, TRY, shards)["summary"]
        note("select", {k: summary.get(k) for k in ("lines", "by_split", "calls", "dropped", "sha256")})

        pilot_key = sources.tonetext_labels_key(
            support_label.PILOT_SETS["committee"], "combined_summary.json"
        )
        pilot = summary_of(pilot_key)
        if pilot is None:
            systems, meta = committee_label.prepare("pilot", support_label.QUESTIONS, **support_label.KEYS)
            pilot = support_label.label_pilot.remote(systems, meta)
        g = gate(pilot)
        note("pilot", {"lines": pilot["lines"], "cost_usd": pilot["cost_usd_total"], "gate": g})
        if not g["pass"] and not skip_gate:
            note("stopped", {"why": "the pilot failed the gate"})
            return progress

        labelled_key = sources.tonetext_labels_key(support_label.SETS["committee"], "combined_summary.json")
        labelled = summary_of(labelled_key)
        if labelled is None:
            systems, meta = committee_label.prepare("label", support_label.QUESTIONS, **support_label.KEYS)
            labelled = support_label.label_all.remote(systems, meta)
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

        # the teachers, and the student that reads r2's teachers, at once
        calls = {}
        for k in TEACHER_SEEDS:
            rid = sources.run_id("r3", "teacher", teacher.ENCODER, k)
            if not storage.exists(sources.run_key(rid, "result.json")):
                calls[rid] = teacher.teacher.spawn(k, recipe="r3")
        ablation = STUDENTS[2]
        if not storage.exists(sources.run_key(student_id(ablation), "result.json")):
            calls[student_id(ablation)] = train.student_r2.spawn(
                ablation["seed"], soft_id=ablation["soft_id"], tag=ablation["tag"], recipe="r3"
            )
        for rid, call in calls.items():
            call.get()
            note(f"trained {rid}", {})

        calls = {}
        for st in STUDENTS[:2]:
            if not storage.exists(sources.run_key(student_id(st), "result.json")):
                calls[student_id(st)] = train.student_r2.spawn(st["seed"], recipe="r3")
        for rid, call in calls.items():
            call.get()
            note(f"trained {rid}", {})

        for st in STUDENTS:
            rid = student_id(st)
            if not storage.exists(sources.run_key(rid, "export", "export_result.json")):
                export.export.remote(rid, recipe="r3", sarcasm_cap=0.015)
            note(f"exported {rid}", {})

        for st in STUDENTS:
            rid = student_id(st)
            res = compare.compare.remote(rid, REF)
            storage.put_json(record_key(f"compare-{rid}.json"), res)
            storage.put(record_key(f"table-{rid}.md"), compare.markdown(res, "rc1", rid).encode())
            c = res["columns"]["with_previous"]
            note(
                f"compared {rid}",
                {
                    "worse_beyond_noise": c["worse_beyond_noise"],
                    "better_beyond_noise": c["better_beyond_noise"],
                    "false_sarcasm_or_face_value_rises": c["false_sarcasm_or_face_value_rises"],
                    "probes_right": res["probes"]["_right"],
                },
            )
        note("done", {})
    except Exception as exc:
        note("failed", {"error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        raise
    return progress


@job(DATA, cpu=1.0, memory=4096, timeout=24 * 3600, nonpreemptible=True)
def no_move_arm() -> dict:
    """The no-move student, once r3's teachers have labelled every line: trained, exported as r3 is, and compared with
    rc1 and, once its export is there, with r3 at the same seed, each step recorded in progress-nomove.json."""
    t0 = time.time()
    progress: dict = summary_of(record_key("progress-nomove.json")) or {}

    def note(step: str, res: dict) -> None:
        progress[step] = {"at_seconds": round(time.time() - t0), **res}
        storage.put_json(record_key("progress-nomove.json"), progress)

    rid = student_id(NO_MOVE)
    try:
        soft = data.r2_soft_id(recipe="r3")
        wait([sources.r2_soft_key(t, soft) for t in sources.R2_TEACHERS], 20)
        note("teachers", {})
        if not storage.exists(sources.run_key(rid, "result.json")):
            train.student_r2.remote(NO_MOVE["seed"], tag=NO_MOVE["tag"], recipe="r3", move=False)
        note(f"trained {rid}", {})
        if not storage.exists(sources.run_key(rid, "export", "export_result.json")):
            export.export.remote(rid, recipe="r3", sarcasm_cap=0.015)
        note(f"exported {rid}", {})
        r3 = student_id(STUDENTS[0])
        for ref, name in ((REF, rid), (r3, f"{rid}-vs-{r3}")):
            wait([sources.run_key(ref, "export", "export_result.json")], 12)
            res = compare.compare.remote(rid, ref)
            storage.put_json(record_key(f"compare-{name}.json"), res)
            storage.put(record_key(f"table-{name}.md"), compare.markdown(res, ref, rid).encode())
            c = res["columns"]["with_previous"]
            note(
                f"compared {rid} with {ref}",
                {
                    "worse_beyond_noise": c["worse_beyond_noise"],
                    "better_beyond_noise": c["better_beyond_noise"],
                    "probes_right": res["probes"]["_right"],
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
    call = chain.spawn(skip_gate)
    save("r3_chain", "spawned.json", {"call": call.object_id, "skip_gate": skip_gate})
    print("spawned", call.object_id)


@app.local_entrypoint()
def no_move():
    """The no-move student's job, spawned to outlive this client under `modal run --detach`."""
    call = no_move_arm.spawn()
    save("r3_chain", "no-move-spawned.json", {"call": call.object_id})
    print("spawned", call.object_id)


@app.local_entrypoint()
def fetch():
    """What the chain has recorded so far, saved under results/reader/ as each stage saves its own: counts and scores,
    never line text."""
    progress = summary_of(record_key("progress.json")) or {}
    save("r3_chain", "progress.json", progress)
    print(json.dumps(progress, indent=1))
    no_move_progress = summary_of(record_key("progress-nomove.json"))
    if no_move_progress:
        save("r3_chain", "progress-nomove.json", no_move_progress)
        print(json.dumps(no_move_progress, indent=1))
    found = {
        ("support_generate", f"summary-{sources.SUPPORT_ID}-try{TRY}.json"): sources.support_key(
            "summary.json"
        ),
        ("support_label", "pilot.json"): sources.tonetext_labels_key(
            support_label.PILOT_SETS["committee"], "combined_summary.json"
        ),
        ("support_label", "summary.json"): sources.tonetext_labels_key(
            support_label.SETS["committee"], "combined_summary.json"
        ),
    }
    for k in TEACHER_SEEDS:
        rid = sources.run_id("r3", "teacher", teacher.ENCODER, k)
        found[("teacher", f"result-{rid}.json")] = sources.run_key(rid, "result.json")
    r3, no_move_id = student_id(STUDENTS[0]), student_id(NO_MOVE)
    for st in (*STUDENTS, NO_MOVE):
        rid = student_id(st)
        found[("train", f"result-{rid}.json")] = sources.run_key(rid, "result.json")
        found[("export", f"results-{rid}.json")] = sources.run_key(rid, "export", "export_result.json")
        found[("compare", f"results-compare-{rid}-vs-{REF}.json")] = record_key(f"compare-{rid}.json")
    found[("compare", f"results-compare-{no_move_id}-vs-{r3}.json")] = record_key(
        f"compare-{no_move_id}-vs-{r3}.json"
    )
    for (folder, name), key in found.items():
        obj = summary_of(key)
        if obj is not None:
            path = save(folder, name, obj)
            print("wrote", path.relative_to(RESULTS.parent.parent))
    tables = {student_id(st): REF for st in (*STUDENTS, NO_MOVE)}
    tables[f"{no_move_id}-vs-{r3}"] = None
    for name, ref in tables.items():
        key = record_key(f"table-{name}.md")
        if storage.exists(key):
            path = RESULTS / "compare" / (f"table-{name}-vs-{ref}.md" if ref else f"table-{name}.md")
            path.write_bytes(storage.get(key))
            print("wrote", path.relative_to(RESULTS.parent.parent))
