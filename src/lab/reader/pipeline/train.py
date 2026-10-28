"""The reader's students, MiniLM-L6 with every output at once, trained on r1's lines and labels, on r2's (Pipecat
Tone rc1 is r2's seed 1) or on r3's, r2's with the support-call set and r3's teachers.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.train::smoke_r2
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.train::spawn_r2 --seed 1
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.train::spawn_r2 --seed 1 --recipe r3
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.train::spawn_r2 --seed 1 --recipe r3 --no-move --tag nomove
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.train::collect_r2 --runs tonetext-r2-student-minilm-l6-seed1
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.train::smoke
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.train::spawn_one --seed 0   # and 1, 2
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.train::collect
"""

import json

from lab import ROOT
from lab.reader import data, sources, storage
from lab.reader.app import TORCH, app, job, save

ENCODER = "minilm-l6"
# Training settings, recorded in each run's config.
LR, HEAD_LR, BATCH, SMOOTHING, DROP_PREVIOUS = 1e-4, 1e-3, 64, 0.1, 0.2


def write_run(
    rid: str, net, cfg: dict, info: dict, summary: dict, reads: dict, t0: float, recipe: str | None = None
) -> dict:
    """The weights, config and result in the run's folder."""
    import hashlib
    import io
    import time

    import torch

    from lab.reader import model
    from lab.reader.schemas import CONTRACT, EMOTIONS, INTENSITIES, MOVES

    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
    blob = buf.getvalue()
    spec = model.ENCODERS[ENCODER]
    cfg = {
        "run_id": rid,
        "role": "student",
        **({"recipe": recipe} if recipe else {}),
        "encoder": ENCODER,
        "repo": spec["repo"],
        "revision": spec["rev"],
        "render": data.RENDER_VERSION,
        "max_len": data.MAX_LEN,
        "contract": CONTRACT,
        "outputs": list(model.OUTPUTS),
        "emotions": list(EMOTIONS),
        "intensities": list(INTENSITIES),
        "moves": list(MOVES),
        **cfg,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
    }
    storage.put(sources.run_key(rid, "model.pt"), blob)
    storage.put_json(sources.run_key(rid, "config.json"), cfg)
    res = {
        **cfg,
        **info,
        "data": summary,
        "reads": reads,
        "params": sum(p.numel() for p in net.parameters()),
        "seconds": round(time.time() - t0),
    }
    storage.put_json(sources.run_key(rid, "result.json"), res)
    return res


@job(TORCH, gpu="H100", cpu=8.0, memory=65536, timeout=24 * 3600)
def student(
    seed: int,
    epochs: int = 10,
    drop_previous: float = DROP_PREVIOUS,
    sarcasm_teachers: str = ",".join(sources.SARCASM_TEACHERS),
    limit: int = 0,
    tag: str = "",
) -> dict:
    """r1's student on one H100: Tone's label sets, Tone's teachers' labels and r1's sarcasm teachers' labels
    (``data.build``), the best dev epoch kept, dev and test read with and without the line before (test never chosen
    on). `limit` cuts the training lines, for a quick check."""
    import time

    from lab.reader import model

    storage.reload()
    t0 = time.time()
    teachers = tuple(t for t in sarcasm_teachers.split(",") if t)
    missing = [t for t in teachers if not storage.exists(sources.sarcasm_soft_key(t))]
    if missing:
        raise SystemExit(
            f"the sarcasm teachers' labels are not there yet: {', '.join(missing)} (run sarcasm_teacher)"
        )
    lines = data.build(storage.jsonl, sarcasm_teachers=teachers)
    summary = lines.summary()
    print(json.dumps(summary), flush=True)
    net, _, info, reads = model.fit(
        lines,
        ENCODER,
        seed,
        limit,
        epochs=epochs,
        batch_size=BATCH,
        lr=LR,
        head_lr=HEAD_LR,
        smoothing=SMOOTHING,
        drop_previous=drop_previous,
    )
    cfg = {
        "sarcasm_teachers": list(teachers),
        "seed": seed,
        "epochs": epochs,
        "lr": LR,
        "head_lr": HEAD_LR,
        "batch": BATCH,
        "smoothing": SMOOTHING,
        "drop_previous": drop_previous,
        "limit": limit,
    }
    return write_run(sources.run_id("r1", "student", ENCODER, seed, tag), net, cfg, info, summary, reads, t0)


@app.local_entrypoint()
def smoke(limit: int = 3000, epochs: int = 1, sarcasm_teachers: str = ""):
    """r1's job on a few thousand training lines for one epoch, without the sarcasm teachers by default: checks the
    data, the losses and the reads before the real runs."""
    res = student.remote(0, epochs=epochs, limit=limit, sarcasm_teachers=sarcasm_teachers, tag="smoke")
    print(
        json.dumps(
            {k: res[k] for k in ("run_id", "data", "best_epoch", "dev_selection", "seconds")}, indent=1
        )
    )
    print(json.dumps(res["reads"]["test"], indent=1))


@app.local_entrypoint()
def spawn_one(seed: int, epochs: int = 10, drop_previous: float = DROP_PREVIOUS):
    """One r1 student, spawned and left running: launch with `modal run --detach`, one app per seed."""
    call = student.spawn(seed, epochs=epochs, drop_previous=drop_previous)
    print("spawned", sources.run_id("r1", "student", ENCODER, seed), "call", call.object_id)


def summary_line(r: dict) -> str:
    t = r["reads"]["test"]

    def g(col, name, key):
        return (t[col].get(name) or {}).get(key)

    return (
        f"{r['run_id']}: best epoch {r['best_epoch']} dev {r['dev_selection']} | with the line before: emotion "
        f"{g('with_previous', 'emotion_main', 'macro_f1')} speech {g('with_previous', 'emotion_player_speech', 'macro_f1')} "
        f"move {g('with_previous', 'move', 'macro_f1')} sarcasm F1 {g('with_previous', 'sarcasm_hard', 'f1')} "
        f"false {g('with_previous', 'sarcasm_natural', 'false_sarcasm_rate')} | without: emotion "
        f"{g('without_previous', 'emotion_main', 'macro_f1')} move {g('without_previous', 'move', 'macro_f1')} "
        f"sarcasm F1 {g('without_previous', 'sarcasm_hard', 'f1')}"
    )


@app.local_entrypoint()
def collect(seeds: str = "0,1,2", tag: str = "", wait_h: float = 6.0):
    """r1 students' results, waiting for them; saved to results/reader/train/results-students.json."""
    rids = [sources.run_id("r1", "student", ENCODER, int(s), tag) for s in seeds.split(",") if s]
    results = storage.wait_for([sources.run_key(rid, "result.json") for rid in rids], wait_h)
    path = save("train", "results-students.json", {"results": results})
    for r in results:
        print(summary_line(r))
    print("chosen on dev:", max(results, key=lambda r: r["dev_selection"])["run_id"])
    print("wrote", path.relative_to(ROOT))


# --- r2 --------------------------------------------------------------------------------------------------------------


@job(TORCH, gpu="H100", cpu=8.0, memory=98304, timeout=24 * 3600)
def student_r2(
    seed: int,
    epochs: int = 15,
    teachers: str = "",
    soft_id: str = "",
    exclude: str = "",
    speech: bool = True,
    tag: str = "",
    limit: int = 0,
    recipe: str = "r2",
    move: bool = True,
) -> dict:
    """r2's student on one H100: Tone's and r2's committee labels, the r2 teachers' labels on every training line, the
    open sets' draws and the speech twins (``data.build_r2``), the best epoch chosen by r2's rule (Tone's dev sets and
    the voice-agent dev set weigh the same), dev and test read with and without the line before. `exclude goemotions`
    with the `nogo` teachers leaves that source out. `recipe="r3"` adds the support-call set's committee labels and
    reads r3's teachers' labels. Without `move`, the move head is kept, so the export loads where rc1's does, but never
    trained, and the epoch is chosen on emotion and sarcasm alone."""
    import time

    from lab.reader import model

    storage.reload()
    t0 = time.time()
    names = tuple(t for t in (teachers or ",".join(sources.R2_TEACHERS)).split(",") if t)
    sid = soft_id or data.r2_soft_id(recipe=recipe)
    missing = [t for t in names if not storage.exists(sources.r2_soft_key(t, sid))]
    if missing:
        raise SystemExit(
            f"the r2 teachers' labels are not there yet: {', '.join(missing)} (run teacher first)"
        )
    excluded = tuple(x for x in exclude.split(",") if x)
    lines = data.build_r2(
        storage.jsonl,
        teachers=names,
        speech=speech,
        exclude_sources=excluded,
        soft_id=sid,
        support=recipe == "r3",
    )
    summary = lines.summary()
    print(json.dumps(summary), flush=True)
    net, _, info, reads = model.fit(
        lines,
        ENCODER,
        seed,
        limit,
        select=model.selection_score_r2 if move else model.selection_score_r2_without_move,
        groups_fn=data.eval_groups_r2,
        epochs=epochs,
        batch_size=BATCH,
        lr=LR,
        head_lr=HEAD_LR,
        smoothing=SMOOTHING,
        drop_previous=DROP_PREVIOUS,
        **({} if move else {"move": False}),
    )
    cfg = {
        "teachers": list(names),
        **({} if move else {"move_trained": False}),
        "soft_id": sid,
        "exclude_sources": list(excluded),
        "speech": speech,
        "seed": seed,
        "epochs": epochs,
        "lr": LR,
        "head_lr": HEAD_LR,
        "batch": BATCH,
        "smoothing": SMOOTHING,
        "drop_previous": DROP_PREVIOUS,
        "limit": limit,
    }
    rid = sources.run_id(recipe, "student", ENCODER, seed, tag)
    return write_run(rid, net, cfg, info, summary, reads, t0, recipe=recipe)


@app.local_entrypoint()
def spawn_r2(
    seed: int,
    epochs: int = 15,
    exclude: str = "",
    soft_id: str = "",
    tag: str = "",
    recipe: str = "r2",
    move: bool = True,
):
    """One r2 (or r3) student, spawned and left running: launch with `modal run --detach`, one app per run."""
    call = student_r2.spawn(
        seed, epochs=epochs, exclude=exclude, soft_id=soft_id, tag=tag, recipe=recipe, move=move
    )
    print("spawned", sources.run_id(recipe, "student", ENCODER, seed, tag), "call", call.object_id)


@app.local_entrypoint()
def smoke_r2(
    limit: int = 3000,
    epochs: int = 1,
    teachers: str = sources.R2_TEACHERS[0],
    soft_id: str = data.r2_soft_id("smoke"),
):
    """r2's job on a few thousand training lines for one epoch, with the smoke teacher's labels (from
    ``teacher::smoke``) and no speech twins: checks the data, the losses and the reads."""
    res = student_r2.remote(
        0, epochs=epochs, limit=limit, teachers=teachers, soft_id=soft_id, tag="smoke", speech=False
    )
    print(
        json.dumps(
            {k: res[k] for k in ("run_id", "data", "best_epoch", "dev_selection", "seconds")}, indent=1
        )
    )
    print(json.dumps(res["reads"]["test"]["with_previous"], indent=1))


@app.local_entrypoint()
def collect_r2(runs: str = "", wait_h: float = 8.0):
    """r2 students' results by run id (comma-separated), waiting for them; saved to
    results/reader/train/results-students-r2.json."""
    results = storage.wait_for(
        [sources.run_key(rid, "result.json") for rid in runs.split(",") if rid], wait_h
    )
    path = save("train", "results-students-r2.json", {"results": results})
    for r in results:
        t = r["reads"]["test"]["with_previous"]
        g = lambda k, m: (t.get(k) or {}).get(m)  # noqa: E731
        print(
            f"{r['run_id']}: best epoch {r['best_epoch']} dev {r['dev_selection']} | emotion "
            f"{g('emotion_main', 'macro_f1')} move {g('move', 'macro_f1')} sarcasm {g('sarcasm_hard', 'f1')} | voice "
            f"emotion {g('voice_emotion', 'macro_f1')} move {g('voice_move', 'macro_f1')}"
        )
    print("wrote", path.relative_to(ROOT))
