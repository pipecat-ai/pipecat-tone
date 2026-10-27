"""r2's text-only teachers, the student's model with Ettin-1B as its encoder, trained on committee labels only, and
their labels on every training line.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.teacher::smoke
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.teacher::spawn_one --seed 0   # 0 to 4
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.teacher::spawn_one --seed 0 --exclude goemotions --tag nogo
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.teacher::collect [--tag nogo]
"""

import json

from lab import ROOT
from lab.reader import sources, storage
from lab.reader.app import TORCH, app, job, save

ENCODER = "ettin-1b"
# Training settings, recorded in each result; the learning rate is model.TEACHER_LR's.
HEAD_LR, BATCH, SMOOTHING, DROP_PREVIOUS = 1e-3, 32, 0.1, 0.2


@job(TORCH, gpu="H100", cpu=8.0, memory=65536, timeout=24 * 3600)
def teacher(
    seed: int,
    epochs: int = 6,
    exclude: str = "",
    tag: str = "",
    limit_train: int = 0,
    limit_pool: int = 0,
) -> dict:
    """One teacher on one H100: every output at once, trained on committee labels only (Tone's sets and r2's,
    ``data.build_r2`` without pools) and chosen on dev by r2's rule; test is read, never chosen on. Its labels on every
    clean training line the student reads (``data.r2_lines_to_label``) go under the tag's soft id, its result to the
    run's folder; its weights are not kept. `exclude` leaves those sources out. `limit_train` and `limit_pool` cut the
    training lines and the lines labelled, for a quick check of the job."""
    import time
    from collections import Counter

    from lab.reader import data, model

    storage.reload()
    t0 = time.time()
    excluded = tuple(x for x in exclude.split(",") if x)
    lines = data.build_r2(storage.jsonl, pools=False, exclude_sources=excluded)
    lr = model.TEACHER_LR.get(ENCODER, model.DEFAULT_TEACHER_LR)
    net, tok, info, reads = model.fit(
        lines,
        ENCODER,
        seed,
        limit_train,
        labelled_only=True,
        dev_without_previous=False,
        select=model.selection_score_r2,
        groups_fn=data.eval_groups_r2,
        epochs=epochs,
        batch_size=BATCH,
        lr=lr,
        head_lr=HEAD_LR,
        smoothing=SMOOTHING,
        drop_previous=DROP_PREVIOUS,
    )
    pool = data.r2_lines_to_label(storage.jsonl, exclude_sources=excluded)
    if limit_pool:
        pool = pool[:limit_pool]
    exs = [data.example(ln, "train") for ln in pool]
    P = model.predict(net, tok, exs, info["device"], True, batch_size=512)
    rnd = lambda v: [round(float(x), 4) for x in v]  # noqa: E731
    soft_id = data.r2_soft_id(tag)
    storage.put_jsonl(
        sources.r2_soft_key(f"teacher-{ENCODER}-seed{seed}", soft_id),
        (
            {
                "line_id": e.line_id,
                "emo": rnd(P["emotion"][i]),
                "emo_int": rnd(P["emotion_intensity"][i]),
                "back": round(float(P["backchannel"][i][1]), 4),
                "move": rnd(P["move"][i]),
                "sarc": round(float(P["sarcasm"][i][1]), 4),
            }
            for i, e in enumerate(exs)
        ),
    )
    rid = sources.run_id("r2", "teacher", ENCODER, seed, tag)
    res = {
        "run_id": rid,
        "encoder": ENCODER,
        "seed": seed,
        "exclude": list(excluded),
        "soft_id": soft_id,
        "lr": lr,
        "head_lr": HEAD_LR,
        "epochs": epochs,
        "batch": BATCH,
        "drop_previous": DROP_PREVIOUS,
        **info,
        "data": lines.summary(),
        "reads": reads,
        "labelled_lines": len(exs),
        "labelled_by_source": dict(Counter(e.source for e in exs).most_common()),
        "params": sum(p.numel() for p in net.parameters()),
        "seconds": round(time.time() - t0),
    }
    storage.put_json(sources.run_key(rid, "result.json"), res)
    return res


@app.local_entrypoint()
def spawn_one(seed: int, epochs: int = 6, exclude: str = "", tag: str = ""):
    """One teacher, spawned and left running: launch with `modal run --detach`, one app per seed."""
    call = teacher.spawn(seed, epochs=epochs, exclude=exclude, tag=tag)
    print("spawned", sources.run_id("r2", "teacher", ENCODER, seed, tag), "call", call.object_id)


@app.local_entrypoint()
def collect(seeds: str = "0,1,2,3,4", tag: str = "", wait_h: float = 4.0):
    """The teachers' results, waiting for them; saved to results/reader/teacher/results-teachers[-tag].json."""
    rids = [sources.run_id("r2", "teacher", ENCODER, int(s), tag) for s in seeds.split(",") if s]
    results = storage.wait_for([sources.run_key(rid, "result.json") for rid in rids], wait_h)
    path = save("teacher", f"results-teachers{('-' + tag) if tag else ''}.json", {"results": results})
    for r in results:
        t, d = r["reads"]["test"]["with_previous"], r["reads"]["dev"]["with_previous"]
        g = lambda x, k, m: (x.get(k) or {}).get(m)  # noqa: E731
        print(
            f"{r['run_id']}: best epoch {r['best_epoch']} dev {r['dev_selection']} | test emotion "
            f"{g(t, 'emotion_main', 'macro_f1')} move {g(t, 'move', 'macro_f1')} sarcasm {g(t, 'sarcasm_hard', 'f1')} "
            f"| voice test emotion {g(t, 'voice_emotion', 'macro_f1')} move {g(t, 'voice_move', 'macro_f1')} | dev voice "
            f"emotion {g(d, 'voice_emotion', 'macro_f1')} | {r['seconds']} s"
        )
    print("wrote", path.relative_to(ROOT))


@app.local_entrypoint()
def smoke(limit_train: int = 2000, limit_pool: int = 3000):
    """One teacher on a few thousand lines for one epoch, labelling a few thousand pool lines, under the `smoke` tag."""
    res = teacher.remote(0, epochs=1, tag="smoke", limit_train=limit_train, limit_pool=limit_pool)
    keys = ("run_id", "soft_id", "best_epoch", "dev_selection", "labelled_lines", "params", "seconds")
    print(json.dumps({k: res[k] for k in keys}, indent=1))
    print(json.dumps(res["data"], indent=1)[:1500])
