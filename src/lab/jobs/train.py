"""Exports corpus batches to rows and trains a tagger on them together, on a GPU.

The model's folder goes to the ``tone-tagger-models`` volume. The sentence pass is timed on the
container's CPU, a rough guide only.
"""

import json
from pathlib import Path

from lab.jobs.common import CORPUS_DIR, MODELS_DIR, app, corpus, image_with, models


@app.function(volumes={CORPUS_DIR: corpus}, timeout=30 * 60, cpu=4.0, memory=8192)
def export(batch: str) -> dict:
    from lab.corpus.export import export as export_rows

    corpus.reload()
    stats = export_rows(Path(CORPUS_DIR) / batch)
    corpus.commit()
    return stats


@app.function(
    image=image_with(["lab", "train"]),
    gpu="L4",
    volumes={CORPUS_DIR: corpus, MODELS_DIR: models},
    timeout=4 * 60 * 60,
    cpu=8.0,
    memory=32768,
)
def train(
    batch: str,
    backbone: str,
    name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    cut_call_first: bool = False,
    seed: int = 0,
    layout_options: dict | None = None,
    speech_stats: dict | None = None,
    breaths_per_reply: int | None = None,
    init_seed: int | None = None,
    delivery_balance: bool = False,
) -> dict:
    """Train on the batches' rows, export, calibrate its breaths, score through the runtime and time the sentence
    pass."""
    from lab import quiet_logs
    from lab.corpus.rows import load_rows
    from lab.train.calibrate import calibrate
    from lab.train.evaluate import benchmark, predictor, score
    from lab.train.fit import fit
    from pipecat_tone.tagger import RulesTagger
    from pipecat_tone.tagger.features import Layout

    quiet_logs()
    corpus.reload()
    folders = [Path(CORPUS_DIR) / b for b in batch.split(",")]
    train_rows = [r for f in folders for r in load_rows(f / "train.jsonl")]
    dev_rows = [r for f in folders for r in load_rows(f / "dev.jsonl")]
    out = Path(MODELS_DIR) / name
    layout = Layout(cut_call_first=cut_call_first, **(layout_options or {}))
    fit(
        train_rows,
        dev_rows,
        backbone,
        out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        layout=layout,
        seed=seed,
        init_seed=init_seed,
        delivery_balance=delivery_balance,
    )
    calibrate(out, dev_rows, speech_stats, breaths_per_reply)
    config = json.loads((out / "config.json").read_text())
    config.update(
        rules_dev=score(dev_rows, predictor(RulesTagger())),
        cpu_latency=[benchmark(out, dev_rows, threads=t) for t in (1, 2)],
        batch=batch,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    models.commit()
    return {
        k: config[k]
        for k in ("epoch", "shipped_dev", "rules_dev", "cpu_latency", "onnx", "train_rows", "dev_rows")
    }


@app.local_entrypoint()
def train_tagger(
    batch: str = "v0",
    backbone: str = "jhu-clsp/ettin-encoder-17m",
    name: str = "",
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 32,
    skip_export: bool = False,
    cut_call_first: bool = False,
    seed: int = 0,
    layout: str = "",
    speech_breaths: bool = False,
    breaths_per_reply: int = 0,
    init_seed: int = -1,
    delivery_balance: bool = False,
):
    """``--cut-call-first`` trains with ``Layout.cut_call_first``; ``--layout`` sets more of the
    ``Layout``, as JSON. ``--speech-breaths`` calibrates breaths to real speakers' rate rather than the teacher's,
    ``--breaths-per-reply`` caps a reply's breaths, ``--init-seed`` starts the heads from another seed's
    weights, so seeds that share it can be averaged, and ``--delivery-balance`` weighs rare deliveries up."""
    if not skip_export:
        for stats in export.map(batch.split(",")):
            print(json.dumps(stats, indent=2), flush=True)
    name = name or f"{backbone.split('/')[-1]}-{batch.replace(',', '+')}"
    # Spawned, so with --detach the training goes on if this client stops; its folder holds the result.
    options = json.loads(layout) if layout else None
    # Real speech's statistics come from here, as the job's image holds only the code.
    from lab.corpus.speech_breaths import icsi_turns, summary

    stats = summary(icsi_turns()) if speech_breaths else None
    call = train.spawn(
        batch,
        backbone,
        name,
        epochs,
        lr,
        batch_size,
        cut_call_first,
        seed,
        options,
        stats,
        breaths_per_reply or None,
        None if init_seed < 0 else init_seed,
        delivery_balance,
    )
    print(f"training {name} as {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
    print(
        f"fetch with: MODAL_PROFILE=daily uv run modal volume get tone-tagger-models {name} models/ --env tone-tagger"
    )


@app.local_entrypoint()
def export_batches(batch: str):
    """Export batches, separated by commas, to rows without training."""
    for stats in export.map(batch.split(",")):
        print(json.dumps(stats, indent=2), flush=True)
