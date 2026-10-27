"""Builds corpus calls, one container per call, saved to the ``tone-tagger-corpus`` volume.

Specs are seeded, so ``--start`` and ``--n`` pick the same calls every time, and saved calls are
skipped. Reteaching and reviewing again touch only calls done by an older teacher or rubric.
"""

import asyncio
import json
from pathlib import Path

import modal

from lab.jobs.common import CORPUS_DIR, app, corpus

# How many calls run at once; each makes a handful of slow requests.
CONCURRENCY = 48


@app.function(volumes={CORPUS_DIR: corpus}, timeout=45 * 60, max_containers=CONCURRENCY, cpu=1.0, memory=2048)
def run_stage(item: dict | str, stage: str, batch: str, into: str = "") -> dict:
    """Build a call from its spec, or reteach, review again or breathe a saved call by id; save it, under ``into``
    for a trial that leaves the batch as it is, and return a summary."""
    from lab import quiet_logs
    from lab.corpus import build
    from lab.corpus.specs import Spec

    quiet_logs()
    corpus.reload()
    if stage == "build":
        result = asyncio.run(build.build_call(Spec(**item)))
        # A failed build is kept apart, so the next run builds the call again.
        path = Path(CORPUS_DIR) / batch / ("errors" if result.get("error") else "") / f"{item['id']}.json"
    else:
        path = Path(CORPUS_DIR) / batch / f"{item}.json"
        relabel = {
            "reteach": build.reteach_call,
            "rereview": build.rereview_call,
            "breathe": build.breathe_call,
        }
        result = asyncio.run(relabel[stage](json.loads(path.read_text())))
        if into:
            path = Path(CORPUS_DIR) / into / batch / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    corpus.commit()
    return _summary(result)


@app.local_entrypoint()
def build_corpus(batch: str, n: int, start: int = 0):
    from lab.corpus.specs import make_spec

    try:
        done = {Path(e.path).stem for e in corpus.listdir(batch)}
    except modal.exception.NotFoundError:
        done = set()
    specs = [
        s for s in (make_spec(seed).to_dict() for seed in range(start, start + n)) if s["id"] not in done
    ]
    print(f"{batch}: building {len(specs)} calls ({len(done)} already done)", flush=True)
    _run("build", batch, specs)


@app.local_entrypoint()
def reteach_corpus(batch: str):
    _run("reteach", batch, _stale(batch, "teacher"))


@app.local_entrypoint()
def rereview_corpus(batch: str):
    _run("rereview", batch, _stale(batch, "reviewer"))


@app.local_entrypoint()
def breathe_corpus(batch: str, n: int = 0, trial: bool = False):
    """Breath variants for the saved calls taught without BREATHE (``lab.corpus.breaths``), in id order; ``--n``
    caps them. ``--trial`` saves them under ``trials/breaths-v<version>/`` instead, leaving the batch as it is."""
    from lab.corpus.breaths import BREATH_VERSION, needs_breaths
    from lab.corpus.build import is_call

    due = sorted(
        Path(entry.path).stem
        for entry in corpus.listdir(batch)
        if is_call(entry.path) and needs_breaths(json.loads(b"".join(corpus.read_file(entry.path))))
    )
    print(f"{batch}: {len(due)} calls need a breath pass", flush=True)
    _run("breathe", batch, due[:n] if n else due, f"trials/breaths-v{BREATH_VERSION}" if trial else "")


def _stale(batch: str, role: str) -> list[str]:
    """Saved calls whose teacher or reviewer is older than they need, or whose teacher failed."""
    from lab.corpus.build import is_call, is_stale

    def due(call: dict) -> bool:
        failed = role == "teacher" and bool((call.get("teacher") or {}).get("error"))
        return failed or is_stale(call, role)

    stale = [
        Path(entry.path).stem
        for entry in corpus.listdir(batch)
        if is_call(entry.path) and due(json.loads(b"".join(corpus.read_file(entry.path))))
    ]
    print(f"{batch}: {len(stale)} calls with a failed or older {role} than they need", flush=True)
    return stale


def _run(stage: str, batch: str, items: list, into: str = ""):
    totals = {
        "calls": 0,
        "errors": 0,
        "replies": 0,
        "passed": 0,
        "sentences": 0,
        "breaths": 0,
        "cost_usd": 0.0,
    }
    for summary in run_stage.map(
        items,
        kwargs={"stage": stage, "batch": batch, "into": into},
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(summary, Exception):
            totals["errors"] += 1
            print(f"  crashed: {summary}", flush=True)
            continue
        totals["calls"] += 1
        totals["errors"] += bool(summary["error"])
        for k in ("replies", "passed", "sentences", "breaths", "cost_usd"):
            totals[k] += summary[k]
        print(
            f"  {summary['id']} {summary['split']:5} {summary['agent_model']:18} {summary['tts']:10} "
            f"replies {summary['passed']}/{summary['replies']} passed, ${summary['cost_usd']:.3f}"
            + (f"  ERROR {summary['error']}" if summary["error"] else ""),
            flush=True,
        )
    totals["cost_usd"] = round(totals["cost_usd"], 2)
    print(f"{batch}: {totals}", flush=True)


def _summary(result: dict) -> dict:
    from lab.corpus.build import call_cost, call_error

    spec, turns = result["spec"], result.get("turns", [])
    return {
        "id": spec["id"],
        "split": spec["split"],
        "agent_model": spec["agent_model"],
        "tts": spec["tts"],
        "error": call_error(result) or result.get("breaths_error"),
        "replies": len(turns),
        "passed": sum(t.get("verdict") == "pass" for t in turns),
        "sentences": sum(len(t["sentences"]) for t in turns),
        "cost_usd": call_cost(result),
        "breaths": sum(len(t["breaths"]) for t in (result.get("breaths") or {}).get("turns", [])),
    }
