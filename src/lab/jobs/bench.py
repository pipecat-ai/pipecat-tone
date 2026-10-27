"""Runs tone-bench with a tagger, judge included, and brings the run back to results/runs.

A run stops with the local client: add ``--detach`` after ``modal run`` to keep it going, then fetch
it from the ``tone-tagger-results`` volume and log it with ``python -m lab.bench.run --finish``.
"""

import asyncio
from pathlib import Path

from lab.jobs.common import MODELS_DIR, REMOTE_RUNS, RESULTS_DIR, app, models, pack, results, unpack

# A full suite with the judge takes tens of minutes per pass.
TIMEOUT_S = 6 * 60 * 60


@app.function(volumes={RESULTS_DIR: results, MODELS_DIR: models}, timeout=TIMEOUT_S, cpu=2.0, memory=4096)
def run_remote(
    model: str, tagger: str, scenarios: list[str], passes: int, judge: bool, prov: dict, tagger_model: str
) -> tuple[str, dict[str, bytes]]:
    from lab import quiet_logs
    from lab.bench.run import run_bench

    quiet_logs()
    model_dir = Path(MODELS_DIR) / tagger_model if tagger_model else None
    folder = asyncio.run(
        run_bench(model, tagger, scenarios or None, passes, judge, prov, REMOTE_RUNS, model_dir)
    )
    results.commit()
    return folder.name, pack(folder)


@app.local_entrypoint()
def run_bench_job(
    model: str,
    tagger: str = "noop",
    scenario: str = "",
    passes: int = 3,
    no_judge: bool = False,
    tagger_model: str = "",
):
    """``--scenario`` takes names separated by commas (default: all); ``--tagger-model`` names a models volume folder."""
    from lab.bench.results import RUNS, finish, provenance

    scenarios = [s.strip() for s in scenario.split(",") if s.strip()]
    name, files = run_remote.remote(
        model, tagger, scenarios, passes, not no_judge, provenance(), tagger_model
    )
    finish(unpack(files, RUNS / name))
