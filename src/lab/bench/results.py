"""Scored runs in tone-bench's layout, so its scoring and report read them, and the log of every run.

``score_call`` builds each turn as tone-bench's ``bench.run.run_pass`` does: keep the two in step.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bench.run as bench_run
from bench.format_check import check_reply
from bench.profiles import tts_adapter
from bench.report import write_report
from bench.score import summarize

from lab import ROOT, TONE_BENCH

RUNS = ROOT / "results" / "runs"
REJUDGED = ROOT / "results" / "rejudged"
LOG = ROOT / "results" / "runs.jsonl"


async def score_call(scenario, records: list[dict], judge) -> list[dict]:
    """Each turn's format checks and, with a judge, its verdicts, as tone-bench saves a turn."""
    adapter = tts_adapter(scenario.tts)
    formats = [check_reply(r["reply"], adapter, scenario.policy, slow=scenario.slow) for r in records]
    judgements = (
        await judge.judge_call(scenario.turns, [SimpleNamespace(**r) for r in records], formats)
        if judge
        else [None] * len(records)
    )
    return [
        {
            "index": r["index"],
            "expect_tag": turn.expect_tag,
            "cues": {"laugh": turn.laugh_cue, "hostile": turn.hostile},
            "checks": {f"s{n}": c.text for n, c in enumerate(turn.checks, 1)},
            "probe": turn.probe,
            "window": bench_run._window(r["index"], len(scenario.turns)),
            "record": r,
            "format": {**asdict(f), "tagged": f.tagged, "expressive": f.expressive},
            "judgement": asdict(j) if j else {},
        }
        for r, f, j, turn in zip(records, formats, judgements, scenario.turns)
    ]


def run_dir(name: str, started: str, base: Path) -> Path:
    """A new run folder, named by the local clock so runs here and on Modal sort together."""
    return base / f"{started}-{name}"


def run_meta(prov: dict, tagger: dict, judge: bool = True) -> dict:
    """What a run records beside its model and scenarios: the judge, Pipecat, the tagger and tone-bench."""
    from bench.judge import JUDGE_EFFORT, JUDGE_MODEL

    return {
        "judge": {"model": JUDGE_MODEL, "effort": JUDGE_EFFORT} if judge else None,
        "pipecat": bench_run._pipecat_version(),
        "tagger": {**tagger, **prov["repo"]},
        "tone_bench": prov["tone_bench"],
    }


def save(folder: Path, run: dict, calls: dict[str, dict[str, dict]]):
    """Write ``{scenario: {pass file: call}}``, then run.json with the scores, then the report."""
    by_scenario, all_turns = {}, []
    for scenario, passes in calls.items():
        (folder / scenario).mkdir(parents=True, exist_ok=True)
        turns = []
        for name, call in passes.items():
            (folder / scenario / name).write_text(json.dumps(call, indent=2))
            turns += call["turns"]
        by_scenario[scenario] = summarize(turns)
        all_turns += turns
    run = {
        **run,
        "started": folder.name[:15],
        "scores": {"overall": summarize(all_turns), "by_scenario": by_scenario},
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2))
    write_report(folder)


def repo_state(path: Path) -> dict:
    """A checkout's commit, and whether it has uncommitted changes."""
    git = ["git", "-C", str(path)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout
        dirty = subprocess.run(
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit.strip(), "dirty": bool(dirty.strip())}


def provenance() -> dict:
    """The commits of this repo and tone-bench, and the local start time, read where the checkouts are."""
    return {
        "repo": repo_state(ROOT),
        "tone_bench": repo_state(TONE_BENCH),
        "started": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }


def finish(folder: Path):
    """Log a run saved here and print tone-bench's summary of it."""
    run = json.loads((folder / "run.json").read_text())
    overall = run["scores"]["overall"]
    entry = {
        "run": folder.name,
        "model": run["model"]["name"],
        "tagger": run.get("tagger"),
        "tone_bench": run.get("tone_bench"),
        "pipecat": run["pipecat"]["commit"],
        "judged": run["judge"] is not None,
        "scenarios": sorted(run["scenarios"]),
        "passes": run["passes"],
        "headline": overall["headline"],
        "recall": overall["recall"],
        "precision": overall["precision"],
    }
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    # tone-bench names runs relative to its own root.
    with mock.patch.object(bench_run, "ROOT", ROOT):
        bench_run._print_summary(run, folder)
    print(f"report: {(folder / 'report.html').relative_to(ROOT)}")
