"""The stages as Modal sees them, without running anything: each module imports on its own and registers its jobs and
entrypoints."""

import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab import ROOT
from lab.reader import app

STAGES = {
    "pull": (["pull"], ["main"]),
    "voice_generate": (["select_lines", "write_conversations"], ["generate", "select", "status"]),
    "voice_label": (["label_all", "label_pilot", "label_resume"], ["collect", "label", "pilot", "resume"]),
    "open_ingest": (
        ["download_unit", "ingest_source", "survey_sources", "verify_source"],
        ["download", "ingest", "manifests", "show", "survey", "verify"],
    ),
    "speech": (
        ["build_twins", "check_models", "draw_lines", "sample_twins", "speak_shard", "verify_twins"],
        ["collect", "manifest", "pilot", "prepare", "read", "smoke", "speak", "status", "verify"],
    ),
    "r2_select": (["select"], ["main"]),
    "r2_label": (["label_all", "label_pilot"], ["collect", "label", "pilot"]),
    "support_generate": (["select_lines", "write_calls"], ["generate", "select", "status", "trial"]),
    "support_label": (["label_all", "label_pilot"], ["collect", "label", "pilot"]),
    "r3_chain": (["chain", "no_move_arm"], ["fetch", "no_move", "run"]),
    "sarcasm_teacher": (["teacher"], ["collect", "spawn_one"]),
    "teacher": (["teacher"], ["collect", "smoke", "spawn_one"]),
    "train": (
        ["student", "student_r2"],
        ["collect", "collect_r2", "smoke", "smoke_r2", "spawn_one", "spawn_r2"],
    ),
    "export": (["export"], ["main"]),
    "score": (["score"], ["main"]),
    "compare": (["compare"], ["main"]),
    "precision": (["measure"], ["main"]),
    "fetch": (["check_keys", "download_upstream", "rebuild_open", "rebuild_standins", "run"], ["main"]),
}

REGISTERED = """
import json, modal
import lab.reader.pipeline.{stage} as stage
found = vars(stage).items()
print(json.dumps([sorted(n for n, v in found if isinstance(v, kind)) for kind in (modal.Function, modal.app.LocalEntrypoint)]))
"""


@pytest.fixture(scope="module")
def registered():
    def one(stage):
        run = subprocess.run(
            [sys.executable, "-c", REGISTERED.format(stage=stage)], capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr[-2000:]
        return stage, json.loads(run.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, [*STAGES, "latency"]))


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_registers_its_jobs_and_entrypoints(registered, stage):
    assert registered[stage] == [sorted(x) for x in STAGES[stage]]


def test_latency_runs_here_not_on_modal(registered):
    assert registered["latency"] == [[], []]


def test_the_speech_image_runs_the_reader_groups_stack():
    group = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["reader"]
    assert set(app.READER_STACK) <= set(group)
