"""Judges finished tone-bench runs again with another vendor's model, against same-family grading.

tone-bench's judge and the teacher are both Claude, so a tagger distilled from the teacher might be
graded kindly. This keeps tone-bench's prompt, checks, replies and format results, and changes only
the model, to another vendor's than the corpus reviewer's. Takes run folders separated by commas.
"""

import asyncio
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from bench.judge import STANDARD_CHECKS, Judge, TurnJudgement, Verdict, _prompt, _schema
from bench.score import CHECK_PILLARS
from openai import AsyncOpenAI

from lab.jobs.common import app, pack, unpack
from lab.llm import openai_json

REJUDGE_MODEL = "gpt-5.6-luna"
REJUDGE_EFFORT = "medium"


class OpenAIJudge(Judge):
    """tone-bench's judge with an OpenAI model. ``_judge_turn``'s checks are built as tone-bench's are."""

    def __init__(self, instruction: str):
        super().__init__(instruction)
        self._openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def _judge_turn(self, i, turns, records, formats) -> TurnJudgement:
        checks = dict(STANDARD_CHECKS)
        pillars = {c: CHECK_PILLARS[c] for c in STANDARD_CHECKS}
        for n, check in enumerate(turns[i].checks, 1):
            checks[f"s{n}"] = check.text
            pillars[f"s{n}"] = check.pillar
        prompt = _prompt(i, records, formats[i], checks)
        async with self._semaphore:
            reply = await openai_json(
                self._openai, REJUDGE_MODEL, REJUDGE_EFFORT, self._system, prompt, _schema(checks)["schema"]
            )
        if reply.error:
            return TurnJudgement(verdicts=[], judge_model=None, error=reply.error)
        return TurnJudgement(
            verdicts=[
                Verdict(c, reply.data[c]["verdict"], reply.data[c]["note"], pillars[c]) for c in checks
            ],
            judge_model=REJUDGE_MODEL,
            input_tokens=reply.usage["input"],
            output_tokens=reply.usage["output"],
        )


@app.function(timeout=3 * 60 * 60, cpu=2.0, memory=4096)
def rejudge(name: str, run: dict, calls: dict[str, dict[str, dict]]) -> dict[str, bytes]:
    """Judge a run's ``{scenario: {pass file: call}}`` again; returns the re-judged run's files."""
    from lab import quiet_logs
    from lab.bench.results import save

    quiet_logs()
    rejudged = asyncio.run(_rejudge(calls))
    folder = Path(tempfile.mkdtemp()) / name
    save(
        folder,
        {**run, "judge": {"model": REJUDGE_MODEL, "effort": REJUDGE_EFFORT, "rejudged": True}},
        rejudged,
    )
    return pack(folder)


async def _rejudge(calls: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    from bench.scenario import load_scenario

    from lab.bench import scenario_path

    out: dict[str, dict[str, dict]] = {}
    for scenario_name, passes in calls.items():
        scenario = load_scenario(scenario_path(scenario_name))
        for pass_name, call in sorted(passes.items()):
            judge = OpenAIJudge(
                call.get("judge_instruction") or call.get("instruction") or scenario.system_prompt
            )
            records = [SimpleNamespace(**t["record"]) for t in call["turns"]]
            formats = [SimpleNamespace(shown=t["format"]["shown"]) for t in call["turns"]]
            judgements = await judge.judge_call(scenario.turns, records, formats)
            turns = [{**t, "judgement": asdict(j)} for t, j in zip(call["turns"], judgements)]
            out.setdefault(scenario_name, {})[pass_name] = {**call, "turns": turns}
    return out


@app.local_entrypoint()
def rejudge_runs(runs: str):
    from lab.bench.results import REJUDGED

    jobs = []
    for folder in [Path(r.strip()).resolve() for r in runs.split(",") if r.strip()]:
        run = json.loads((folder / "run.json").read_text())
        calls = {
            s: {p.name: json.loads(p.read_text()) for p in sorted((folder / s).glob("pass-*.json"))}
            for s in run["scenarios"]
        }
        jobs.append((folder.name, run, calls))
    print(f"re-judging {len(jobs)} runs with {REJUDGE_MODEL}", flush=True)
    for (name, _, _), files in zip(jobs, rejudge.starmap(jobs)):
        overall = json.loads((unpack(files, REJUDGED / name) / "run.json").read_text())["scores"]["overall"]
        print(
            f"{name}: headline {overall['headline']}, expression {overall['expression']}, "
            f"perception {overall['perception']}, conversation {overall['conversation']}",
            flush=True,
        )
