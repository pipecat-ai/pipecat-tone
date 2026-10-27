"""Baseline E: the teacher labels a plain run's replies offline, and tone-bench's judge scores them.

The source is a no-op tagger run on the ``tone-tagger-results`` volume. Each call is labeled whole, as
corpus calls are, then checked and judged again; the words are the source's. The tagger's ceiling.
"""

import asyncio
import json
from pathlib import Path

from lab.jobs.common import REMOTE_RUNS, RESULTS_DIR, app, pack, results, unpack


@app.function(volumes={RESULTS_DIR: results}, timeout=3 * 60 * 60, cpu=2.0, memory=2048)
def teach(source: str, prov: dict) -> tuple[str, dict[str, bytes]]:
    from lab import quiet_logs

    quiet_logs()
    results.reload()
    folder = asyncio.run(_teach(REMOTE_RUNS / source, prov))
    results.commit()
    return folder.name, pack(folder)


async def _teach(source: Path, prov: dict) -> Path:
    import anthropic
    from bench.judge import Judge
    from bench.scenario import load_scenario

    from lab.bench import scenario_path
    from lab.bench.agent import expressive_instruction
    from lab.bench.results import run_dir, run_meta, save, score_call
    from lab.corpus.teacher import Exchange, Teacher
    from pipecat_tone.tagger import LabelSpace
    from pipecat_tone.tagger.markup import render_reply
    from pipecat_tone.tagger.processor import split_sentences

    run = json.loads((source / "run.json").read_text())
    name = f"{run['model']['name'].split('+')[0]}+teacher-offline"
    teacher = Teacher(anthropic.AsyncAnthropic())
    cost = 0.0
    calls: dict[str, dict[str, dict]] = {}
    for scenario_name in run["scenarios"]:
        scenario = load_scenario(scenario_path(scenario_name))
        space = LabelSpace.for_tts(scenario.tts, scenario.policy)
        judging = expressive_instruction(scenario)
        for pass_file in sorted((source / scenario_name).glob("pass-*.json")):
            call = json.loads(pass_file.read_text())
            records = [t["record"] for t in call["turns"]]
            exchanges = [
                Exchange(st.user, st.heard, await split_sentences(r["reply"]))
                for st, r in zip(scenario.turns, records)
            ]
            teaching = await teacher.label(exchanges, space, scenario.policy)
            cost += teaching.cost_usd
            print(
                f"{scenario_name} {pass_file.stem}: teacher ${teaching.cost_usd:.3f} {teaching.error or ''}",
                flush=True,
            )
            for record, exchange, labels in zip(records, exchanges, teaching.labels):
                record["plain_reply"] = record["reply"]
                record["reply"] = render_reply(space, exchange.sentences, labels)
            turns = await score_call(scenario, records, Judge(judging))
            calls.setdefault(scenario_name, {})[pass_file.name] = {
                **call,
                "judge_instruction": judging,
                "turns": turns,
            }
    meta = {
        **run,
        "model": {**run["model"], "name": name},
        **run_meta(prov, {"spec": "offline", "source_run": source.name}),
        "teacher_cost_usd": round(cost, 3),
    }
    folder = run_dir(name, prov["started"], source.parent)
    save(folder, meta, calls)
    return folder


@app.local_entrypoint()
def teach_offline(source: str):
    from lab.bench.results import RUNS, finish, provenance

    name, files = teach.remote(source, provenance())
    finish(unpack(files, RUNS / name))
