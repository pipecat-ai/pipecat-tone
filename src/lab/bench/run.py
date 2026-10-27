"""Runs tone-bench's scenarios with a tagger. Locally, unjudged only; judged runs go through ``lab.jobs.bench``.

uv run python -m lab.bench.run --model phonellm-mini --tagger rules --scenario bank_fraud_alert --passes 1
uv run python -m lab.bench.run --finish results/runs/<run>    # log a run fetched from Modal
"""

import argparse
import asyncio
import importlib
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from lab import ROOT, quiet_logs
from lab.bench import SCENARIOS, model_config, scenario_path
from lab.bench.agent import expressive_instruction, run_call
from lab.bench.results import RUNS, finish, provenance, run_dir, run_meta, save, score_call
from pipecat_tone.tagger import ModelTagger, NoOpTagger, RulesTagger, Tagger, ToneTaggerModel

TAGGERS: dict[str, type[Tagger]] = {"noop": NoOpTagger, "rules": RulesTagger, "model": ModelTagger}


def tagger_class(spec: str) -> type[Tagger]:
    """A name in ``TAGGERS``, or ``module:Class``."""
    if spec in TAGGERS:
        return TAGGERS[spec]
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def load_tagger(spec: str, model_dir: Path | None) -> Tagger:
    """A tagger; the model tagger needs its model folder."""
    cls = tagger_class(spec)
    if cls is not ModelTagger:
        return cls()
    if model_dir is None:
        raise ValueError("the model tagger needs a model folder")
    return ModelTagger(ToneTaggerModel(model_dir))


async def run_bench(
    model: str,
    tagger: str,
    scenarios: list[str] | None,
    passes: int,
    judge: bool,
    prov: dict,
    base: Path = RUNS,
    model_dir: Path | None = None,
) -> Path:
    """Hold, check and (optionally) judge each scenario's calls, saving after every scenario."""
    from bench.judge import Judge
    from bench.scenario import load_scenario

    config = model_config(model)
    config.name = f"{config.name}+{tagger_class(tagger).name}"
    # The tagger's model in the folder's name too, so runs of two models started together don't share one.
    folder = run_dir(config.name + (f"-{model_dir.name}" if model_dir else ""), prov["started"], base)
    paths = [scenario_path(s) for s in scenarios] if scenarios else sorted(SCENARIOS.glob("*.yaml"))
    tagged_by = {"model": model_dir.resolve().name} if model_dir else {}
    meta = {
        "model": config.describe(),
        "passes": passes,
        **run_meta(prov, {"spec": tagger, **tagged_by}, judge),
        "scenarios": {},
    }
    # One tagger for every call: it forgets the last call when the next starts.
    tagging = load_tagger(tagger, model_dir)
    calls: dict[str, dict[str, dict]] = {}
    for path in paths:
        scenario = load_scenario(path)
        meta["scenarios"][scenario.name] = {
            "digest": scenario.digest,
            "tts": scenario.tts,
            "turns": len(scenario.turns),
        }
        for n in range(1, passes + 1):
            print(f"{scenario.name} pass {n}/{passes} ({len(scenario.turns)} turns)...", flush=True)
            instruction, records = await run_call(scenario, config, tagging)
            judging = expressive_instruction(scenario)
            turns = await score_call(
                scenario, [asdict(r) for r in records], Judge(judging) if judge else None
            )
            calls.setdefault(scenario.name, {})[f"pass-{n}.json"] = {
                "scenario": scenario.name,
                "instruction": instruction,
                "judge_instruction": judging,
                "turns": turns,
            }
        save(folder, meta, calls)
    return folder


def main(args):
    if args.finish:
        return finish(Path(args.finish).resolve())
    if not args.model or not args.tagger:
        sys.exit("--model and --tagger are required")
    load_dotenv(ROOT / ".env", override=True)
    model_dir = (
        Path(args.tagger_model or ROOT / "models" / "current")
        if tagger_class(args.tagger) is ModelTagger
        else None
    )
    prov = provenance()
    finish(
        asyncio.run(
            run_bench(args.model, args.tagger, args.scenario, args.passes, False, prov, model_dir=model_dir)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="an LLM from tone-bench's models.yaml")
    parser.add_argument("--tagger", help=f"one of {list(TAGGERS)}, or module:Class")
    parser.add_argument(
        "--scenario", action="append", help="a scenario name; repeat for several (default: all)"
    )
    parser.add_argument("--tagger-model", help="the model tagger's folder (default: models/current)")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--finish", metavar="RUN_DIR", help="log and summarize a run fetched from Modal")
    quiet_logs()
    main(parser.parse_args())
