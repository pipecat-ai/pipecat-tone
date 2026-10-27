"""tone-bench's leaderboard with the taggers' runs ranked beside its models, written here, or with ``--tone-bench`` to
tone-bench's own docs/leaderboard.md. By default, the newest judged run of each LLM and tagger here.

    uv run python -m lab.bench.leaderboard [--tone-bench] [results/runs/<run> ...]
"""

import json
import sys
from pathlib import Path

from lab import ROOT
from lab.bench.results import RUNS

OUT = ROOT / "results"
NOTE = (
    "A model with `+` is an LLM with a tagger after it, run from [pipecat-tone](https://github.com/pipecat-ai/pipecat-"
    "tone): `+model` the tone tagger, `+rules` its rules alone, `+teacher-offline` its teacher labelling a plain run's "
    "replies whole (the tagger's ceiling), and `+noop` no tagger. The rest are tone-bench's own runs."
)


def tagger_runs() -> list[Path]:
    """The newest judged run of each LLM and tagger pair here."""
    newest: dict[str, Path] = {}
    for path in sorted(RUNS.glob("*/run.json"), reverse=True):
        run = json.loads(path.read_text())
        if run.get("judge"):
            newest.setdefault(run["model"]["name"], path.parent)
    return list(newest.values())


def main():
    import bench.leaderboard as board

    args = sys.argv[1:]
    in_tone_bench = "--tone-bench" in args
    if not in_tone_bench:
        board.ROOT, board.RESULTS, board.DOCS_LEADERBOARD = ROOT, OUT, OUT / "leaderboard.md"
    runs = [Path(a).resolve() for a in args if a != "--tone-bench"] or tagger_runs()
    ours = [{name: d for name in json.loads((d / "run.json").read_text())["scenarios"]} for d in runs]
    out, models = board.write_leaderboard(board.latest_results() + ours)
    board._print(models, out)
    md = board.write_markdown(models)
    text = md.read_text()
    if not in_tone_bench:
        text = text.replace("[Methodology](methodology.md)", "tone-bench's `docs/methodology.md`")
    title, intro, rest = text.split("\n\n", 2)
    md.write_text("\n\n".join([title, intro, NOTE, rest]))
    print(md.relative_to(board.ROOT))


if __name__ == "__main__":
    main()
