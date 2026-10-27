"""Time an export on this machine: the reader with and without the line before, and its 16-bit file when there is one,
at 1 and 4 threads.

    MODAL_PROFILE=daily MODAL_ENVIRONMENT=tone-tagger uv run python -m lab.reader.pipeline.latency --run-id tonetext-r2-student-minilm-l6-seed1
"""

import argparse
import platform
import subprocess
from functools import partial
from pathlib import Path

from lab import ROOT
from lab.reader import storage
from lab.reader.app import save

TEXT = "You still owe me ten coins, you layabout, and don't think I've forgotten it."
PREVIOUS = "Lovely day for it, isn't it?"
# Files from the volume, fetched once and kept at their keys.
LOCAL = ROOT / "data" / "reader"


def cpu_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return platform.processor()


def fetch_export(run_id: str, prefix: str, source: str) -> Path:
    """A run's export, or the folder at `source`, with a 16-bit file beside it when there is one: the one precision
    built for the run, or the folder's own."""
    from lab.reader import runtime, sources
    from pipecat_tone.reader.model import MODEL_FILES

    key = source.rstrip("/") or sources.run_key(run_id, prefix)
    folder = runtime.fetch_files(key, folder=LOCAL / key)
    fp16_at = key if source else sources.run_key(run_id, "precision")
    if storage.exists(f"{fp16_at}/{MODEL_FILES['fp16']}"):
        storage.fetch(fp16_at, (MODEL_FILES["fp16"],), folder)
    return folder


def main() -> None:
    from lab.reader import runtime
    from pipecat_tone.reader.model import MODEL_FILES

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument(
        "--source", default="", help="a folder on the volume, timed with its 16-bit file when it has one"
    )
    ap.add_argument("--prefix", default="export")
    ap.add_argument("--runs", type=int, default=300)
    a = ap.parse_args()
    if not (a.run_id or a.source):
        raise SystemExit("give --run-id, or --source with a folder on the volume")
    tt_dir = fetch_export(a.run_id, a.prefix, a.source)
    out = {"machine": cpu_name(), "runs": a.runs, "text": TEXT, "previous": PREVIOUS, "threads": {}}
    for threads in (1, 4):
        tt = runtime.load(tt_dir, threads=threads)
        cases = {
            "tonetext, with the line before": partial(tt.tag, TEXT, PREVIOUS),
            "tonetext, without it": partial(tt.tag, TEXT, None),
        }
        if (tt_dir / MODEL_FILES["fp16"]).exists():
            half = runtime.load(tt_dir, MODEL_FILES["fp16"], threads)
            cases["tonetext 16-bit, with the line before"] = partial(half.tag, TEXT, PREVIOUS)
        out["threads"][str(threads)] = runtime.timed(cases, a.runs)
    name = a.source.rstrip("/").split("/")[-1] if a.source else a.run_id
    path = save("latency", f"this-mac-{name}.json", out)
    for th, cases in out["threads"].items():
        for k, v in cases.items():
            print(f"{th} thread(s) | {k:32s} p50 {v['p50_ms']:.2f} ms  p95 {v['p95_ms']:.2f} ms")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
