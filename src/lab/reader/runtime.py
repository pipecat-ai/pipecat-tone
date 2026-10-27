"""The shipped runtime, ``pipecat_tone.reader.model.ToneReaderModel``, as the stages load and time it."""

import statistics
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from lab.reader import sources, storage
from pipecat_tone.reader.model import MODEL_FILES, ToneReaderModel

WARMUP = 30


def load(folder: str | Path, onnx_file: str = MODEL_FILES["fp32"], threads: int = 4) -> ToneReaderModel:
    """The runtime on one of an export's graphs: a file it ships (the optimized graph or its 16-bit weights) as it is,
    or the plain graph, linked in the optimized one's place."""
    folder = Path(folder)
    precision = next((p for p, f in MODEL_FILES.items() if f == onnx_file), None)
    if precision is None:
        view = Path(tempfile.mkdtemp())
        for name in ("config.json", "tokenizer.json"):
            (view / name).symlink_to(folder / name)
        (view / MODEL_FILES["fp32"]).symlink_to(folder / onnx_file)
        folder, precision = view, "fp32"
    return ToneReaderModel(folder, precision=precision, threads=threads)


def fetch_files(key: str, onnx_file: str = MODEL_FILES["fp32"], folder: Path | None = None) -> Path:
    """The config, tokenizer and one graph of a folder on the volume, in `folder` or a new temporary one."""
    return storage.fetch(key, ("config.json", "tokenizer.json", onnx_file), folder)


def fetch(
    run_id: str, prefix: str = "export", onnx_file: str = MODEL_FILES["fp32"], threads: int = 8
) -> ToneReaderModel:
    """A run's export from the volume, loaded as ``load`` loads it."""
    return load(fetch_files(sources.run_key(run_id, prefix), onnx_file), onnx_file, threads)


def timed(cases: dict[str, Callable[[], object]], runs: int) -> dict[str, dict[str, float]]:
    """Milliseconds per call of each case at the median and the 95th percentile, after WARMUP calls each. The cases take
    turns round by round, so a slow moment of the machine hits every case alike."""
    for f in cases.values():
        for _ in range(WARMUP):
            f()
    times = {k: [] for k in cases}
    for _ in range(runs):
        for k, f in cases.items():
            t = time.perf_counter()
            f()
            times[k].append((time.perf_counter() - t) * 1000)
    return {
        k: {"p50_ms": round(statistics.median(v), 3), "p95_ms": round(sorted(v)[int(0.95 * len(v)) - 1], 3)}
        for k, v in times.items()
    }


def latency(model: ToneReaderModel, runs: int = 200) -> dict:
    """Milliseconds per turn, end to end, for a short turn, one with the line before, and a long one."""
    cases = {
        "short": ("thanks, that's all i needed", None),
        "with_previous": (
            "Oh, brilliant. You lost the key. Again.",
            "I think I dropped the mill key in the river.",
        ),
        "long": (
            "You still owe me ten coins, you layabout, and don't think I've forgotten it, because I haven't. "
            * 3,
            "Lovely day for it, isn't it? The sun's out and the market's busy, and I've sold every apple I had. "
            * 3,
        ),
    }
    return timed({k: partial(model.tag, *turn) for k, turn in cases.items()}, runs)
