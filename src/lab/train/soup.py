"""Averages trained models' weights into one, a model soup. It works for seeds that share their starting weights
(``train_tagger --init-seed``) and differ only in data order: they end near one another, and their average smooths
out a seed's luck at one model's speed. Seeds whose heads started apart average into a broken model.

    uv run --group train python -m lab.train.soup models/<out> models/<seed> models/<seed> ...
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lab.train.fit import quantize

# What a seed's config says of that seed alone: its epochs, scores and calibration.
OWN = (
    "epoch",
    "history",
    "shipped_dev",
    "rules_dev",
    "cpu_latency",
    "sound_bias",
    "breath_calibration",
    "breaths_per_reply",
    "breath_kinds",
)


def soup(out: Path, folders: list[Path]):
    """Average the folders' fp32 ONNX weights into ``out``, quantize it as training does, and take the first folder's
    tokenizer and config, without what belongs to that seed alone."""
    import onnx
    from onnx import numpy_helper

    graphs = [onnx.load(str(f / "model.onnx")) for f in folders]
    weights = [{t.name: numpy_helper.to_array(t) for t in g.graph.initializer} for g in graphs]
    if any(w.keys() != weights[0].keys() for w in weights):
        raise ValueError("the models' graphs differ")
    for t in graphs[0].graph.initializer:
        values = [w[t.name] for w in weights]
        if values[0].dtype.kind == "f":
            t.CopyFrom(numpy_helper.from_array(np.mean(values, axis=0).astype(values[0].dtype), t.name))
        elif any(not np.array_equal(v, values[0]) for v in values):
            raise ValueError(f"{t.name} differs between the models")
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(graphs[0], str(out / "model.onnx"))
    quantize(out)
    (out / "tokenizer.json").write_bytes((folders[0] / "tokenizer.json").read_bytes())
    config = {k: v for k, v in json.loads((folders[0] / "config.json").read_text()).items() if k not in OWN}
    config.update(onnx="model.int8.onnx", soup=[f.name for f in folders])
    (out / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    soup(args.out, args.folders)
