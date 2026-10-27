"""The 16-bit file, built from an export and measured against full precision on every test line: identical tags,
paired scorecard items, size, memory and speed, and the rule it ships by. It goes to
models/tonetext/<run_id>/precision/model_fp16.onnx, never overwritten.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.precision --run-id tonetext-r2-student-minilm-l6-seed1
"""

import json
import pathlib
import sys

from lab import ROOT
from lab.reader import storage
from lab.reader.app import TORCH, app, job, save

MEMORY = """import json, resource, sys
from pipecat_tone.reader.model import ToneReaderModel
clf = ToneReaderModel(sys.argv[1], precision=sys.argv[2], threads=4)
for _ in range(50):
    clf.tag("oh great, thanks a lot", "Sorry, we're out of rooms tonight.")
print(json.dumps({"max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}))
"""


def fp16_weights(folder: pathlib.Path) -> dict:
    """model_fp16.onnx beside the optimized graph: weights of 1,024 values or more stored in 16 bits, each with a Cast
    back to 32 bits in front of its users. The fp32 files are never touched."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from pipecat_tone.reader.model import MODEL_FILES

    out = folder / MODEL_FILES["fp16"]
    m = onnx.load(str(folder / MODEL_FILES["fp32"]))
    casts, n = [], 0
    for init in list(m.graph.initializer):
        if init.data_type == TensorProto.FLOAT and len(init.dims) >= 1 and int(np.prod(init.dims)) >= 1024:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
            m.graph.initializer.remove(init)
            m.graph.initializer.append(h)
            casts.append(
                helper.make_node(
                    "Cast", [h.name], [init.name], name=init.name + "_Cast", to=TensorProto.FLOAT
                )
            )
            n += 1
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, str(out))
    return {
        "variant": "fp16w",
        "file": out.name,
        "how": "weights of 1,024 values or more stored as fp16 and cast back to fp32: fp32 compute",
        "weights": n,
        "bytes": out.stat().st_size,
    }


@job(TORCH, cpu=8.0, memory=16384, timeout=3 * 3600)
def measure(run_id: str, prefix: str = "export") -> dict:
    import hashlib
    import subprocess
    import time

    from lab.reader import runtime, scoring, sources
    from pipecat_tone.reader.model import MODEL_FILES

    storage.reload()
    t0 = time.time()
    d = runtime.fetch_files(sources.run_key(run_id, prefix))
    built = fp16_weights(d)
    fp32, fp16 = runtime.load(d, threads=8), runtime.load(d, MODEL_FILES["fp16"], threads=8)

    rows = scoring.test_rows(storage.jsonl)
    act_rows, ex_rows, sar_rows, voice, real = (
        rows["act"],
        rows["exchange"],
        rows["sarcasm"],
        rows["voice"],
        rows["voice_real"],
    )
    every = act_rows + ex_rows + sar_rows + voice["emotion"] + voice["move"] + voice["sarcasm"]
    every += real["emotion"] + real["move"] + real["sarcasm"]

    out = {"run_id": run_id, "built": built, "columns": {}}
    paired = scoring.Paired(fp32, fp16)
    for col, keep in (("with_previous", True), ("without_previous", False)):
        same, n = {"emotion": 0, "move": 0, "sarcasm": 0, "backchannel": 0}, 0
        for turn in {scoring.turn(r, keep) for r in every}:
            a, b = paired.tag(*turn)
            n += 1
            same["emotion"] += (a["emotion"]["label"], a["emotion"]["intensity"]) == (
                b["emotion"]["label"],
                b["emotion"]["intensity"],
            )
            same["move"] += a["move"]["label"] == b["move"]["label"]
            same["sarcasm"] += a["sarcasm"]["sarcastic"] == b["sarcasm"]["sarcastic"]
            same["backchannel"] += a["backchannel"]["is_backchannel"] == b["backchannel"]["is_backchannel"]
        c = paired.column(keep)
        c.emotion("emotion_main", [r for r in act_rows if scoring.main_test(r["line"])])
        c.emotion(
            "emotion_player_speech", [r for r in act_rows if scoring.in_slice(r["line"], "player_speech")]
        )
        c.emotion("emotion_unseen", [r for r in act_rows if scoring.in_slice(r["line"], "unseen_genre")])
        c.emotion("emotion_voice", voice["emotion"])
        c.move("move", [r for r in ex_rows if scoring.move_test(r["line"])])
        c.move("move_voice", voice["move"])
        c.sarcasm(
            "sarcasm_held_out", [r for r in sar_rows if r["set"] in scoring.HARD_SARCASM_SETS], True, False
        )
        c.sarcasm(
            "sarcasm_natural", [r for r in sar_rows if r["set"] == scoring.NATURAL_SARCASM_SET], False, True
        )
        c.sarcasm("sarcasm_voice", voice["sarcasm"], True, True)
        c.emotion("emotion_voice_real", real["emotion"])
        c.move("move_voice_real", real["move"])
        c.sarcasm("sarcasm_voice_real", real["sarcasm"], True, True)
        out["columns"][col] = {
            "lines": n,
            "identical": {k: round(v / max(1, n), 5) for k, v in same.items()},
            "identical_counts": same,
            "items": c.items,
            "worse_beyond_noise": c.worse(),
            "false_sarcasm_rises": c.rises(),
        }
    rule1 = all(v >= 0.99 for c in out["columns"].values() for v in c["identical"].values())
    rule2 = not any(c["worse_beyond_noise"] for c in out["columns"].values())
    rule3 = not any(c["false_sarcasm_rises"] for c in out["columns"].values())
    out["rule"] = {
        "tags_identical_99": rule1,
        "no_item_worse": rule2,
        "false_sarcasm_not_up": rule3,
        "ships": rule1 and rule2 and rule3,
    }
    files = (MODEL_FILES["fp32"], MODEL_FILES["fp16"])
    out["sizes_mb"] = {f: round((d / f).stat().st_size / 1e6, 2) for f in files}
    out["latency_4_threads_modal_cpu"] = {
        p: runtime.latency(runtime.load(d, MODEL_FILES[p], threads=4)) for p in ("fp32", "fp16")
    }
    script = d / "memory.py"
    script.write_text(MEMORY)
    out["memory"] = {}
    for prec in ("fp32", "fp16"):
        run = subprocess.run(
            [sys.executable, str(script), str(d), prec], capture_output=True, text=True, cwd="/tmp"
        )
        out["memory"][prec] = (
            json.loads(run.stdout.strip().splitlines()[-1])
            if run.returncode == 0
            else {"error": run.stderr[-500:]}
        )
    blob = (d / MODEL_FILES["fp16"]).read_bytes()
    out["key"] = sources.run_key(run_id, "precision", MODEL_FILES["fp16"])
    out["sha256"] = hashlib.sha256(blob).hexdigest()
    out["bytes"] = len(blob)
    if not storage.exists(out["key"]):
        storage.put(out["key"], blob)
        out["uploaded"] = True
    else:
        # Never overwritten: a second run must have built the same bytes.
        out["uploaded"] = False
        out["existing_sha256_matches"] = hashlib.sha256(storage.get(out["key"])).hexdigest() == out["sha256"]
    out["seconds"] = round(time.time() - t0)
    return out


@app.local_entrypoint()
def main(run_id: str, prefix: str = "export"):
    res = measure.remote(run_id, prefix)
    path = save("precision", f"results-fp16-{run_id}.json", res)
    for col, c in res["columns"].items():
        print(
            col,
            c["lines"],
            "lines | identical",
            c["identical"],
            "| worse beyond noise",
            c["worse_beyond_noise"],
            "| false sarcasm rises",
            c["false_sarcasm_rises"],
        )
        for k, v in c["items"].items():
            print(f"   {k:44s} fp32 {v['ref']} fp16 {v['var']} diff {v['diff']} ci {v['ci']}")
    print("rule", res["rule"], "| sizes", res["sizes_mb"], "| memory", res["memory"])
    print("latency (Modal CPU, 4 threads)", res["latency_4_threads_modal_cpu"])
    print("volume", res["key"], res["sha256"], "uploaded", res["uploaded"])
    print("wrote", path.relative_to(ROOT))
