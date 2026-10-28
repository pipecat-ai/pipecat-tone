"""Calibrate a student on dev, export it to ONNX with the transformer optimizer's graph beside it, and check the shipped
runtime against PyTorch. Files go to models/tonetext/<run_id>/<prefix>/; an existing file is never overwritten, and
each one kept is checked against the new build.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.export --run-id tonetext-r2-student-minilm-l6-seed1 --recipe r2 --sarcasm-cap 0.015
"""

import json

from lab import ROOT
from lab.reader import storage
from lab.reader.app import TORCH, app, job, save

FILES = ("model.onnx", "model.opt.onnx", "tokenizer.json", "config.json")


def export_onnx(model, tok, path: str, max_len: int) -> None:
    import torch

    from lab.reader.data import render
    from lab.reader.model import OUTPUTS

    enc = tok(
        render("You still owe me ten coins, you layabout.", "Lovely day for it, isn't it?"),
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    model.eval()
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=list(OUTPUTS),
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            **{o: {0: "batch"} for o in OUTPUTS},
        },
        opset_version=17,
        dynamo=False,
    )


def optimize_transformer(src: str, dst: str) -> dict:
    """ONNX Runtime's offline transformer optimizer: attention, layer norm and GELU subgraphs fused into single kernels.
    Same weights, the same arithmetic up to float rounding; only basic graph optimizations are baked in (opt_level 1),
    so the graph stays portable."""
    from onnxruntime.transformers.optimizer import optimize_model

    m = optimize_model(src, model_type="bert", num_heads=0, hidden_size=0, opt_level=1, use_gpu=False)
    m.save_model_to_file(dst)
    return {k: v for k, v in m.get_fused_operator_statistics().items() if v}


def parity(clf, model, tok, exs, tau_raw: float, keep_previous: bool) -> dict:
    """The share of lines where the runtime's tags match PyTorch's at the same operating point."""
    from collections import Counter

    from lab.reader.calibration import softmax
    from lab.reader.model import predict
    from lab.reader.schemas import EMOTIONS, INTENSITIES, MOVES

    logits = predict(model, tok, exs, "cpu", keep_previous=keep_previous, logits=True, bf16=False)
    same = Counter()
    for i, e in enumerate(exs):
        t = clf.tag(e.text, e.previous if keep_previous else None)
        emo = EMOTIONS[int(logits["emotion"][i].argmax())]
        intensity = "low" if emo == "neutral" else INTENSITIES[int(logits["emotion_intensity"][i].argmax())]
        same["emotion"] += t["emotion"]["label"] == emo
        same["intensity"] += t["emotion"]["intensity"] == intensity
        same["move"] += t["move"]["label"] == MOVES[int(logits["move"][i].argmax())]
        same["sarcasm"] += t["sarcasm"]["sarcastic"] == (float(softmax(logits["sarcasm"][i])[1]) >= tau_raw)
        same["backchannel"] += t["backchannel"]["is_backchannel"] == (
            float(softmax(logits["backchannel"][i])[1]) >= 0.5
        )
    return {"lines": len(exs), **{k: round(v / max(1, len(exs)), 4) for k, v in same.items()}}


@job(TORCH, cpu=8.0, memory=32768, timeout=3 * 3600)
def export(
    run_id: str,
    prefix: str = "export",
    parity_lines: int = 2000,
    recipe: str = "r1",
    sarcasm_cap: float = 0.02,
) -> dict:
    """`recipe="r2"`: the calibration is fitted on Tone's dev lines and the voice-agent dev set together; r2 takes a
    `sarcasm_cap` of 0.015, a margin under the 2% the scorecard asks of test. `recipe="r3"` is calibrated on the same
    lines, without the support-call set's, so r3 and rc1 are calibrated alike."""
    import hashlib
    import io
    import os
    import tempfile
    import time

    import torch
    from transformers import AutoTokenizer

    from lab.reader import data, runtime, sources
    from lab.reader.calibration import ece, ece_top, fit_platt, fit_temperature, platt, softmax
    from lab.reader.model import Model, predict
    from lab.reader.scoring import threshold_on

    storage.reload()
    t0 = time.time()
    torch.set_num_threads(8)
    cfg = json.loads(storage.get(sources.run_key(run_id, "config.json")))
    model = Model(cfg["repo"], cfg["revision"], attn="eager")
    state = torch.load(io.BytesIO(storage.get(sources.run_key(run_id, "model.pt"))), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    if recipe in ("r2", "r3"):
        lines = data.build_r2(storage.jsonl, pools=False)
        dev, test = lines.split("dev"), lines.split("test")
        g2 = data.eval_groups_r2(dev, "dev")
        g = {
            "emotion": g2["emotion"] + g2["voice_emotion"],
            "move": g2["move"] + g2["voice_move"],
            "sarcasm": g2["sarcasm"] + g2["voice_sarcasm"],
        }
    else:
        lines = data.build(storage.jsonl, pools=False)
        dev, test = lines.split("dev"), lines.split("test")
        g = data.eval_groups(dev, "dev")
    Ld = predict(model, tok, dev, "cpu", keep_previous=True, logits=True, bf16=False)
    ye, ym = [dev[i].emo for i in g["emotion"]], [dev[i].move for i in g["move"]]
    t_emo = fit_temperature(Ld["emotion"][g["emotion"]], ye)
    t_move = fit_temperature(Ld["move"][g["move"]], ym)
    raw = softmax(Ld["sarcasm"])[:, 1]
    nat = [i for i in g["sarcasm"] if sources.NATURAL_SARCASM[0] in dev[i].sets]
    tau = threshold_on([float(raw[i]) for i in nat], [dev[i].sarc for i in nat], cap=sarcasm_cap)
    ys = [dev[i].sarc for i in g["sarcasm"]]
    curve = fit_platt([float(raw[i]) for i in g["sarcasm"]], ys)
    dev_cal = {
        "emotion": {
            "temperature": t_emo,
            "ece_raw": ece_top(softmax(Ld["emotion"][g["emotion"]]), ye),
            "ece": ece_top(softmax(Ld["emotion"][g["emotion"]], t_emo), ye),
        },
        "move": {
            "temperature": t_move,
            "ece_raw": ece_top(softmax(Ld["move"][g["move"]]), ym),
            "ece": ece_top(softmax(Ld["move"][g["move"]], t_move), ym),
        },
        "sarcasm": {
            "curve": curve,
            "tau_raw": tau,
            "ece_raw": ece([float(raw[i]) for i in g["sarcasm"]], ys)["ece"],
            "ece": ece([platt(float(raw[i]), curve) for i in g["sarcasm"]], ys)["ece"],
        },
    }
    d = tempfile.mkdtemp()
    tok.backend_tokenizer.save(os.path.join(d, "tokenizer.json"))
    runtime_cfg = {
        "run_id": run_id,
        "model": "tonetext",
        "contract": cfg["contract"],
        "encoder": cfg["encoder"],
        "repo": cfg["repo"],
        "revision": cfg["revision"],
        "max_len": cfg["max_len"],
        "render": cfg["render"],
        "outputs": cfg["outputs"],
        "emotions": cfg["emotions"],
        "intensities": cfg["intensities"],
        "moves": cfg["moves"],
        "calibration": {"emotion_temperature": t_emo, "move_temperature": t_move, "sarcasm": curve},
        "calibration_note": (
            "Fitted on the dev lines as given"
            + (", Tone's and the voice-agent set's" if recipe in ("r2", "r3") else "")
            + ": one temperature per multi-class output by log loss, so the top probability is the "
            "chance the label is right; sarcasm, a Platt curve on the raw probability"
        ),
        "sarcasm_tau_raw": tau,
        "sarcasm_tau_rule": f"the lowest threshold at which at most {sarcasm_cap:.1%} of sincere natural dev lines are "
        "called sarcastic",
        "backchannel_rule": "a backchannel when its probability reaches 0.5",
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(runtime_cfg, f, indent=1)
    fp32 = os.path.join(d, "model.onnx")
    export_onnx(model, tok, fp32, cfg["max_len"])
    fused = optimize_transformer(fp32, os.path.join(d, "model.opt.onnx"))
    sample = [e for e in test if not e.twin][:parity_lines]
    res = {
        "run_id": run_id,
        "dev_calibration": dev_cal,
        "fused_operators": fused,
        "sizes_mb": {f: round(os.path.getsize(os.path.join(d, f)) / 1e6, 2) for f in FILES},
        "variants": {},
    }
    for f in ("model.onnx", "model.opt.onnx"):
        clf = runtime.load(d, f, threads=4)
        res["variants"][f] = {
            "parity_with_previous": parity(clf, model, tok, sample, tau, True),
            "parity_without_previous": parity(clf, model, tok, sample, tau, False),
            "latency_4_threads_modal_cpu": runtime.latency(clf),
            "latency_1_thread_modal_cpu": runtime.latency(runtime.load(d, f, threads=1), runs=100),
        }
    res["uploaded"], res["kept"], res["existing_sha256_matches"] = [], [], {}
    for f in FILES:
        key = sources.run_key(run_id, prefix, f)
        with open(os.path.join(d, f), "rb") as fh:
            blob = fh.read()
        if storage.exists(key):
            res["kept"].append(f)
            res["existing_sha256_matches"][f] = (
                hashlib.sha256(storage.get(key)).hexdigest() == hashlib.sha256(blob).hexdigest()
            )
        else:
            storage.put(key, blob)
            res["uploaded"].append(f)
    res["prefix"] = f"{sources.run_key(run_id, prefix)}/"
    res["seconds"] = round(time.time() - t0)
    record = sources.run_key(run_id, prefix, "export_result.json")
    if not storage.exists(record):
        storage.put_json(record, res)
    return res


@app.local_entrypoint()
def main(run_id: str, prefix: str = "export", recipe: str = "r1", sarcasm_cap: float = 0.02):
    res = export.remote(run_id, prefix, recipe=recipe, sarcasm_cap=sarcasm_cap)
    path = save("export", f"results-{run_id}.json", res)
    c = res["dev_calibration"]
    print(
        f"{run_id}: emotion T {c['emotion']['temperature']} (ECE {c['emotion']['ece_raw']} -> {c['emotion']['ece']}) "
        f"| move T {c['move']['temperature']} (ECE {c['move']['ece_raw']} -> {c['move']['ece']}) | sarcasm tau "
        f"{c['sarcasm']['tau_raw']} (ECE {c['sarcasm']['ece_raw']} -> {c['sarcasm']['ece']})"
    )
    for f, v in res["variants"].items():
        print(
            f, "parity with/without the line before", v["parity_with_previous"], v["parity_without_previous"]
        )
        print(f, "latency 4 threads (Modal CPU)", v["latency_4_threads_modal_cpu"])
    print("sizes", res["sizes_mb"], "uploaded", res["uploaded"], "kept", res["existing_sha256_matches"])
    print("wrote", path.relative_to(ROOT))
