"""The shipped runtime as the stages load and time it."""

from lab.reader import runtime


def test_the_runtime_reads_an_exports_plain_graph_in_place_of_the_optimized_one(tmp_path, monkeypatch):
    for name in ("config.json", "tokenizer.json", "model.onnx", "model.opt.onnx"):
        (tmp_path / name).write_text(name)
    loaded = []
    monkeypatch.setattr(
        runtime, "ToneReaderModel", lambda folder, precision, threads: loaded.append((folder, precision))
    )
    runtime.load(tmp_path, "model.opt.onnx")
    runtime.load(tmp_path, "model.onnx")
    runtime.load(tmp_path, "model_fp16.onnx")
    assert loaded[0] == (tmp_path, "fp32") and loaded[2] == (tmp_path, "fp16")
    view, precision = loaded[1]
    assert precision == "fp32" and (view / "model.opt.onnx").read_text() == "model.onnx"
    assert (view / "config.json").read_text() == "config.json"


def test_timing_warms_up_every_case_then_takes_turns():
    calls = []
    out = runtime.timed({"a": lambda: calls.append("a"), "b": lambda: calls.append("b")}, runs=5)
    assert calls == ["a"] * runtime.WARMUP + ["b"] * runtime.WARMUP + ["a", "b"] * 5
    assert list(out) == ["a", "b"] and set(out["a"]) == {"p50_ms", "p95_ms"}
