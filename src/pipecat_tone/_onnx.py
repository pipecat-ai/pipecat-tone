"""ONNX Runtime and tokenizers on the CPU, as the tone models run. They come with the ``model`` extra."""

from pathlib import Path


def session(path: Path, threads: int):
    """An inference session for a model file, with ``threads`` for one run."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as e:
        raise need_extra(e) from e
    return Tokenizer.from_file(str(path))


def need_extra(e: ModuleNotFoundError) -> ImportError:
    return ImportError(f'{e.name} is missing; the tone models need the model extra, "pipecat-tone[model]"')
