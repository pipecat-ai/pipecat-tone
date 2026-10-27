"""The speech round trip: noisy twins of r2's pool lines, spoken by an open text-to-speech model and transcribed again
by an open speech-to-text one (``lab.reader.speech``).

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::prepare
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::smoke
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::pilot --pilot-n 400
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.speech::speak --shards 10 --gpu L4
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::status
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::collect
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::verify
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::read
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.speech::manifest
"""

import json
import pathlib

import modal

from lab.reader import sources, speech, storage
from lab.reader.app import DATA, SPEECH, app, job, save, save_yaml

VERSIONS = (
    "torch",
    "transformers",
    "kokoro",
    "misaki",
    "spacy",
    "en_core_web_sm",
    "espeakng-loader",
    "phonemizer-fork",
    "num2words",
    "scipy",
    "numpy",
    "huggingface-hub",
)


def work_key(name: str) -> str:
    return sources.speech_key(f"work/{name}")


def _sha(body: bytes) -> str:
    import hashlib

    return hashlib.sha256(body).hexdigest()


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for name in VERSIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _batch(gpu: str, batch: int) -> int:
    """Whisper's batch: the one given, else what the GPU fits."""
    return batch or (96 if gpu == "H100" else 32)


# --- prepare: the draw -----------------------------------------------------------------------------------------------


@job(DATA, cpu=2.0, memory=32768, timeout=40 * 60)
def draw_lines() -> dict:
    import time
    from collections import Counter

    from lab.reader.data import r2_open_lines

    storage.reload()
    t0 = time.time()
    read: dict[str, dict] = {}

    def jsonl(key: str) -> list[dict]:
        body = storage.get(key)
        read[key] = {"sha256": _sha(body), "bytes": len(body)}
        return [json.loads(x) for x in body.decode().splitlines() if x.strip()]

    open_lines = r2_open_lines(jsonl)
    pool = jsonl(sources.pool_key())
    drawn, counts = speech.draw(open_lines, pool)
    del open_lines, pool
    samples = speech.pick_samples(drawn)
    storage.put_jsonl(work_key("draw.jsonl"), drawn)
    tasks = [speech.task(ln, ln["line_id"] in samples) for ln in drawn]
    storage.put_jsonl(work_key("tasks.jsonl"), tasks)
    counts.update(
        samples=len(samples),
        samples_by_source=dict(Counter(ln["source"] for ln in drawn if ln["line_id"] in samples)),
        voices=dict(sorted(Counter(t["voice"] for t in tasks).items())),
        chars={
            "total": sum(len(t["text"]) for t in tasks),
            "mean": round(sum(len(t["text"]) for t in tasks) / max(1, len(tasks)), 1),
        },
        read=read,
        open_draw="lab.reader.data.r2_open_lines (each source's train split under its cap, sources.OPEN_CAPS)",
        pool_key=sources.pool_key(),
        seconds=round(time.time() - t0),
    )
    storage.put_json(work_key("draw.json"), counts)
    return counts


# --- speak: G2P, Kokoro, Whisper -------------------------------------------------------------------------------------

_G2P: dict = {}


def _g2p_init() -> None:
    import logging

    from kokoro import KPipeline

    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    for lang in ("a", "b"):
        _G2P[lang] = KPipeline(lang_code=lang, repo_id=speech.TTS["repo"], model=False)


def _g2p_one(item: tuple[str, str, str]) -> tuple[str, list[str], str | None]:
    """A line's phoneme chunks, as Kokoro's pipeline makes them (misaki's G2P, then chunks of at most 510)."""
    line_id, text, lang = item
    try:
        pipe = _G2P[lang]
        _, tokens = pipe.g2p(text)
        return line_id, [ps[:510] for _, ps, _ in pipe.en_tokenize(tokens) if ps], None
    except Exception as e:  # noqa: BLE001 - counted per line
        return line_id, [], f"{type(e).__name__}: {str(e)[:160]}"


def _g2p_all(items: list[tuple[str, str, str]], procs: int) -> dict[str, tuple[list[str], str | None]]:
    """G2P for every line in a pool of forked CPU processes, before this process touches CUDA."""
    import multiprocessing as mp

    with mp.get_context("fork").Pool(procs, initializer=_g2p_init) as pool:
        return {lid: (ps, err) for lid, ps, err in pool.imap_unordered(_g2p_one, items, chunksize=32)}


def _check_kokoro(model, weights_path: str) -> dict:
    """Kokoro loads its checkpoint non-strictly when a strict load fails, which would leave a renamed weight random
    under a newer transformers. Every checkpoint tensor must be in the model with the same values, and every parameter
    loaded, except the scale and shift of the AdaIN layers' InstanceNorm1d: Kokoro builds them with affine=True only to
    get round an old ONNX export bug, so they are not in the checkpoint and must sit at the identity (1 and 0)."""
    import torch

    ck = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, differ, not_loaded, checked, identity = [], [], [], 0, 0
    for key, state in ck.items():
        mod = getattr(model, key)
        msd = mod.state_dict()
        names = set()
        for k, v in state.items():
            name = k[7:] if k.startswith("module.") else k
            names.add(name)
            if name not in msd:
                if not name.endswith("position_ids"):
                    missing.append(f"{key}.{name}")
                continue
            checked += 1
            if not torch.equal(msd[name].detach().cpu().to(v.dtype), v):
                differ.append(f"{key}.{name}")
        for n, p in mod.named_parameters():
            if n in names:
                continue
            owner, _, leaf = n.rpartition(".")
            sub = mod.get_submodule(owner) if owner else mod
            want = 1.0 if leaf == "weight" else 0.0
            if (
                isinstance(sub, torch.nn.InstanceNorm1d)
                and leaf in ("weight", "bias")
                and bool((p.detach() == want).all())
            ):
                identity += 1
            else:
                not_loaded.append(f"{key}.{n}")
    if missing or differ or not_loaded:
        raise RuntimeError(
            f"Kokoro weights not loaded as saved: missing {missing[:5]}, differ {differ[:5]}, "
            f"not loaded {not_loaded[:5]} ({len(missing)}, {len(differ)}, {len(not_loaded)})"
        )
    return {"tensors_checked": checked, "instance_norm_affine_at_identity": identity}


def _load_kokoro(device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from kokoro import KModel

    repo, rev = speech.TTS["repo"], speech.TTS["revision"]
    files = {
        "config.json": hf_hub_download(repo, "config.json", revision=rev),
        speech.TTS["weights"]: hf_hub_download(repo, speech.TTS["weights"], revision=rev),
    }
    model = (
        KModel(repo_id=repo, config=files["config.json"], model=files[speech.TTS["weights"]])
        .to(device)
        .eval()
    )
    check = _check_kokoro(model, files[speech.TTS["weights"]])
    packs = {}
    for v, _ in speech.VOICES:
        files[f"voices/{v}.pt"] = hf_hub_download(repo, f"voices/{v}.pt", revision=rev)
        packs[v] = torch.load(files[f"voices/{v}.pt"], weights_only=True).to(device)
    return model, packs, files, check


def _load_whisper(device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repo, rev = speech.ASR["repo"], speech.ASR["revision"]
    proc = WhisperProcessor.from_pretrained(repo, revision=rev)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            repo, revision=rev, dtype=torch.float16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    return model, proc


def _speak(model, packs: dict, voice: str, chunks: list[str]):
    """One line's audio: Kokoro on each phoneme chunk, joined, 24 kHz to 16 kHz, as int16."""
    import numpy as np
    from kokoro import KPipeline
    from scipy.signal import resample_poly

    parts = []
    for ps in chunks:
        a = KPipeline.infer(model, ps, packs[voice], 1.0).audio
        if a is not None:
            parts.append(a.detach().float().cpu().numpy().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.int16)
    a16 = resample_poly(np.concatenate(parts), 2, 3)
    return (np.clip(a16, -1.0, 1.0) * 32767).astype(np.int16)


def _phone(a16):
    """A telephone channel: 8 kHz, 8-bit mu-law, back to 16 kHz."""
    import numpy as np
    from scipy.signal import resample_poly

    mu = 255.0
    x = np.clip(resample_poly(a16.astype(np.float32) / 32767.0, 1, 2), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) / 2.0 * mu) / mu * 2.0 - 1.0
    x = np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu
    return (np.clip(resample_poly(x, 2, 1), -1.0, 1.0) * 32767).astype(np.int16)


def _transcribe(model, proc, audios: list, batch: int, device: str) -> list[str]:
    import numpy as np
    import torch

    out: list[str] = []
    for i in range(0, len(audios), batch):
        chunk = [a.astype(np.float32) / 32767.0 for a in audios[i : i + batch]]
        feats = proc.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", device=device)
        x = feats.input_features.to(device, torch.float16)
        with torch.inference_mode():
            ids = model.generate(
                input_features=x,
                language="en",
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=128,
            )
        out += [" ".join(t.split()) for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return out


def _wav(a16, rate: int = 16000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(a16.tobytes())
    return buf.getvalue()


def _pilot_reads(tasks: list[dict], rows: list[dict], examples: int = 24) -> dict:
    """Word error rate on the pilot lines (counts), as given and through the telephone channel, and a few pairs to
    read."""
    texts = {t["line_id"]: t["text"] for t in tasks}
    out = {}
    for field in ("transcript", "transcript_phone"):
        ws = [
            (
                speech.wer(texts[r["line_id"]], r.get(field) or ""),
                *speech.wer_counts(texts[r["line_id"]], r.get(field) or ""),
            )
            for r in rows
            if r.get("seconds") and (r.get(field) or "").strip()
        ]
        same = sum(
            1
            for r in rows
            if r.get("seconds") and speech.norm(r.get(field)) == speech.norm(texts[r["line_id"]])
        )
        out[field] = {
            **speech.distribution(ws),
            "same_as_clean": same,
            "over_0.6": sum(1 for w in ws if w[0] > speech.MAX_WER),
            "empty": sum(1 for r in rows if r.get("seconds") and not speech.norm(r.get(field))),
        }
    picks = sorted(rows, key=lambda r: speech.h("read", r["line_id"]))[:examples]
    # Printed to read, never saved.
    out["examples"] = [
        {
            "clean": texts[r["line_id"]],
            "asr": r.get("transcript"),
            "phone": r.get("transcript_phone"),
            "voice": r.get("voice"),
        }
        for r in picks
    ]
    return out


@job(
    SPEECH,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=3 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0),
)
def speak_shard(
    shard: int,
    n_shards: int,
    tag: str = "run",
    pilot_n: int = 0,
    phone: bool = False,
    block: int = 2000,
    batch: int = 32,
    procs: int = 7,
) -> dict:
    """One shard (lines whose bucket % n_shards == shard), or with `pilot_n` the pilot's lines. Resumes from its
    parts."""
    import time

    storage.reload()
    t0 = time.time()
    tasks = storage.jsonl(work_key("tasks.jsonl"))
    if pilot_n:
        tasks = sorted(tasks, key=lambda t: speech.h("pilot", t["line_id"]))[:pilot_n]
        prefix = work_key(tag)
    else:
        tasks = [t for t in tasks if t["bucket"] % n_shards == shard]
        prefix = work_key(f"{tag}/shard-{shard:02d}")
    parts = sorted(k for k in storage.keys(prefix) if "/part-" in k)
    done = {r["line_id"] for k in parts for r in storage.jsonl(k)}
    todo = [t for t in tasks if t["line_id"] not in done]
    timing = {"read": round(time.time() - t0, 1)}

    t1 = time.time()
    phon = _g2p_all([(t["line_id"], t["text"], t["lang"]) for t in todo], procs) if todo else {}
    timing["g2p"] = round(time.time() - t1, 1)

    import torch

    device = "cuda"
    t1 = time.time()
    kmodel, packs, _, check = _load_kokoro(device)
    asr, proc = _load_whisper(device)
    timing["load"] = round(time.time() - t1, 1)
    timing["tts"] = timing["asr"] = 0.0
    n_err = n_silent = 0
    audio_s = 0.0
    all_rows: list[dict] = []
    for b in range(0, len(todo), block):
        blk = todo[b : b + block]
        t1 = time.time()
        rows, audios, idx = [], [], []
        for t in blk:
            ps, err = phon.get(t["line_id"], ([], "no g2p result"))
            row = {"line_id": t["line_id"], "voice": t["voice"], "seconds": 0.0, "transcript": None}
            if err:
                row["error"] = f"g2p: {err}"
            else:
                try:
                    with torch.inference_mode():
                        a = _speak(kmodel, packs, t["voice"], ps)
                    row["seconds"] = round(len(a) / 16000, 3)
                    if len(a):
                        audios.append(a)
                        idx.append(len(rows))
                except Exception as e:  # noqa: BLE001 - counted per line
                    row["error"] = f"tts: {type(e).__name__}: {str(e)[:160]}"
            n_err += bool(row.get("error"))
            n_silent += not row.get("error") and not row["seconds"]
            rows.append(row)
        timing["tts"] += time.time() - t1
        t1 = time.time()
        if phone:  # the pilot: every clip as spoken, and again through the telephone channel
            for i, text in zip(idx, _transcribe(asr, proc, audios, batch, device)):
                rows[i]["transcript"] = text
            for i, text in zip(idx, _transcribe(asr, proc, [_phone(a) for a in audios], batch, device)):
                rows[i]["transcript_phone"] = text
        else:  # each line through its own channel; a sample keeps what Whisper heard
            for i in idx:
                rows[i]["channel"] = speech.channel_for(rows[i]["line_id"])
            audios = [_phone(a) if rows[i]["channel"] == "phone" else a for i, a in zip(idx, audios)]
            for i, text in zip(idx, _transcribe(asr, proc, audios, batch, device)):
                rows[i]["transcript"] = text
        timing["asr"] += time.time() - t1
        audio_s += sum(len(a) for a in audios) / 16000
        key = f"{prefix}/part-{len(parts):04d}.jsonl"
        storage.put_jsonl(key, rows)
        parts.append(key)
        if not pilot_n:  # the sample clips, from the full run only
            for i, a in zip(idx, audios):
                t = blk[i]
                if t.get("sample") and len(a) <= 8 * 16000:
                    storage.put(sources.speech_key(f"samples/{t['line_id']}.wav"), _wav(a))
        all_rows += rows
        print(
            f"{prefix}: {b + len(blk)}/{len(todo)} lines, tts {timing['tts']:.0f} s, asr {timing['asr']:.0f} s",
            flush=True,
        )
    timing = {k: round(v, 1) for k, v in timing.items()}
    stats = {
        "shard": shard,
        "n_shards": n_shards,
        "tag": tag,
        "lines": len(tasks),
        "resumed_from": len(done),
        "spoken": len(todo),
        "errors": n_err,
        "no_audio": n_silent,
        "audio_seconds": round(audio_s, 1),
        "gpu": torch.cuda.get_device_name(0),
        "batch": batch,
        "block": block,
        "procs": procs,
        "timing": timing,
        "wall": round(time.time() - t0, 1),
        "kokoro_check": check,
        "versions": _versions(),
    }
    if todo:
        stats["per_line_ms"] = {k: round(1000 * timing[k] / len(todo), 2) for k in ("g2p", "tts", "asr")}
    if pilot_n:
        stats["reads"] = _pilot_reads(tasks, all_rows)
    storage.put_json(f"{prefix}/done.json", {k: v for k, v in stats.items() if k != "reads"})
    return stats


@job(SPEECH, gpu="L4", cpu=4.0, memory=16384, timeout=30 * 60)
def check_models() -> dict:
    """The stack on a few made-up sentences, and the models' files: hashes, licence fields, package versions."""
    import hashlib
    import re
    import time

    from huggingface_hub import hf_hub_download

    sentences = [
        ("af_heart", "a", "Hi, I'd like to move my appointment to Thursday at 3:30, if that's okay."),
        ("am_michael", "a", "Oh great, thanks a lot. That's exactly what I needed today."),
        ("bf_emma", "b", "Sorry, we're fully booked tonight, but I can offer you a table tomorrow."),
        ("bm_george", "b", "Um, yeah, I think the card was charged twice for the same order."),
        ("am_puck", "a", "Mm-hmm."),
        ("bf_alice", "b", "Could you spell the reference number for me, please? It begins with Q."),
    ]
    t0 = time.time()
    phon = _g2p_all([(str(i), s, lang) for i, (_, lang, s) in enumerate(sentences)], 2)
    t1 = time.time()
    kmodel, packs, files, check = _load_kokoro("cuda")
    asr, proc = _load_whisper("cuda")
    t2 = time.time()
    audios = [_speak(kmodel, packs, v, phon[str(i)][0]) for i, (v, _, _) in enumerate(sentences)]
    t3 = time.time()
    texts = _transcribe(asr, proc, audios, 8, "cuda")
    t4 = time.time()

    def sha(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def front_licence(repo: str, rev: str) -> str | None:
        text = pathlib.Path(hf_hub_download(repo, "README.md", revision=rev)).read_text()
        m = re.search(r"^license:\s*(\S+)", text.split("---")[1] if text.startswith("---") else "", re.M)
        return m.group(1) if m else None

    whisper_weights = hf_hub_download(
        speech.ASR["repo"], speech.ASR["weights"], revision=speech.ASR["revision"]
    )
    models = {
        "tts": {
            **speech.TTS,
            "readme_license_field": front_licence(speech.TTS["repo"], speech.TTS["revision"]),
            "sha256": {name: sha(path) for name, path in files.items()},
            "kokoro_check": check,
            "voices": [v for v, _ in speech.VOICES],
        },
        "asr": {
            **speech.ASR,
            "readme_license_field": front_licence(speech.ASR["repo"], speech.ASR["revision"]),
            "sha256": {speech.ASR["weights"]: sha(whisper_weights)},
            "decoding": "greedy (num_beams 1, no sampling), language en, task transcribe, no prompt, no timestamps, "
            "max 128 new tokens, fp16, SDPA attention",
        },
        "versions": _versions(),
    }
    storage.put_json(work_key("models.json"), models)
    return {
        "models": models,
        "timing": {
            "g2p": round(t1 - t0, 1),
            "load": round(t2 - t1, 1),
            "tts": round(t3 - t2, 2),
            "asr": round(t4 - t3, 2),
        },
        "pairs": [
            {
                "voice": v,
                "said": s,
                "heard": h_,
                "seconds": round(len(a) / 16000, 2),
                "wer": round(speech.wer(s, h_), 3),
            }
            for (v, _, s), h_, a in zip(sentences, texts, audios)
        ],
    }


# --- collect, verify, read -------------------------------------------------------------------------------------------


def _results(tag: str) -> tuple[dict[str, dict], list[dict]]:
    keys = storage.keys(work_key(tag))
    results: dict[str, dict] = {}
    for k in sorted(k for k in keys if "/part-" in k):
        for r in storage.jsonl(k):
            results[r["line_id"]] = r
    dones = [json.loads(storage.get(k)) for k in sorted(k for k in keys if k.endswith("/done.json"))]
    return results, dones


@job(DATA, cpu=2.0, memory=16384, timeout=60 * 60)
def build_twins(tag: str = "run", n_shards: int = 10) -> dict:
    import time

    storage.reload()
    t0 = time.time()
    drawn = storage.jsonl(work_key("draw.jsonl"))
    results, dones = _results(tag)
    missing_shards = sorted(set(range(n_shards)) - {d["shard"] for d in dones})
    twins, summary, details = speech.summarize(drawn, results)
    body = "".join(json.dumps(t) + "\n" for t in twins).encode()
    storage.put(sources.speech_key(), body)
    storage.put_jsonl(sources.speech_key("details.jsonl"), details)
    samples = sorted(k for k in storage.keys(sources.speech_key("samples")) if k.endswith(".wav"))
    by_id = {ln["line_id"]: ln for ln in drawn}
    index = []
    for k in samples:
        lid = k.rsplit("/", 1)[1][:-4]
        r = results.get(lid) or {}
        index.append(
            {
                "line_id": lid,
                "key": k,
                "source": by_id[lid]["source"],
                "voice": r.get("voice"),
                "channel": r.get("channel"),
                "seconds": r.get("seconds"),
                "clean": by_id[lid]["input"]["text"],
                "transcript": r.get("transcript"),
            }
        )
    storage.put_json(sources.speech_key("samples/index.json"), index)  # line text: the volume only
    models = (
        json.loads(storage.get(work_key("models.json"))) if storage.exists(work_key("models.json")) else None
    )
    timing = {k: round(sum(d["timing"].get(k, 0) for d in dones), 1) for k in ("g2p", "load", "tts", "asr")}
    summary.update(
        {
            "key": sources.speech_key(),
            "sha256": _sha(body),
            "bytes": len(body),
            "details_key": sources.speech_key("details.jsonl"),
            "samples": {
                "prefix": sources.speech_key("samples/"),
                "clips": len(samples),
                "index": sources.speech_key("samples/index.json"),
            },
            "shards": {
                "n": n_shards,
                "done": len(dones),
                "missing": missing_shards,
                "gpu": sorted({d["gpu"] for d in dones}),
                "container_seconds": round(sum(d["wall"] for d in dones), 1),
                "phase_seconds": timing,
                "errors": sum(d["errors"] for d in dones),
                "resumed": sum(1 for d in dones if d["resumed_from"]),
            },
            "models": models,
            "built_by": "lab/reader/pipeline/speech.py collect; lab/reader/speech.py",
            "seconds": round(time.time() - t0),
        }
    )
    storage.put_json(sources.speech_key("summary.json"), summary)
    return summary


@job(DATA, cpu=2.0, memory=16384, timeout=30 * 60)
def verify_twins() -> dict:
    """Every twin re-read against its clean line, apart from the code that wrote it."""
    import hashlib
    from collections import Counter

    from lab.reader.schemas import LineInput

    storage.reload()
    body = storage.get(sources.speech_key())
    summ = json.loads(storage.get(sources.speech_key("summary.json")))
    drawn = {ln["line_id"]: ln for ln in storage.jsonl(work_key("draw.jsonl"))}
    bad: Counter = Counter()
    seen = set()
    lower = 0
    rows = [json.loads(x) for x in body.decode().splitlines() if x.strip()]
    for t in rows:
        c = drawn.get(t.get("twin_of"))
        if c is None:
            bad["twin of a line outside the draw"] += 1
            continue
        bad["repeated line_id"] += t["line_id"] in seen
        seen.add(t["line_id"])
        bad["line_id not <clean id>-asr"] += t["line_id"] != c["line_id"] + "-asr"
        try:
            LineInput.model_validate(t["input"])
        except Exception:  # noqa: BLE001 - counted
            bad["input refused by LineInput"] += 1
        others = set(c) | set(t)
        bad["a field other than text, slices, twin_of, generator changed"] += any(
            t.get(k) != c.get(k)
            for k in others
            if k not in ("line_id", "input", "slices", "twin_of", "generator")
        )
        bad["an input field other than text changed"] += any(
            t["input"].get(k) != c["input"].get(k) for k in set(c["input"]) | set(t["input"]) if k != "text"
        )
        bad["slices not the clean line's plus speech_twin"] += t["slices"] != [*c["slices"], "speech_twin"]
        bad["generator not both models"] += t["generator"] != speech.GENERATOR
        bad["not train"] += t["split"] != "train"
        w = speech.wer(c["input"]["text"], t["input"]["text"])
        bad["wer over 0.6"] += w > speech.MAX_WER
        bad["normalised text equals the clean line's"] += speech.norm(t["input"]["text"]) == speech.norm(
            c["input"]["text"]
        )
        bad["empty text"] += not t["input"]["text"].strip()
        if speech.lowercased(c["line_id"]):
            lower += 1
            bad["lowercased twin not in raw speech-to-text form"] += t["input"]["text"] != speech.norm(
                t["input"]["text"]
            )
    return {
        "rows": len(rows),
        "sha256_matches_summary": hashlib.sha256(body).hexdigest() == summ["sha256"],
        "rows_match_summary": len(rows) == summ["twins_kept"],
        "lowercased": lower,
        "lowercased_matches_summary": lower == summ["lowercased"]["twins"],
        "problems": {k: v for k, v in bad.items() if v},
    }


@job(DATA, cpu=1.0, memory=8192, timeout=15 * 60)
def sample_twins(tag: str = "run", n: int = 40, dropped: int = 12) -> dict:
    """Twins, and lines dropped from `tag`'s results, to read in the session (line text: printed, never saved)."""
    storage.reload()
    twins = storage.jsonl(sources.speech_key())
    drawn = {ln["line_id"]: ln for ln in storage.jsonl(work_key("draw.jsonl"))}
    results, _ = _results(tag)
    pick = sorted(twins, key=lambda t: speech.h("read", t["line_id"]))[:n]
    out = {
        "twins": [
            {
                "source": t["source"],
                "kind": speech.kind_of(t),
                "voice": (results.get(t["twin_of"]) or {}).get("voice"),
                "channel": (results.get(t["twin_of"]) or {}).get("channel"),
                "clean": drawn[t["twin_of"]]["input"]["text"],
                "twin": t["input"]["text"],
                "wer": round(speech.wer(drawn[t["twin_of"]]["input"]["text"], t["input"]["text"]), 3),
            }
            for t in pick
        ]
    }
    for reason in ("wer_over_0.6", "same_as_clean", "empty"):
        rows = []
        for lid, ln in sorted(drawn.items(), key=lambda kv: speech.h("read-drop", kv[0])):
            _, outcome, w = speech.outcome_of(ln, results.get(lid))
            if outcome == reason:
                rows.append(
                    {
                        "source": ln["source"],
                        "channel": (results.get(lid) or {}).get("channel"),
                        "clean": ln["input"]["text"],
                        "heard": (results.get(lid) or {}).get("transcript"),
                        "wer": None if w is None else round(w, 3),
                    }
                )
            if len(rows) >= dropped:
                break
        out[reason] = rows
    return out


# --- the local entrypoints -------------------------------------------------------------------------------------------


@app.local_entrypoint()
def prepare():
    """The fixed draw (``speech.draw``: about 60,000 lines of r2's open draw and 40,000 of the line classifier's pool) on one
    CPU container, under work/: draw.jsonl (the clean lines), tasks.jsonl (what a speaking worker needs: id, text,
    voice, shard bucket, sample flag) and draw.json (counts, and the sha256 of every file read), also saved here."""
    res = draw_lines.remote()
    save("speech", "draw.json", res)
    keys = ("lines", "by_source", "by_group_and_kind", "repeated_text_skipped", "samples", "seconds")
    print(json.dumps({k: res[k] for k in keys}, indent=1))


@app.local_entrypoint()
def smoke():
    """The stack on a few made-up sentences, printed to read, and the models' files: hashes, licence fields and
    package versions, saved here and to work/models.json."""
    res = check_models.remote()
    for p in res["pairs"]:
        print(json.dumps(p))
    save(
        "speech",
        "smoke.json",
        {"timing": res["timing"], "models": res["models"], "wer": [p["wer"] for p in res["pairs"]]},
    )
    print(
        json.dumps(
            {
                "timing": res["timing"],
                "versions": res["models"]["versions"],
                "licence fields": [
                    res["models"]["tts"]["readme_license_field"],
                    res["models"]["asr"]["readme_license_field"],
                ],
                "kokoro_check": res["models"]["tts"]["kokoro_check"],
            },
            indent=1,
        )
    )


@app.local_entrypoint()
def pilot(pilot_n: int = 400, batch: int = 0):
    """The timing test that picks the GPU: the same `pilot_n` lines spoken on an L4 and an H100, each clip transcribed
    as spoken and again through the telephone channel. Counts saved here; a few pairs printed to read."""
    calls = {
        g.lower(): speak_shard.with_options(gpu=g).spawn(
            -1,
            0,
            tag=f"pilot-{g.lower()}",
            pilot_n=pilot_n,
            phone=True,
            batch=_batch(g, batch),
        )
        for g in ("L4", "H100")
    }
    out = {}
    for g, call in calls.items():
        try:
            res = call.get()
        except Exception as e:  # noqa: BLE001
            print("FAILED", g, repr(e))
            continue
        ex = res["reads"].pop("examples")
        out[g] = res
        keys = ("gpu", "spoken", "errors", "no_audio", "audio_seconds", "timing", "per_line_ms", "wall")
        print(f"\n######## {g}: {json.dumps({k: res[k] for k in keys})}")
        print(json.dumps({k: res["reads"][k] for k in ("transcript", "transcript_phone")}))
        if g == "l4":
            for e in ex:
                print(json.dumps(e))
    save("speech", "pilot.json", out)


@app.local_entrypoint()
def speak(shards: int = 10, gpu: str = "L4", batch: int = 0, tag: str = "run"):
    """Every drawn line spoken and heard again, one GPU container per shard, spawned to keep going without this client.
    misaki's G2P runs in a pool of CPU processes first; then Kokoro-82M speaks each line in its voice (24 kHz, resampled
    to 16 kHz), half the clips go through a telephone channel (8 kHz, 8-bit mu-law; ``speech.channel_for``), and Whisper
    large-v3-turbo transcribes them (English, no prompt, greedy). Results go out in parts of `block` lines
    (work/<tag>/shard-NN/part-NNNN.jsonl), so a restarted container goes on where it stopped, and done.json closes a
    shard with its timing. No audio is kept but the about 50 sample lines' clips as Whisper heard them
    (samples/<line_id>.wav)."""
    batch = _batch(gpu, batch)
    calls = []
    for k in range(shards):
        call = speak_shard.with_options(gpu=gpu).spawn(k, shards, tag=tag, batch=batch)
        print("spawned shard", k, "of", shards, "on", gpu, "call", call.object_id)
        calls.append(call.object_id)
    save(
        "speech",
        f"speak-spawned-{tag}.json",
        {"calls": calls, "tag": tag, "shards": shards, "gpu": gpu, "batch": batch},
    )


@app.local_entrypoint()
def status(shards: int = 10, tag: str = "run"):
    """The shards' progress, from the volume."""
    keys = storage.keys(work_key(tag))
    parts = [k for k in keys if "/part-" in k]
    dones = [json.loads(storage.get(k)) for k in keys if k.endswith("/done.json") and "/shard-" in k]
    print(f"{len(parts)} parts, {len(dones)} of {shards} shards done")
    for d in sorted(dones, key=lambda d: d["shard"]):
        print(
            json.dumps(
                {k: d[k] for k in ("shard", "lines", "spoken", "errors", "no_audio", "gpu", "wall", "timing")}
            )
        )


@app.local_entrypoint()
def collect(shards: int = 10, tag: str = "run"):
    """The twins (``speech.make_twin``) on one CPU container to lines.jsonl, with summary.json (drawn, kept, drops by
    reason, word error rates, the share lowercased; also saved here) and details.jsonl (one row per drawn line, no
    text) beside it."""
    res = build_twins.remote(tag=tag, n_shards=shards)
    save("speech", "summary.json", res)
    print(
        json.dumps(
            {k: res[k] for k in ("lines_drawn", "results", "twins_kept", "dropped", "lowercased", "shards")},
            indent=1,
        )
    )
    print(json.dumps({k: res["wer"][k] for k in ("all", "kept")}, indent=1))


@app.local_entrypoint()
def verify():
    """lines.jsonl re-read apart from the code that wrote it."""
    res = verify_twins.remote()
    save("speech", "verify.json", res)
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def read(tag: str = "run"):
    """A sample of twins and dropped lines, printed to read, never saved."""
    res = sample_twins.remote(tag)
    for k, rows in res.items():
        print(f"\n######## {k}")
        for r in rows:
            print(json.dumps(r))


@app.local_entrypoint()
def manifest():
    """results/reader/manifests/tonetext-speech-0.1.yaml from the counts on the volume: no line text."""
    summ = json.loads(storage.get(sources.speech_key("summary.json")))
    drawc = json.loads(storage.get(work_key("draw.json")))
    models = summ.get("models") or json.loads(storage.get(work_key("models.json")))
    v = models["versions"]
    out = {
        "corpus_id": sources.SPEECH_ID,
        "track": "the reader's r2 pool: the speech round trip",
        "format": "noisy twins: copies of their clean lines in the line classifier's corpus-line shape (the open lines' "
        "`native` kept): line_id '<clean id>-asr', twin_of the clean id, input.text the transcript, the line "
        "before and every other field unchanged, slices plus speech_twin, generator naming both models; every input "
        "checked with lab.reader.schemas.LineInput",
        "key": sources.speech_key(),
        "sha256": summ["sha256"],
        "bytes": summ["bytes"],
        "summary_key": sources.speech_key("summary.json"),
        "details_key": summ["details_key"],
        "lines_drawn": summ["lines_drawn"],
        "twins_kept": summ["twins_kept"],
        "draw": {
            "rule": "fixed by sha256 of the line id: r2's open draw (lab.reader.data.r2_open_lines), user and agent "
            "turns of the two-party sources (two thirds the user's) and AMI participants' turns, and the line classifier's "
            "pool (runs/linecls/pool/linecls-pool-0.6/lines.jsonl), clean lines only, nine in ten the player's; TRAIN "
            "split; texts of 3 to 200 characters with a letter; machine transcripts never spoken again; one normalised "
            "text at most twice",
            "lines": drawc["lines"],
            "by_source": drawc["by_source"],
            "by_group_and_kind": drawc["by_group_and_kind"],
            "drawn": drawc["drawn"],
            "eligible": drawc["eligible"],
            "repeated_text_skipped": drawc["repeated_text_skipped"],
            "quotas": drawc["quotas"],
            "inputs_read": drawc["read"],
        },
        "dropped": summ["dropped"],
        "wer_over_0.6_numerals_one_side": summ["wer_over_0.6_numerals_one_side"],
        "by_source": summ["by_source"],
        "by_group_and_kind": summ["by_group_and_kind"],
        "by_channel": summ["by_channel"],
        "channel_rule": "sha256 of the clean line id even: a telephone channel before Whisper (16 kHz to 8 kHz, 8-bit "
        "mu-law, back to 16 kHz), as a phone call; else the clip as spoken (16 kHz), as a WebRTC call",
        "wer": {
            k: summ["wer"][k] for k in ("note", "all", "kept", "by_source", "by_group_and_kind", "by_channel")
        },
        "quotation_marks_removed": {
            "transcripts": summ["quotation_marks_removed"],
            "rule": "Whisper's double quotes at either end of a transcript, then single quotes that open and close it, "
            "removed before the twin is built (no speech-to-text service writes them)",
        },
        "lowercased": {
            **summ["lowercased"],
            "rule": "sha256 of the clean line id even: lowercase, every punctuation mark but the apostrophe removed; "
            "else Whisper's own writing",
        },
        "voices_kept": summ["voices_kept"],
        "audio_seconds": summ["audio_seconds"],
        "models": {
            "tts": {
                k: models["tts"][k]
                for k in (
                    "name",
                    "repo",
                    "revision",
                    "weights",
                    "licence",
                    "licence_source",
                    "readme_license_field",
                    "sha256",
                    "voices",
                )
            }
            | {
                "inference": f"kokoro {v['kokoro']} (github.com/hexgrad/kokoro, Apache-2.0), speed 1.0, a voice per line "
                "by sha256 of its id; 24 kHz output resampled to 16 kHz"
            },
            "asr": {
                k: models["asr"][k]
                for k in (
                    "name",
                    "repo",
                    "revision",
                    "weights",
                    "licence",
                    "licence_source",
                    "readme_license_field",
                    "sha256",
                    "decoding",
                )
            }
            | {"inference": f"transformers {v['transformers']}"},
        },
        "tools": {
            "note": "programs run at build time; none of their code or data is in the twins",
            "g2p": f"misaki {v['misaki']} (Apache-2.0) with spaCy {v['spacy']} (MIT) and en_core_web_sm "
            f"{v['en_core_web_sm']} (MIT)",
            "g2p_fallback": f"eSpeak NG (GPL-3.0-or-later), loaded by espeakng-loader {v['espeakng-loader']} (no licence in "
            f"its PyPI metadata), through phonemizer-fork {v['phonemizer-fork']} (GPL-3.0-or-later), for words outside "
            "misaki's dictionary",
            "numbers": f"num2words {v['num2words']} (LGPL)",
            "resampling": f"scipy {v['scipy']} (BSD-3-Clause)",
            "torch": v["torch"],
        },
        "samples": summ["samples"],
        "shards": summ["shards"],
        "use": "training only (twins of TRAIN lines; each takes its clean line's labels). Line text stays on the volume",
        "built_by": "lab/reader/pipeline/speech.py (prepare, speak, collect); lab/reader/speech.py",
    }
    save_yaml("manifests", f"{sources.SPEECH_ID}.yaml", out)
    print(json.dumps({k: out[k] for k in ("lines_drawn", "twins_kept", "sha256", "dropped")}, indent=1))
