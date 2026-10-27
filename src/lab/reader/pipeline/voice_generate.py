"""r1's voice-agent test set: conversations written by a teacher model (``lab.reader.voice``), three lines taken from
each. Test only, never trained on.

    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.voice_generate::generate --try-no 1
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.voice_generate::status --try-no 1
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.voice_generate::select --try-no 1
"""

import json
import time

from lab.reader import sources, storage, voice
from lab.reader.app import DATA, VLLM, app, hf_cache, job, save

TEACHER_REPO = "Qwen/Qwen3.8-27B-FP8"
TEACHER_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 1400}
N_PER_DOMAIN = 33  # prompts per domain: 429 conversations, 11 per seed, against 8, 8 and 7 taken
PER_DOMAIN = 23  # conversations taken per domain: 299, three lines each


def raw_key(try_no: int, name: str) -> str:
    """A try's teacher output, beside the set."""
    return f"runs/tonetext/generated/{sources.VOICE_ID}-raw/try{try_no}/{name}"


@job(VLLM, gpu="H100", cpu=8.0, memory=65536, timeout=90 * 60, volumes={"/cache": hf_cache})
def write_conversations(n_per_domain: int, seed: int, try_no: int) -> dict:
    from collections import Counter

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    t0 = time.perf_counter()
    prompts = voice.build_prompts(n_per_domain, seed=seed, prefix=f"tv{try_no}")
    llm = LLM(
        model=TEACHER_REPO,
        revision=TEACHER_REVISION,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        seed=seed,
        max_num_seqs=256,
    )
    load_s = time.perf_counter() - t0
    # one structured-output schema per prompt: the rounds alternate by construction
    params = [
        SamplingParams(
            n=1,
            temperature=SAMPLING["temperature"],
            top_p=SAMPLING["top_p"],
            max_tokens=SAMPLING["max_tokens"],
            seed=seed,
            structured_outputs=StructuredOutputsParams(json=voice.schema(p["first"])),
        )
        for p in prompts
    ]
    outs = llm.chat(
        [p["messages"] for p in prompts],
        params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    gen_s = time.perf_counter() - t0 - load_s
    rows, problems, finish = [], Counter(), Counter()
    tok_in = tok_out = 0
    for p, o in zip(prompts, outs):
        text = o.outputs[0].text
        turns, probs = voice.parse_json_conversation(text, p["first"])
        problems.update(probs)
        finish[str(o.outputs[0].finish_reason)] += 1
        tok_in += len(o.prompt_token_ids)
        tok_out += len(o.outputs[0].token_ids)
        rows.append(
            {
                **{k: v for k, v in p.items() if k != "messages"},
                "prompt": p["messages"][1]["content"],
                "raw": text,
                "turns": turns,
                "problems": probs,
                "finish_reason": str(o.outputs[0].finish_reason),
            }
        )
    storage.put_jsonl(raw_key(try_no, "conversations.jsonl"), rows)
    result = {
        "gen_id": sources.VOICE_ID,
        "try": try_no,
        "prompts": len(prompts),
        "usable": sum(voice.usable(r) for r in rows),
        "marked_ok": sum(voice.marked_ok(r) for r in rows),
        "turns_median": sorted(len(r["turns"]) for r in rows)[len(rows) // 2],
        "problems": dict(problems),
        "finish_reasons": dict(finish),
        "tokens": {"in": tok_in, "out": tok_out},
        "seconds": {"load": round(load_s), "generate": round(gen_s)},
        "teacher": f"{TEACHER_REPO}@{TEACHER_REVISION}",
        "sampling": {**SAMPLING, "seed": seed},
        "key": raw_key(try_no, "conversations.jsonl"),
    }
    storage.put_json(raw_key(try_no, "result.json"), result)
    return result


def training_texts() -> tuple[set, set]:
    """The norm_key texts, and (text, line before) pairs, of every line r1's student trains on, twins included."""
    from lab.reader import data

    texts, pairs = set(), set()
    for _, ln in data.r1_training_lines(storage.jsonl):
        k = voice.norm_key(ln["input"]["text"])
        texts.add(k)
        pairs.add((k, voice.norm_key(ln["input"].get("previous") or "")))
    return texts, pairs


@job(DATA, cpu=4.0, memory=32768, timeout=40 * 60)
def select_lines(per_domain: int, seed: int, try_no: int) -> dict:
    import hashlib
    import random
    from collections import Counter

    storage.reload()
    convs = storage.jsonl(raw_key(try_no, "conversations.jsonl"))
    lines, meta, counts = voice.select(convs, per_domain, seed)
    body = "".join(ln.model_dump_json() + "\n" for ln in lines).encode()
    storage.put(sources.voice_key("lines.jsonl"), body)
    storage.put_jsonl(sources.voice_key("meta.jsonl"), meta)
    by_id = {m["line_id"]: m for m in meta}
    users = [ln for ln in lines if ln.input.speaker_kind == "player"]
    agents = [ln for ln in lines if ln.input.speaker_kind == "npc"]
    words = sorted(len(ln.input.text.split()) for ln in lines)
    summary = {
        "gen_id": sources.VOICE_ID,
        "try": try_no,
        "key": sources.voice_key("lines.jsonl"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "meta_key": sources.voice_key("meta.jsonl"),
        "raw_key": raw_key(try_no, "conversations.jsonl"),
        "conversations_written": len(convs),
        **counts,
        "lines": len(lines),
        "user_lines": len(users),
        "agent_lines": len(agents),
        "by_domain": dict(sorted(Counter(by_id[ln.line_id]["domain"] for ln in lines).items())),
        "by_domain_and_role": dict(
            sorted(
                Counter(f"{by_id[ln.line_id]['domain']}/{by_id[ln.line_id]['who']}" for ln in lines).items()
            )
        ),
        "asr_style_user_lines": sum("asr_style" in ln.slices for ln in users),
        "sarcasm_seed": sum("sarcasm_seed" in ln.slices for ln in lines),
        "lookalike_seed": sum("lookalike_seed" in ln.slices for ln in lines),
        "move_focus": dict(
            Counter(
                ln.generator.split(":")[1] for ln in lines if ln.generator.startswith("move_focus")
            ).most_common()
        ),
        "seed_kinds": dict(Counter(ln.generator for ln in lines if "_seed:" in ln.generator).most_common()),
        "generators": dict(Counter(ln.generator.split(":")[0] for ln in lines).most_common()),
        "agent_line_kinds": dict(Counter(ln.generator for ln in agents).most_common()),
        "without_previous": sum(ln.input.previous is None for ln in lines),
        "arcs": dict(Counter(by_id[ln.line_id]["arc"] for ln in users).most_common()),
        "words_median": words[len(words) // 2],
        "words_max": words[-1],
        "one_to_three_words": sum(w <= 3 for w in words),
        "distinct_texts": len({voice.norm_key(ln.input.text) for ln in lines}),
        "agent_names": dict(Counter(ln.input.speaker for ln in agents).most_common()),
    }
    texts, pairs = training_texts()
    hits = [ln for ln in lines if voice.norm_key(ln.input.text) in texts]
    summary["in_training_lines"] = {
        "texts": len(hits),
        "pairs": sum(
            (voice.norm_key(ln.input.text), voice.norm_key(ln.input.previous or "")) in pairs for ln in lines
        ),
        "of": len(lines),
        "training_texts": len(texts),
        "top": dict(Counter(voice.norm_key(ln.input.text) for ln in hits).most_common(25)),
    }
    storage.put_json(sources.voice_key("summary.json"), summary)
    rng = random.Random(seed)
    sample = []
    for domain in sorted(voice.DOMAINS):
        mine = [ln for ln in lines if by_id[ln.line_id]["domain"] == domain]
        for ln in rng.sample(mine, min(7, len(mine))):
            sample.append(
                {
                    "line_id": ln.line_id,
                    "domain": domain,
                    "generator": ln.generator,
                    "slices": ln.slices,
                    "previous": ln.input.previous,
                    "speaker": ln.input.speaker,
                    "text": ln.input.text,
                }
            )
    taken = sorted({m["prompt_id"] for m in meta})
    whole = []
    for pid in rng.sample(taken, min(4, len(taken))):
        c = next(x for x in convs if x["prompt_id"] == pid)
        rows = "\n".join(
            f"{c['agent'] if t['who'] == 'agent' else 'User'}{' (moment)' if t['marked'] else ''}: {t['text']}"
            for t in c["turns"]
        )
        whole.append(
            {
                "prompt_id": pid,
                "seed": c["seed"],
                "seed_kind": c["seed_kind"],
                "asr": c["asr"],
                "arc": c["arc"],
                "persona": c["persona"],
                "raw": rows,
            }
        )
    storage.put_json(raw_key(try_no, "read_sample.json"), {"sample": sample, "conversations": whole})
    return {"summary": summary, "sample": sample, "conversations": whole}


TEXT_FIELDS = ("top",)  # summary fields that hold line text: kept on the volume, never written to the repo


def no_text(summary: dict) -> dict:
    out = dict(summary)
    if "in_training_lines" in out:
        out["in_training_lines"] = {k: v for k, v in out["in_training_lines"].items() if k not in TEXT_FIELDS}
    return out


@app.local_entrypoint()
def generate(n_per_domain: int = N_PER_DOMAIN, seed: int = voice.SEED, try_no: int = 1):
    """One conversation per prompt, 33 per domain over 13 domains, from the teacher (Qwen3.8-27B in FP8 on vLLM, weights
    cached on the tone-reader-hf-cache volume) on one H100, spawned to keep going without this client. Written as JSON
    rounds under vLLM's structured output, so turns alternate and exactly one turn is the moment. The prompts, the raw
    output, the parsed turns and each prompt's choices go to conversations.jsonl under
    runs/tonetext/generated/<voice id>-raw/try<n>/, the run's counts beside it (result.json)."""
    call = write_conversations.spawn(n_per_domain, seed, try_no)
    save(
        "voice_generate",
        f"generate-spawned-try{try_no}.json",
        {
            "call": call.object_id,
            "gen_id": sources.VOICE_ID,
            "try": try_no,
            "n_per_domain": n_per_domain,
            "seed": seed,
            "result_key": raw_key(try_no, "result.json"),
        },
    )
    print("spawned", call.object_id)


@app.local_entrypoint()
def status(try_no: int = 1):
    """A try's counts, once its conversations are written."""
    if not storage.exists(raw_key(try_no, "result.json")):
        print("not yet")
        return
    res = json.loads(storage.get(raw_key(try_no, "result.json")))
    save("voice_generate", f"generate-result-try{try_no}.json", res)
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def select(per_domain: int = PER_DOMAIN, seed: int = voice.SEED, try_no: int = 1):
    """23 usable conversations per domain (those with a usable mark first: 8 sarcasm seeds, 8 look-alikes, 7 marked
    moves), three lines from each, to runs/tonetext/generated/<voice id>/lines.jsonl, each line's prompt choices to
    meta.jsonl beside it. Counts, and how many texts r1's training lines already hold, are saved here; the lines are
    private evaluation data, so the sample to read goes to the volume beside the raw output and is printed, never
    saved in the repo."""
    res = select_lines.remote(per_domain, seed, try_no)
    save("voice_generate", f"summary-{sources.VOICE_ID}-try{try_no}.json", no_text(res["summary"]))
    print(json.dumps({k: v for k, v in res["summary"].items() if k not in ("by_domain_and_role",)}, indent=1))
    for e in res["sample"]:
        asr = " asr" if "asr_style" in e["slices"] else ""
        print(f"[{e['domain']} {e['generator']}{asr}] {e['previous']}\n    -> {e['speaker']}: {e['text']}")
    for c in res["conversations"]:
        print(
            f"\n=== {c['prompt_id']} seed {c['seed']}/{c['seed_kind']} asr {c['asr']} arc {c['arc']} "
            f"| {c['persona']}\n{c['raw']}"
        )
