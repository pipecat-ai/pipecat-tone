"""r3's support-call set (``lab.reader.support``): calls written by the voice-agent set's teacher, both answers at each
call's moment taken, and from half the calls one more of the person's turns. Its train split trains r3; its dev and
test splits are read, never trained on. Each line's earlier turns are kept on the volume beside it (meta.jsonl).

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.support_generate::trial
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.support_generate::generate
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.support_generate::status
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.support_generate::select
"""

import json
import time

from lab.reader import sources, storage, support
from lab.reader.app import DATA, VLLM, app, hf_cache, job, save
from lab.reader.voice import TEACHER_REPO, TEACHER_REVISION

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 1600}
N_PER_DOMAIN = 420  # 14,700 calls over 35 domains, for 30,000 lines
SHARDS = 4  # one H100 each
MAX_LINES = 30_000


def raw_key(try_no: int, name: str) -> str:
    """A try's teacher output, beside the set."""
    return f"runs/tonetext/generated/{sources.SUPPORT_ID}-raw/try{try_no}/{name}"


@job(VLLM, gpu="H100", cpu=8.0, memory=65536, timeout=4 * 3600, volumes={"/cache": hf_cache})
def write_calls(n_per_domain: int, seed: int, try_no: int, shard: int = 0, shards: int = 1) -> dict:
    """Every `shards`-th prompt from `shard`, one call each, as JSON under vLLM's structured output."""
    from collections import Counter

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    t0 = time.perf_counter()
    prompts = support.build_prompts(n_per_domain, seed=seed)[shard::shards]
    llm = LLM(
        model=TEACHER_REPO,
        revision=TEACHER_REVISION,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        seed=seed + shard,
        max_num_seqs=256,
    )
    load_s = time.perf_counter() - t0
    params = SamplingParams(
        n=1,
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        max_tokens=SAMPLING["max_tokens"],
        seed=seed + shard,
        structured_outputs=StructuredOutputsParams(json=support.schema()),
    )
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
        turns, instead, probs = support.parse_call(text)
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
                "instead_text": instead,
                "problems": probs,
                "finish_reason": str(o.outputs[0].finish_reason),
            }
        )
    storage.put_jsonl(raw_key(try_no, f"calls-{shard}.jsonl"), rows)
    result = {
        "set": sources.SUPPORT_ID,
        "try": try_no,
        "shard": shard,
        "shards": shards,
        "prompts": len(prompts),
        "with_a_moment": sum(support.moment_lines(r) is not None for r in rows),
        "turns_median": sorted(len(r["turns"]) for r in rows)[len(rows) // 2] if rows else 0,
        "problems": dict(problems),
        "finish_reasons": dict(finish),
        "tokens": {"in": tok_in, "out": tok_out},
        "seconds": {"load": round(load_s), "generate": round(gen_s)},
        "teacher": f"{TEACHER_REPO}@{TEACHER_REVISION}",
        "sampling": {**SAMPLING, "seed": seed + shard},
        "key": raw_key(try_no, f"calls-{shard}.jsonl"),
    }
    storage.put_json(raw_key(try_no, f"result-{shard}.json"), result)
    return result


def evaluation_pairs():
    """(text, the line before's words) of every line the scorecard or a dev read takes: Tone's act and emotion, move
    and sarcasm sets' dev and test lines, r2's voice-agent dev and test sets, r1's voice-agent test set and
    Topical-Chat's test split."""
    from lab.reader.data import previous_words

    def rows(key: str, held_only: bool = True):
        for r in storage.jsonl(key):
            ln = r.get("line") or r
            if not held_only or ln.get("split") in ("dev", "test"):
                yield ln["input"]["text"], previous_words(ln["input"].get("previous"))

    for sid in sources.ACT_EMOTION_SETS:
        yield from rows(sources.labels_key(sid))
    yield from rows(sources.labels_key(sources.EXCHANGE_SET))
    for sid in (*sources.SARCASM_SETS["dev"], *sources.SARCASM_SETS["test"]):
        yield from rows(sources.sarcasm_key(sid), held_only=False)
    for sid in sources.R2_SETS.values():
        yield from rows(sources.tonetext_labels_key(sid))
    for sid in sources.VOICE_GENERATED.values():
        yield from rows(sources.tonetext_labels_key(sid), held_only=False)
    yield from rows(sources.open_key("topicalchat_test"), held_only=False)


@job(DATA, cpu=4.0, memory=32768, timeout=3600)
def select_lines(max_lines: int, try_no: int, shards: int) -> dict:
    import hashlib
    from collections import Counter

    storage.reload()
    calls = [c for k in range(shards) for c in storage.jsonl(raw_key(try_no, f"calls-{k}.jsonl"))]
    texts, pairs = support.bans(evaluation_pairs())
    ptexts, ppairs = support.probe_bans()
    keeper = support.Keeper(texts | ptexts, pairs | ppairs)
    lines, metas, counts = support.select(calls, keeper, max_lines)
    body = "".join(ln.model_dump_json() + "\n" for ln in lines).encode()
    storage.put(sources.support_key("lines.jsonl"), body)
    storage.put_jsonl(sources.support_key("meta.jsonl"), metas)
    by_id = {m["line_id"]: m for m in metas}

    def spread(rows, key):
        return dict(sorted(Counter(key(r) for r in rows).items()))

    words = sorted(len(ln.input.text.split()) for ln in lines)
    summary = {
        "set": sources.SUPPORT_ID,
        "try": try_no,
        "key": sources.support_key("lines.jsonl"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "meta_key": sources.support_key("meta.jsonl"),
        "calls_written": len(calls),
        **counts,
        "lines": len(lines),
        "banned": {
            "evaluation_texts": len(texts),
            "evaluation_pairs": len(pairs),
            "probes": len(support.PROBES),
        },
        "by_split": spread(lines, lambda ln: ln.split),
        "by_split_and_construction": spread(
            lines, lambda ln: f"{ln.split}/{by_id[ln.line_id]['sarcastic_by_construction']}"
        ),
        "by_moment_and_reply": spread(lines, lambda ln: ln.generator),
        "by_domain": spread(lines, lambda ln: by_id[ln.line_id]["domain"]),
        "held_domains": list(support.HELD_DOMAINS),
        "asr_style": sum("asr_style" in ln.slices for ln in lines),
        "words_median": words[len(words) // 2] if words else 0,
        "words_max": words[-1] if words else 0,
        "distinct_texts": len({support.norm_key(ln.input.text) for ln in lines}),
        "earlier_turns_median": sorted(len(m["before"]) for m in metas)[len(metas) // 2] if metas else 0,
    }
    storage.put_json(sources.support_key("summary.json"), summary)
    import random

    rng = random.Random(support.SEED)
    sample = [
        {
            "line_id": ln.line_id,
            "split": ln.split,
            "generator": ln.generator,
            "previous": ln.input.previous,
            "text": ln.input.text,
        }
        for ln in rng.sample(lines, min(80, len(lines)))
    ]
    storage.put_json(raw_key(try_no, "read_sample.json"), {"sample": sample})
    return {"summary": summary, "sample": sample}


@app.local_entrypoint()
def trial(n_per_domain: int = 2, seed: int = support.SEED, show: int = 12):
    """Two calls per domain under try 0, written now: their counts, and a few calls to read on the terminal (never saved
    in the repo)."""
    res = write_calls.remote(n_per_domain, seed + 1000, 0)
    save("support_generate", "trial-result.json", res)
    print(json.dumps(res, indent=1))
    rows = storage.jsonl(raw_key(0, "calls-0.jsonl"))
    for r in rows[:show]:
        pair = support.moment_lines(r)
        print(
            f"\n=== {r['prompt_id']} {r['moment']} main {r['main']} instead {r['instead']} asr {r['asr']} | {r['arc']}"
        )
        for t in r["turns"]:
            print(f"{'AGENT' if t['who'] == 'agent' else 'USER '}{' *' if t['marked'] else '  '} {t['text']}")
        print(f"INSTEAD  {r['instead_text']}")
        print("usable moment:", pair is not None)


@app.local_entrypoint()
def generate(
    n_per_domain: int = N_PER_DOMAIN, seed: int = support.SEED, try_no: int = 1, shards: int = SHARDS
):
    """Every prompt in `shards` spawned calls, one H100 each, left running without this client."""
    calls = [write_calls.spawn(n_per_domain, seed, try_no, k, shards) for k in range(shards)]
    save(
        "support_generate",
        f"generate-spawned-try{try_no}.json",
        {
            "calls": [c.object_id for c in calls],
            "set": sources.SUPPORT_ID,
            "try": try_no,
            "n_per_domain": n_per_domain,
            "seed": seed,
            "shards": shards,
            "results": [raw_key(try_no, f"result-{k}.json") for k in range(shards)],
        },
    )
    print("spawned", [c.object_id for c in calls])


@app.local_entrypoint()
def status(try_no: int = 1, shards: int = SHARDS):
    """Each shard's counts, once written."""
    out = []
    for k in range(shards):
        key = raw_key(try_no, f"result-{k}.json")
        out.append(json.loads(storage.get(key)) if storage.exists(key) else {"shard": k, "done": False})
    save("support_generate", f"generate-result-try{try_no}.json", {"shards": out})
    print(json.dumps(out, indent=1))


@app.local_entrypoint()
def select(max_lines: int = MAX_LINES, try_no: int = 1, shards: int = SHARDS):
    """The set's lines from every shard's calls, to runs/tonetext/generated/<set id>/lines.jsonl with each line's call
    and earlier turns in meta.jsonl; counts saved here, a sample printed (the lines are private data, never saved in the
    repo)."""
    res = select_lines.remote(max_lines, try_no, shards)
    save("support_generate", f"summary-{sources.SUPPORT_ID}-try{try_no}.json", res["summary"])
    print(json.dumps(res["summary"], indent=1))
    for e in res["sample"][:40]:
        print(f"[{e['split']} {e['generator']}] {e['previous']}\n    -> {e['text']}")
