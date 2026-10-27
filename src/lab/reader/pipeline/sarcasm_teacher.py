"""r1's sarcasm teachers, trained end to end on Tone's sarcasm labels, and their labels on every training line r1's
student reads.

    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.sarcasm_teacher::spawn_one --seed 0   # and 1, 2
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.sarcasm_teacher::collect
"""

from lab import ROOT
from lab.reader import sources, storage
from lab.reader.app import TORCH, app, job, save
from lab.reader.schemas import PLAYER

ENCODER = "ettin-1b"
BATCH = 32


def who(n: str | None) -> str:
    return "the player" if n == PLAYER else (n or "everyone")


def render(inp: dict) -> tuple[str, list[tuple[int, int]]]:
    """A line as the teachers read it, everyone present first as markers so truncation can only cut the line before:

        People: [Tomas] [Wren] [the player]. Mara to Tomas: "You still owe me ten coins." Before: Tomas: Lovely day.

    and, per target in the caller's order, the character span of its marker (brackets included). The addressee's
    marker always comes first, then the others in the caller's order."""
    targets = list(inp["targets"])
    to = inp.get("to")
    order = ([to] + [t for t in targets if t != to]) if to in targets else targets
    text, at = "People:", {}
    for t in order:
        start = len(text) + 1
        text += f" [{who(t)}]"
        at[t] = (start, len(text))
    spans = [at[t] for t in targets]
    text += f'. {who(inp["speaker"])} to {who(inp.get("to"))}: "{inp["text"]}"'
    if inp.get("previous"):
        prev = inp["previous"]
        name, sep, rest = prev.partition(":")
        text += f" Before: {who(name.strip()) if sep else ''}{':' if sep else ''}{rest if sep else prev}"
    return text, spans


def sarcasm_rows() -> list[dict]:
    """Every sarcasm-labelled line, train, dev and test, with the set it came from."""
    rows = []
    for split, ids in sources.SARCASM_SETS.items():
        for sid in ids:
            for r in storage.jsonl(sources.sarcasm_key(sid)):
                if r["line"]["split"] != split:
                    raise SystemExit(f"{sid}: a {r['line']['split']} line in a {split} set")
                rows.append({**r, "set": sid})
    return rows


def examples(rows: list[dict]) -> list[dict]:
    """Labelled rows ({"line", "sarcasm": {"label", "agree", ...}, "set"}) as items in ``render``'s form, names
    included: the teachers are never shipped and the student never sees what they read. A line without a majority keeps
    y -100 (masked)."""
    out = []
    for r in rows:
        ln, s = r["line"], r["sarcasm"]
        text, _ = render(ln["input"])
        lab = s.get("label")
        out.append(
            {
                "line_id": ln["line_id"],
                "text": text,
                "y": -100 if lab is None else int(bool(lab)),
                "w": round((s.get("agree") or 3) / 3, 4) if lab is not None else 0.0,
                "split": ln["split"],
                "slices": ln.get("slices") or [],
                "source": ln["source"],
                "generator": ln.get("generator"),
                "speaker_kind": ln["input"]["speaker_kind"],
                "set": r.get("set", ""),
            }
        )
    return out


def lines_to_label() -> tuple[list[dict], dict]:
    """The clean training lines the student reads, each once, without the sarcasm set (``data.r1_training_lines``),
    and how many came from each place; dev and test lines are never labelled."""
    from collections import Counter

    from lab.reader import data

    chosen = list(data.clean_once(data.r1_training_lines(storage.jsonl, sarcasm=False)))
    return [ln for _, ln in chosen], dict(Counter(where for where, _ in chosen))


def read_scores(exs: list[dict], p: list[float]) -> dict:
    """The scorecard's sarcasm reads: the threshold (the lowest at which at most 2% of sincere natural dev lines are called sarcastic),
    then F1 on the held-out sarcasm and chain tests, sincere lines called sarcastic on the natural test, the chain test
    alone, and calibration on test."""
    from lab.reader.calibration import ece
    from lab.reader.scoring import sarcasm_scores, threshold_on

    def pick(cond):
        idx = [i for i, e in enumerate(exs) if e["y"] >= 0 and cond(e)]
        return [p[i] for i in idx], [exs[i]["y"] for i in idx]

    tau = threshold_on(*pick(lambda e: e["set"] == sources.NATURAL_SARCASM[0]))
    hard = lambda e: e["set"] in sources.HARD_SARCASM_TEST  # noqa: E731
    natural = lambda e: e["set"] == sources.NATURAL_SARCASM[1]  # noqa: E731
    chain = lambda e: e["set"] == sources.CHAIN_TEST  # noqa: E731
    test = lambda e: e["split"] == "test"  # noqa: E731
    return {
        "tau": tau,
        "at_tau": {
            "hard_test": sarcasm_scores(*pick(hard), tau),
            "natural_test": sarcasm_scores(*pick(natural), tau),
            "chain_test": sarcasm_scores(*pick(chain), tau),
        },
        "at_half": {
            "hard_test": sarcasm_scores(*pick(hard), 0.5),
            "natural_test": sarcasm_scores(*pick(natural), 0.5),
        },
        "calibration_test_ece": ece(*pick(test))["ece"],
    }


@job(TORCH, gpu="H100", cpu=8.0, memory=65536, timeout=24 * 3600)
def teacher(seed: int, encoder: str = ENCODER, epochs: int = 8) -> dict:
    """One teacher on one H100: Tone's attention head with two classes on the encoder, class weights, the committee's
    agreement as each line's weight, the best dev epoch kept (F1 of the sarcastic class at 0.5 on the three sarcasm dev
    sets). Scored by `read_scores`, it labels every line `lines_to_label` gives under the sarcasm soft id, its result to
    the run's folder; its weights are not kept."""
    import math
    import random
    import time
    from collections import Counter

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from lab.reader import model
    from lab.reader.data import MAX_LEN
    from lab.reader.scoring import f1_sarcastic

    storage.reload()
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = model.ENCODERS[encoder]
    lr = model.TEACHER_LR.get(encoder, model.DEFAULT_TEACHER_LR)
    rows = sarcasm_rows()
    exs = examples(rows)
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"])
    enc = AutoModel.from_pretrained(spec["repo"], revision=spec["rev"]).to("cuda")
    head = model.make_head(enc.config.hidden_size, 2).to("cuda")
    tr = [i for i, e in enumerate(exs) if e["split"] == "train" and e["y"] >= 0]
    dv = [i for i, e in enumerate(exs) if e["split"] == "dev" and e["y"] >= 0]
    cw = torch.tensor(model.class_weights([exs[i]["y"] for i in tr], 2), dtype=torch.float32, device="cuda")
    opt = torch.optim.AdamW(
        [{"params": enc.parameters(), "lr": lr}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01
    )
    steps = epochs * math.ceil(len(tr) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, model.warmup_linear(steps))
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def run(texts):
        b = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
        h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        return head(h.float(), b["attention_mask"])

    def probs(texts, bs=128) -> list[float]:
        enc.eval()
        head.eval()
        out = []
        with torch.inference_mode(), amp:
            for i in range(0, len(texts), bs):
                out.append(torch.softmax(run(texts[i : i + bs]).float(), -1)[:, 1].cpu())
        return torch.cat(out).tolist() if out else []

    best, history = None, []
    gold_dev = [exs[i]["y"] for i in dv]
    for ep in range(epochs):
        enc.train()
        head.train()
        order = tr[:]
        random.shuffle(order)
        tot = 0.0
        for b in range(0, len(order), BATCH):
            idx = order[b : b + BATCH]
            with amp:
                logits = run([exs[i]["text"] for i in idx])
            y = torch.tensor([exs[i]["y"] for i in idx], device="cuda")
            w = torch.tensor([exs[i]["w"] for i in idx], dtype=torch.float32, device="cuda")
            per = torch.nn.functional.cross_entropy(
                logits.float(), y, weight=cw, reduction="none", label_smoothing=0.1
            )
            loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        p = probs([exs[i]["text"] for i in dv])
        f1 = f1_sarcastic(gold_dev, [int(x >= 0.5) for x in p])
        history.append(
            {
                "epoch": ep + 1,
                "loss": round(tot, 2),
                "dev_f1": round(f1, 4),
                "seconds": round(time.time() - t0),
            }
        )
        print(history[-1], flush=True)
        if best is None or f1 > best[0]:
            best = (
                f1,
                {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                ep + 1,
            )
    enc.load_state_dict(best[1])
    head.load_state_dict(best[2])
    p_all = probs([e["text"] for e in exs])
    sc = read_scores(exs, p_all)
    lines, counts = lines_to_label()
    p_pool = probs([render(ln["input"])[0] for ln in lines], bs=256)
    storage.put_jsonl(
        sources.sarcasm_soft_key(f"teacher-{encoder}-seed{seed}"),
        ({"line_id": ln["line_id"], "p": round(float(x), 5)} for ln, x in zip(lines, p_pool)),
    )
    rid = sources.run_id("r1", "sarcasm-teacher", encoder, seed)
    res = {
        "run_id": rid,
        "encoder": encoder,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch": BATCH,
        "best_epoch": best[3],
        "dev_f1_at_half": round(best[0], 4),
        "history": history,
        "scores": sc,
        "train_lines": len(tr),
        "dev_lines": len(dv),
        "labelled_lines": len(lines),
        "labelled_by_source": counts,
        "labelled_called_sarcastic_at_tau": int(sum(1 for x in p_pool if x >= sc["tau"])),
        "labelled_mean_p": round(float(np.mean(p_pool)), 4) if p_pool else None,
        "sets": dict(Counter(e["set"] for e in exs)),
        "seconds": round(time.time() - t0),
        # dev and test probabilities by set and line, so `collect` can score the teachers' mean
        "eval_probs": {
            f"{e['set']}|{e['line_id']}": round(float(x), 5)
            for e, x in zip(exs, p_all)
            if e["split"] != "train"
        },
    }
    storage.put_json(sources.run_key(rid, "result.json"), res)
    return {k: v for k, v in res.items() if k != "eval_probs"}


@app.local_entrypoint()
def spawn_one(seed: int, encoder: str = ENCODER, epochs: int = 8):
    """One teacher, spawned and left running: launch with `modal run --detach`, one app per seed."""
    call = teacher.spawn(seed, encoder, epochs)
    print("spawned", sources.run_id("r1", "sarcasm-teacher", encoder, seed), "call", call.object_id)


@app.local_entrypoint()
def collect(seeds: str = "0,1,2", encoder: str = ENCODER, wait_h: float = 4.0):
    """The teachers' results, waiting for them, and their mean, which the student learns from, scored by the same
    rules; saved to results/reader/sarcasm_teacher/results-teachers.json."""
    rids = [sources.run_id("r1", "sarcasm-teacher", encoder, int(s)) for s in seeds.split(",") if s]
    results = storage.wait_for([sources.run_key(rid, "result.json") for rid in rids], wait_h)
    keys = sorted(set.intersection(*(set(r["eval_probs"]) for r in results)))
    by_key = {f"{r['set']}|{r['line']['line_id']}": r for r in sarcasm_rows()}
    exs = examples([by_key[k] for k in keys])
    mean_p = [sum(r["eval_probs"][k] for r in results) / len(results) for k in keys]
    mean_scores = read_scores(exs, mean_p)
    out = {
        "teachers": [{k: v for k, v in r.items() if k != "eval_probs"} for r in results],
        "mean_of_teachers": mean_scores,
        "eval_lines": len(keys),
    }
    path = save("sarcasm_teacher", "results-teachers.json", out)
    for r in [
        *out["teachers"],
        {"run_id": "mean of teachers", "scores": mean_scores, "best_epoch": "-", "dev_f1_at_half": "-"},
    ]:
        t = r["scores"]
        print(
            f"{r['run_id']:44s} best epoch {r['best_epoch']} dev F1 {r['dev_f1_at_half']} | tau {t['tau']} | "
            f"test F1 {t['at_tau']['hard_test']['f1']} {t['at_tau']['hard_test']['f1_ci']} | "
            f"false sarcasm {t['at_tau']['natural_test']['false_sarcasm_rate']} | chain F1 "
            f"{t['at_tau']['chain_test']['f1']} | ECE {t['calibration_test_ece']}"
        )
    print("wrote", path.relative_to(ROOT))
