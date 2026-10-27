"""The reader's student: an encoder and five per-line heads, trained on hard and soft labels at once, and the reads
that choose and report it."""

import math
import random
import time

import numpy as np

from lab.reader import scoring
from lab.reader.data import MAX_LEN, Example, Lines, eval_groups, render
from lab.reader.schemas import EMOTIONS, INTENSITIES, MOVES

# The encoders, pinned to a revision so a name always means the same weights.
ENCODERS = {
    "minilm-l6": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "rev": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    },
    "ettin-1b": {"repo": "jhu-clsp/ettin-encoder-1b", "rev": "befd76be43d08b89ff9957012f3ff29d0842780b"},
}
# Large models learn best with a smaller step.
TEACHER_LR = {"ettin-1b": 1.5e-5}
DEFAULT_TEACHER_LR = 2e-5
OUTPUTS = ("emotion", "emotion_intensity", "backchannel", "move", "sarcasm")
HEAD_QUERIES = 4


def make_head(d: int, n_out: int):
    """Tone's attention head over a line's token states: HEAD_QUERIES learned queries attending over the tokens,
    joined with the mean of the tokens, then a small two-layer network over `n_out` classes."""
    import torch

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.randn(HEAD_QUERIES, d) * (1.0 / math.sqrt(d)))
            self.drop = torch.nn.Dropout(0.1)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear((HEAD_QUERIES + 1) * d, d),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d, n_out),
            )

        def forward(self, h, mask):
            """h: [batch, tokens, d]; mask: [batch, tokens] with 1 for real tokens."""
            m = mask.unsqueeze(-1).to(h.dtype)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            scores = torch.einsum("qd,bsd->bqs", self.q, h) / math.sqrt(h.shape[-1])
            scores = scores.masked_fill(mask[:, None, :] == 0, torch.finfo(scores.dtype).min)
            att = torch.softmax(scores, -1)
            attended = torch.einsum("bqs,bsd->bqd", att, h).flatten(1)
            return self.mlp(self.drop(torch.cat([attended, mean], -1)))

    return Head()


def Model(repo: str, revision: str, attn: str | None = None):
    """The encoder and five heads, one pass per line over the rendered input: emotion, intensity and backchannel are
    linear heads on the mean of the tokens; the move and sarcasm use `make_head`, so they can weigh the line against the
    one before. Built lazily so the module imports without torch; `attn="eager"` for export, so the traced graph does
    not bake in the export input's shape."""
    import torch
    from transformers import AutoModel

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            kw = {"attn_implementation": attn} if attn else {}
            self.enc = AutoModel.from_pretrained(repo, revision=revision, **kw)
            d = self.enc.config.hidden_size
            self.emo = torch.nn.Linear(d, len(EMOTIONS))
            self.emo_int = torch.nn.Linear(d, len(INTENSITIES))
            self.back = torch.nn.Linear(d, 2)
            self.move = make_head(d, len(MOVES))
            self.sarc = make_head(d, 2)
            self.drop = torch.nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            hp = self.drop((h * m).sum(1) / m.sum(1).clamp(min=1))
            return (
                self.emo(hp),
                self.emo_int(hp),
                self.back(hp),
                self.move(h, attention_mask),
                self.sarc(h, attention_mask),
            )

    return _M()


def text_of(e: Example, keep_previous: bool = True) -> str:
    return render(e.text, e.previous if keep_previous else None)


def collate(tok, batch: list[Example], device, keep: list[bool] | None = None) -> dict:
    """Tensors for a batch. `keep[i]` False drops line i's previous line (training's dropout, or the no-previous read)."""
    import torch

    keep = keep or [True] * len(batch)
    enc = tok(
        [text_of(e, k) for e, k in zip(batch, keep)],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    to = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
    f32 = torch.float32
    return {
        "enc": {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")},
        "emo": to([e.emo for e in batch]),
        "emo_int": to([e.emo_int for e in batch]),
        "back": to([e.back for e in batch]),
        "move": to([e.move for e in batch]),
        "move_w": to([e.move_w for e in batch], f32),
        "sarc": to([e.sarc for e in batch]),
        "sarc_w": to([e.sarc_w for e in batch], f32),
        "has_s_emo": to([e.s_emo is not None for e in batch], torch.bool),
        "s_emo": to([e.s_emo or [0.0] * len(EMOTIONS) for e in batch], f32),
        "s_emo_int": to([e.s_emo_int or [0.0] * len(INTENSITIES) for e in batch], f32),
        "s_back": to([e.s_back or 0.0 for e in batch], f32),
        "has_s_move": to([e.s_move is not None for e in batch], torch.bool),
        "s_move": to([e.s_move or [0.0] * len(MOVES) for e in batch], f32),
        "has_s_sarc": to([e.s_sarc is not None for e in batch], torch.bool),
        "s_sarc": to([e.s_sarc or 0.0 for e in batch], f32),
    }


def _soft_ce(logits, probs, weights=None):
    import torch

    per = -(probs * torch.log_softmax(logits.float(), -1)).sum(-1)
    if weights is None:
        return per.mean()
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


def _hard_ce(logits, y, class_w, line_w, smoothing):
    import torch

    ok = y >= 0
    per = torch.nn.functional.cross_entropy(
        logits.float(), y, weight=class_w, ignore_index=-100, reduction="none", label_smoothing=smoothing
    )
    return (per[ok] * line_w[ok]).sum() / line_w[ok].sum().clamp(min=1e-6)


def batch_loss(b: dict, out, w: dict, smoothing: float):
    """Each target's loss, averaged over the lines that carry it. Hard labels: cross-entropy with `class_weights` and
    label smoothing, the committee's agreement weighing the move's and sarcasm's lines, intensity and backchannel at
    half. Soft labels: cross-entropy against the teachers' mean, intensity weighted by the teachers' chance the emotion
    is not neutral and at half, backchannel at half."""
    import torch

    ce = torch.nn.functional.cross_entropy
    le, lei, lb, lm, ls = (x.float() for x in out)
    loss = torch.zeros((), device=le.device)
    if (b["emo"] >= 0).any():
        loss = loss + ce(le, b["emo"], weight=w["emo"], ignore_index=-100, label_smoothing=smoothing)
    if (b["emo_int"] >= 0).any():
        loss = loss + 0.5 * ce(lei, b["emo_int"], ignore_index=-100)
    if (b["back"] >= 0).any():
        loss = loss + 0.5 * ce(lb, b["back"], weight=w["back"], ignore_index=-100)
    if (b["move"] >= 0).any():
        loss = loss + _hard_ce(lm, b["move"], w["move"], b["move_w"], smoothing)
    if (b["sarc"] >= 0).any():
        loss = loss + _hard_ce(ls, b["sarc"], w["sarc"], b["sarc_w"], smoothing)
    se, sm, ss = b["has_s_emo"], b["has_s_move"], b["has_s_sarc"]
    if se.any():
        loss = loss + _soft_ce(le[se], b["s_emo"][se])
        not_neutral = 1 - b["s_emo"][se][:, EMOTIONS.index("neutral")]
        loss = loss + 0.5 * _soft_ce(lei[se], b["s_emo_int"][se], not_neutral)
        pb = b["s_back"][se]
        loss = loss + 0.5 * _soft_ce(lb[se], torch.stack([1 - pb, pb], -1))
    if sm.any():
        loss = loss + _soft_ce(lm[sm], b["s_move"][sm])
    if ss.any():
        ps = b["s_sarc"][ss]
        loss = loss + _soft_ce(ls[ss], torch.stack([1 - ps, ps], -1))
    return loss


def class_weights(labels: list[int], n: int, cap: float = 8.0) -> list[float]:
    """Rare classes up-weighted: the square root of inverse frequency, capped."""
    counts = [0] * n
    for y in labels:
        if y >= 0:
            counts[y] += 1
    total = sum(counts)
    return [min(cap, math.sqrt(total / (n * c))) if c else cap for c in counts]


def predict(
    model,
    tok,
    exs: list[Example],
    device,
    keep_previous: bool = True,
    batch_size: int = 256,
    bf16: bool = True,
    logits: bool = False,
) -> dict:
    """Per line: probabilities (or, with `logits`, raw logits) for every output."""
    import torch

    model.eval()
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (device == "cuda" and bf16)
        else torch.autocast("cpu", enabled=False)
    )
    acc = {k: [] for k in OUTPUTS}
    with torch.inference_mode(), amp:
        for i in range(0, len(exs), batch_size):
            chunk = exs[i : i + batch_size]
            b = collate(tok, chunk, device, [keep_previous] * len(chunk))
            out = model(b["enc"]["input_ids"], b["enc"]["attention_mask"])
            for k, z in zip(OUTPUTS, out):
                z = z.float()
                acc[k].append((z if logits else torch.softmax(z, -1)).cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in acc.items()}


def reads(exs: list[Example], P: dict, groups: dict[str, list[int]]) -> dict:
    """Macro-F1 for emotion and the move, F1 of the sarcastic class at 0.5 and the share of sincere lines called sarcastic,
    on each named group of line indices (`data.eval_groups`)."""
    out = {}
    for name, idx in groups.items():
        if not idx:
            continue
        if "emotion" in name:
            g = [EMOTIONS[exs[i].emo] for i in idx]
            p = [EMOTIONS[int(P["emotion"][i].argmax())] for i in idx]
            out[name] = {
                "lines": len(idx),
                "macro_f1": round(scoring.macro_f1(g, p, EMOTIONS), 4),
                "macro_f1_30": round(scoring.macro_f1_floor(g, p, EMOTIONS), 4),
            }
        elif "move" in name:
            g = [MOVES[exs[i].move] for i in idx]
            p = [MOVES[int(P["move"][i].argmax())] for i in idx]
            out[name] = {
                "lines": len(idx),
                "macro_f1": round(scoring.macro_f1(g, p, MOVES), 4),
                "macro_f1_30": round(scoring.macro_f1_floor(g, p, MOVES), 4),
            }
        else:
            g = [exs[i].sarc for i in idx]
            p = [int(P["sarcasm"][i][1] >= 0.5) for i in idx]
            neg = [x for x, y in zip(p, g) if y == 0]
            out[name] = {
                "lines": len(idx),
                "sarcastic": sum(g),
                "f1": round(scoring.f1_sarcastic(g, p), 4),
                "false_sarcasm_rate": round(sum(neg) / len(neg), 4) if neg else None,
            }
    return out


def selection_score(r: dict) -> float:
    """r1's: the mean of dev emotion macro-F1, dev move macro-F1 and dev sarcasm F1 at 0.5."""
    return round((r["emotion"]["macro_f1"] + r["move"]["macro_f1"] + r["sarcasm"]["f1"]) / 3, 4)


def selection_score_r2(r: dict) -> float:
    """r2's: Tone's dev sets and the voice-agent dev set weigh the same. Half the mean of Tone's dev emotion, move and
    sarcasm F1, half the mean of the voice-agent dev set's emotion and move macro-F1 over its classes with FLOOR lines
    or more (its few sarcastic turns are read, not chosen on)."""
    tone = (r["emotion"]["macro_f1"] + r["move"]["macro_f1"] + r["sarcasm"]["f1"]) / 3
    if "voice_emotion" not in r or "voice_move" not in r:  # no voice-agent dev lines: Tone's dev sets alone
        return round(tone, 4)
    voice = (r["voice_emotion"]["macro_f1_30"] + r["voice_move"]["macro_f1_30"]) / 2
    return round((tone + voice) / 2, 4)


def warmup_linear(steps: int):
    """The learning rate's factor at each of `steps` steps: a linear warm-up over the first tenth, times a linear decay
    to zero at the end."""
    return lambda s: min(1.0, s / max(1, steps // 10)) * max(0.0, (steps - s) / steps)


def train(
    train_exs: list[Example],
    dev_exs: list[Example],
    repo: str,
    revision: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    head_lr: float,
    seed: int,
    smoothing: float,
    drop_previous: float,
    select=None,
    groups_fn=None,
) -> tuple[object, object, dict]:
    """The student trained on `train_exs`, keeping the epoch with the best dev selection score. Each epoch drops the
    line before from a `drop_previous` share of the lines that have one, drawn afresh. `select` scores an epoch from its
    dev reads (default `selection_score`); `groups_fn(exs, split)` names the dev reads (default `data.eval_groups`)."""
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = Model(repo, revision).to(device)
    w = {
        "emo": class_weights([e.emo for e in train_exs], len(EMOTIONS)),
        "back": class_weights([e.back for e in train_exs], 2),
        "move": class_weights([e.move for e in train_exs], len(MOVES)),
        "sarc": class_weights([e.sarc for e in train_exs], 2),
    }
    w = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in w.items()}
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr}, {"params": head_params, "lr": head_lr}], weight_decay=0.01
    )
    steps = epochs * math.ceil(len(train_exs) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_linear(steps))
    groups = (groups_fn or eval_groups)(dev_exs, "dev")
    select = select or selection_score
    best, history, t0 = None, [], time.time()
    order = list(range(len(train_exs)))
    for ep in range(epochs):
        model.train()
        rng = random.Random(f"tonetext-{seed}-{ep}")
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_exs[j] for j in order[i : i + batch_size]]
            keep = [not (e.previous and rng.random() < drop_previous) for e in chunk]
            b = collate(tok, chunk, device, keep)
            with amp:
                out = model(b["enc"]["input_ids"], b["enc"]["attention_mask"])
            loss = batch_loss(b, out, w, smoothing)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        r = reads(dev_exs, predict(model, tok, dev_exs, device), groups)
        sel = select(r)
        history.append(
            {
                "epoch": ep + 1,
                "loss": round(tot / max(1, n), 4),
                "dev": r,
                "selection": sel,
                "seconds": round(time.time() - t0),
            }
        )
        print({k: history[-1][k] for k in ("epoch", "loss", "selection", "seconds")}, flush=True)
        if best is None or sel > best[0]:
            best = (sel, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    return (
        model,
        tok,
        {
            "history": history,
            "best_epoch": best[2],
            "dev_selection": best[0],
            "train_lines": len(train_exs),
            "dev_lines": len(dev_exs),
            "device": device,
            "steps": steps,
        },
    )


def fit(
    lines: Lines,
    encoder: str,
    seed: int,
    limit: int,
    *,
    labelled_only: bool = False,
    dev_without_previous: bool = True,
    select=None,
    groups_fn=None,
    **options,
) -> tuple:
    """`train` with `encoder` on `lines`' train split, then the reads of test and dev with and without the line before
    (dev's without it only with `dev_without_previous`). `labelled_only` keeps the training lines with a committee label
    for emotion, the move or sarcasm; `limit` shuffles the training lines and cuts them to that many. Returns the
    model, its tokenizer, its training info and the reads."""
    groups_fn = groups_fn or eval_groups
    train_exs = lines.split("train")
    if labelled_only:
        train_exs = [e for e in train_exs if e.emo >= 0 or e.move >= 0 or e.sarc >= 0]
    if limit:
        random.Random(seed).shuffle(train_exs)
        train_exs = train_exs[:limit]
    dev_exs, test_exs = lines.split("dev"), lines.split("test")
    spec = ENCODERS[encoder]
    model, tok, info = train(
        train_exs,
        dev_exs,
        spec["repo"],
        spec["rev"],
        seed=seed,
        select=select,
        groups_fn=groups_fn,
        **options,
    )

    def read(exs: list[Example], groups: dict, keep_previous: bool) -> dict:
        return reads(exs, predict(model, tok, exs, info["device"], keep_previous), groups)

    tg, dg = groups_fn(test_exs, "test"), groups_fn(dev_exs, "dev")
    out = {
        "test": {"with_previous": read(test_exs, tg, True), "without_previous": read(test_exs, tg, False)},
        "dev": {"with_previous": read(dev_exs, dg, True)},
    }
    if dev_without_previous:
        out["dev"]["without_previous"] = read(dev_exs, dg, False)
    return model, tok, info, out
