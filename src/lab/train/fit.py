"""Trains the tagger and exports it: an encoder and six heads, the best epoch on dev, ONNX out.

Sentence heads (delivery, sounds before and after, breath) read the first token and the mean of the
sentence's tokens; token heads (slow spans, stress) read each sentence token. Classes a row's label
space masks get no probability, as at run time, and the breath head learns only where breath is offered.
Breath has a head of its own as the corpus draws which replies breathe: its targets are noisy by design, and
shared with the other sounds they cost them. What ships is scored again through the runtime tagger.
"""

import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from lab.corpus.rows import Row
from lab.train.data import IGNORE, Example, make_example
from lab.train.evaluate import predictor, score
from pipecat_tone.tagger import ModelTagger, SentenceLabels, ToneTaggerModel
from pipecat_tone.tagger.features import COMPACT, Layout
from pipecat_tone.tagger.markup import keep
from pipecat_tone.tagger.model import HEADS, INPUTS, decode, delivery_classes, sound_classes


class TaggerNet(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.delivery = nn.Linear(2 * h, len(delivery_classes()))
        self.sound_before = nn.Linear(2 * h, len(sound_classes()))
        self.sound_after = nn.Linear(2 * h, len(sound_classes()))
        self.spans = nn.Linear(h, 3)
        self.emphasis = nn.Linear(h, 3)
        self.breath = nn.Linear(2 * h, 1)

    def forward(self, input_ids, attention_mask, sentence_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = sentence_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        pooled = self.dropout(torch.cat([hidden[:, 0], mean], dim=-1))
        tokens = self.dropout(hidden)
        return (
            self.delivery(pooled),
            self.sound_before(pooled),
            self.sound_after(pooled),
            self.spans(tokens),
            self.emphasis(tokens),
            self.breath(pooled),
        )


def fit(
    train_rows: list[Row],
    dev_rows: list[Row],
    backbone: str,
    out: Path,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    layout: Layout = COMPACT,
    seed: int = 0,
    init_seed: int | None = None,
    delivery_balance: bool = False,
) -> dict:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(backbone).backend_tokenizer
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    started = time.time()
    train_ex = [make_example(r, tokenizer, layout) for r in train_rows]
    dev_ex = [make_example(r, tokenizer, layout) for r in dev_rows]
    print(
        f"encoded {len(train_ex)} train and {len(dev_ex)} dev rows in {time.time() - started:.0f}s",
        flush=True,
    )

    # Seeds that share ``init_seed`` start from the same heads and differ only in data order and dropout, so their
    # weights can be averaged.
    torch.manual_seed(seed if init_seed is None else init_seed)
    model = TaggerNet(backbone).to(device)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train_ex) / batch_size)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    delivery_weight = torch.tensor(_delivery_weights(train_ex, delivery_balance), device=device)
    sound_weight = torch.tensor([0.2] + [1.0] * (len(sound_classes()) - 1), device=device)
    token_weight = torch.tensor([0.3, 1.0, 1.0], device=device)
    # A breath counts as a sound does against none.
    breath_weight = torch.tensor(1.0 / sound_weight[0].item(), device=device)

    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ex)
        total = 0.0
        for i in range(0, len(train_ex), batch_size):
            batch = _collate(train_ex[i : i + batch_size], pad_id, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                d, sb, sa, sp, em, br = model(*(batch[name] for name in INPUTS))
            d = d.float().masked_fill(batch["delivery_mask"] == 0, -1e4)
            sb = sb.float().masked_fill(batch["sound_mask"] == 0, -1e4)
            sa = sa.float().masked_fill(batch["sound_mask"] == 0, -1e4)
            loss = (
                nn.functional.cross_entropy(d, batch["delivery"], weight=delivery_weight)
                + 0.5 * nn.functional.cross_entropy(sb, batch["sound_before"], weight=sound_weight)
                + 0.5 * nn.functional.cross_entropy(sa, batch["sound_after"], weight=sound_weight)
                + _token_loss(sp.float(), batch["spans"], token_weight)
                + 0.5 * _token_loss(em.float(), batch["emphasis"], token_weight)
                + 0.5
                * _breath_loss(
                    br.float().squeeze(-1), batch["breath"], batch["breath_offered"], breath_weight
                )
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item()
        scores = score(dev_rows, _predictions(model, dev_ex, dev_rows, pad_id, device))
        history.append({"epoch": epoch, "loss": total / math.ceil(len(train_ex) / batch_size), "dev": scores})
        print(f"epoch {epoch}: loss {history[-1]['loss']:.4f} dev {json.dumps(scores)}", flush=True)
        if best is None or scores["combined"] > best[0]:
            best = (
                scores["combined"],
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )

    model.load_state_dict(best[2])
    model.to("cpu").eval()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.no_truncation()
    tokenizer.save(str(out / "tokenizer.json"))
    _export(model, out, pad_id)
    config = {
        "backbone": backbone,
        "epoch": best[1],
        "layout": asdict(layout),
        "deliveries": delivery_classes(),
        "sounds": sound_classes(),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "history": history,
    }

    # Score what ships through the runtime tagger: int8, and fp32 to see what quantizing cost.
    shipped = {}
    for name in ("model.int8.onnx", "model.onnx"):
        (out / "config.json").write_text(json.dumps({**config, "onnx": name}, indent=2))
        shipped[name] = score(dev_rows, predictor(ModelTagger(ToneTaggerModel(out))))
    close = shipped["model.int8.onnx"]["combined"] >= shipped["model.onnx"]["combined"] - 0.02
    config.update(onnx="model.int8.onnx" if close else "model.onnx", shipped_dev=shipped)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(
        f"exported epoch {best[1]}; dev through the runtime: {json.dumps(shipped)}; shipping {config['onnx']}",
        flush=True,
    )
    return config


def _token_loss(logits, targets, weight):
    # A batch with no stress targets (no TTS that stresses) would give NaN.
    if (targets != IGNORE).sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), weight=weight, ignore_index=IGNORE
    )


def _delivery_weights(examples: list[Example], balance: bool) -> list[float]:
    """None counts 0.3 of a delivery. Balanced, each delivery counts by the square root of how much rarer it is than
    the average one, within a quarter and four times, so rare deliveries are learned more."""
    weights = [0.3] + [1.0] * (len(delivery_classes()) - 1)
    if not balance:
        return weights
    counts = Counter(e.delivery for e in examples if e.delivery)
    mean = sum(counts.values()) / max(len(counts), 1)
    return [0.3] + [
        min(max((mean / counts[c]) ** 0.5, 0.25), 4.0) if counts[c] else 1.0 for c in range(1, len(weights))
    ]


def _breath_loss(logits, targets, offered, weight):
    # Only where breath is offered: elsewhere no breath can be chosen.
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=weight, reduction="none"
    )
    return (loss * offered).sum() / offered.sum().clamp(min=1)


def _collate(examples: list[Example], pad_id: int, device: str) -> dict:
    width = max(len(e.encoded.input_ids) for e in examples)

    def pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (width - len(values))

    batch = {
        "input_ids": [pad(e.encoded.input_ids, pad_id) for e in examples],
        "attention_mask": [pad(e.encoded.attention_mask, 0) for e in examples],
        "sentence_mask": [pad(e.encoded.sentence_mask, 0) for e in examples],
        "spans": [pad(e.spans, IGNORE) for e in examples],
        "emphasis": [pad(e.emphasis, IGNORE) for e in examples],
        "delivery": [e.delivery for e in examples],
        "sound_before": [e.sound_before for e in examples],
        "sound_after": [e.sound_after for e in examples],
        "delivery_mask": [e.delivery_mask for e in examples],
        "sound_mask": [e.sound_mask for e in examples],
        "breath": [e.breath for e in examples],
        "breath_offered": [e.breath_offered for e in examples],
    }
    return {k: torch.tensor(v, device=device) for k, v in batch.items()}


@torch.no_grad()
def _predictions(
    model: TaggerNet, examples: list[Example], rows: list[Row], pad_id: int, device: str
) -> Callable[[Row], SentenceLabels]:
    """The torch model's labels for rows, decoded as the runtime decodes them."""
    model.eval()
    predictions = {}
    for i in range(0, len(examples), 64):
        chunk = examples[i : i + 64]
        batch = _collate(chunk, pad_id, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs = model(*(batch[name] for name in INPUTS))
        heads = [o.float().cpu().numpy() for o in outputs]
        for j, (example, row) in enumerate(zip(chunk, rows[i : i + 64])):
            labels = decode([h[j] for h in heads], example.encoded.offsets, row.sentence, row.space)
            predictions[id(row)] = keep(row.sentence, labels, row.space)
    model.train()
    return lambda row: predictions[id(row)]


def _export(model: TaggerNet, out: Path, pad_id: int):
    model.encoder.config._attn_implementation = "eager"
    ids = torch.tensor([[pad_id + 1] * 24], dtype=torch.long)
    sentence = torch.zeros_like(ids)
    sentence[0, 12:] = 1
    # Token heads, like the inputs, run the length of the sequence.
    axes = {name: {1: "seq"} for name in (*INPUTS, "spans", "emphasis")}
    torch.onnx.export(
        model,
        (ids, torch.ones_like(ids), sentence),
        str(out / "model.onnx"),
        input_names=list(INPUTS),
        output_names=list(HEADS),
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    quantize(out)


def quantize(out: Path):
    """``model.int8.onnx`` from a folder's ``model.onnx``. Only the MLP up-projections: quantizing the
    down-projections (and attention) cost a third of the dev score on Ettin, from the outlier activations after
    the GeGLU; these alone keep nearly all of it."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    up = [
        n.name
        for n in onnx.load(str(out / "model.onnx")).graph.node
        if n.op_type == "MatMul" and "/mlp/Wi/" in n.name
    ]
    quantize_dynamic(
        str(out / "model.onnx"),
        str(out / "model.int8.onnx"),
        weight_type=QuantType.QInt8,
        nodes_to_quantize=up,
    )
