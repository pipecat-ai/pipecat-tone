"""The trained tagger: an ONNX encoder with its heads on CPU, and the encoding and decoding training shares.

Heads: a delivery, a sound before and a sound after (each a class or none), per sentence token slow
spans and stressed words (begin, inside, outside), and a breath head: whether a breath comes before the
sentence, decided apart from the other sounds. Classes the label space masks never win.
A model folder holds ``config.json``, ``tokenizer.json`` and the ONNX file its config names; releases
are on Hugging Face as ``tagger/<size>`` in ``pipecat-ai/tone-models``.
"""

import asyncio
import json
import random
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import hub
from .._onnx import session, tokenizer
from .features import Layout, encode
from .labels import LabelSpace, SentenceLabels, Span, label_map
from .tagger import Tagger, TurnContext

TAGGER_REVISION = "tagger-rc1"

# The ONNX graph's inputs and heads, in order.
INPUTS = ("input_ids", "attention_mask", "sentence_mask")
HEADS = ("delivery", "sound_before", "sound_after", "spans", "emphasis", "breath")
OUTSIDE, BEGIN, INSIDE = 0, 1, 2


def delivery_classes() -> list[str]:
    return ["none", *label_map().deliveries]


def sound_classes() -> list[str]:
    return ["none", *label_map().sounds]


def class_mask(classes: list[str], allowed) -> list[bool]:
    return [c == "none" or c in allowed for c in classes]


@dataclass
class Encoded:
    """Model input for one sentence; ``offsets`` holds each sentence token's characters, None elsewhere."""

    input_ids: list[int]
    attention_mask: list[int]
    sentence_mask: list[int]
    offsets: list[tuple[int, int] | None]


def encode_tokens(tokenizer, turn: TurnContext, earlier: list[str], sentence: str, layout: Layout) -> Encoded:
    """Tokenize a sentence's text pair with a ``tokenizers.Tokenizer``, within the layout's budget."""
    first, second, start = encode(turn, earlier, sentence, layout)
    enc = _encode_pair(tokenizer, first, second, layout)
    offsets: list[tuple[int, int] | None] = []
    sentence_mask = []
    for seq, special, (a, b) in zip(enc.sequence_ids, enc.special_tokens_mask, enc.offsets):
        inside = seq == 1 and not special and a >= start and b > a
        sentence_mask.append(int(inside))
        offsets.append((a - start, b - start) if inside else None)
    return Encoded(enc.ids, enc.attention_mask, sentence_mask, offsets)


def decode(
    heads,
    offsets: list[tuple[int, int] | None],
    sentence: str,
    space: LabelSpace,
    classes: tuple[list, list] | None = None,
    sound_bias: dict[str, list] | None = None,
) -> SentenceLabels:
    """Labels from one sentence's head outputs, in ``HEADS`` order.

    ``classes`` are the model's delivery and sound classes, from its config; by default, the current ones.
    ``sound_bias`` shifts a sound's logits by the sentence's words, as ``[[most words, shift], ...,
    [None, shift]]``, as a model's config calibrates its breaths. A breath goes where the breath head's logit,
    shifted, is above zero and no other sound goes.
    """
    deliveries, sounds = classes or (delivery_classes(), sound_classes())
    delivery, before, after, spans, emphasis, breath = (np.asarray(h) for h in heads)
    count = words(sentence)
    bias = {c: _shift(b, count) for c, b in (sound_bias or {}).items()}
    shift = np.array([bias.get(c, 0.0) for c in sounds])
    offered = [s for s in space.sounds if s != "breath"]
    sound_before = _choose(before + shift, sounds, offered)
    if not sound_before and "breath" in space.sounds and breath[0] + bias.get("breath", 0.0) > 0:
        sound_before = "breath"
    n = len(offsets)
    return SentenceLabels(
        delivery=_choose(delivery, deliveries, space.deliveries),
        spans=[Span(a, b, "slow") for a, b in decode_ranges(spans[:n].argmax(-1).tolist(), offsets, sentence)]
        if "slow" in space.deliveries
        else [],
        emphasis=decode_ranges(emphasis[:n].argmax(-1).tolist(), offsets, sentence) if space.emphasis else [],
        sound_before=sound_before,
        sound_after=_choose(after + shift, sounds, offered),
    )


def is_word(token: str) -> bool:
    """Whether a token is a word: it has a letter or a digit."""
    return any(c.isalnum() for c in token)


def words(sentence: str) -> int:
    """A sentence's words, as a sound bias by length counts them."""
    return sum(map(is_word, sentence.split()))


def _shift(bias: list, count: int) -> float:
    return next(shift for most, shift in bias if most is None or count <= most)


def decode_ranges(
    classes: list[int], offsets: list[tuple[int, int] | None], sentence: str
) -> list[tuple[int, int]]:
    """Character ranges from begin/inside/outside token classes, widened to whole words."""
    ranges: list[list[int]] = []
    current: list[int] | None = None
    for cls, off in zip(classes, offsets):
        if off is None:
            continue
        if cls == BEGIN or (cls == INSIDE and current is None):
            if current:
                ranges.append(current)
            current = [off[0], off[1]]
        elif cls == INSIDE:
            current[1] = off[1]
        elif current:
            ranges.append(current)
            current = None
    if current:
        ranges.append(current)
    out = []
    for a, b in ranges:
        # A subword token's offsets can include the space before it.
        while a < b and sentence[a].isspace():
            a += 1
        while b > a and sentence[b - 1].isspace():
            b -= 1
        if a == b:
            continue
        while a > 0 and sentence[a - 1].isalnum():
            a -= 1
        while b < len(sentence) and sentence[b].isalnum():
            b += 1
        out.append((a, b))
    return out


def _encode_pair(tokenizer, first: str, second: str, layout: Layout):
    if layout.cut_call_first:
        tokenizer.enable_truncation(layout.max_tokens, strategy="only_first", direction="left")
        # Tokenizers won't cut a text to nothing, so a call too long to keep any of goes whole.
        for call in (first, ""):
            try:
                return tokenizer.encode(call, second)
            except Exception:  # the cut can't fit the budget
                pass
    tokenizer.enable_truncation(layout.max_tokens, strategy="longest_first", direction="left")
    return tokenizer.encode(first, second)


def _choose(logits: np.ndarray, classes: list[str], allowed) -> str | None:
    best = classes[int(np.where(class_mask(classes, allowed), logits, -np.inf).argmax())]
    return None if best == "none" else best


class ToneTaggerModel:
    """The trained tagger's model, on the CPU. One model can serve every call."""

    def __init__(self, model_dir: str | Path, *, threads: int = 1):
        """Load the model from a folder.

        Args:
            model_dir: A folder with the model's ``config.json``, ``tokenizer.json`` and ONNX file.
            threads: The CPU threads for one sentence.
        """
        folder = Path(model_dir)
        self.config = json.loads((folder / "config.json").read_text())
        self.layout = Layout(**self.config["layout"])
        self._tokenizer = tokenizer(folder / "tokenizer.json")
        self._session = session(folder / self.config["onnx"], threads)
        # Encoding sets the tokenizer's truncation, so calls in other threads take turns.
        self._encoding = threading.Lock()

    @classmethod
    def download(
        cls, *, size: str = "17m", revision: str = TAGGER_REVISION, threads: int = 1
    ) -> "ToneTaggerModel":
        """Load a release from Hugging Face, downloading its files the first time.

        Raises:
            ModelDownloadError: The release can't be downloaded or found in the cache.
        """
        folder = f"tagger/{size}"
        config = json.loads((hub.download(folder, revision, ["config.json"]) / "config.json").read_text())
        return cls(
            hub.download(folder, revision, ["config.json", "tokenizer.json", config["onnx"]]), threads=threads
        )

    def heads(self, turn: TurnContext, earlier: list[str], sentence: str) -> tuple[list[np.ndarray], Encoded]:
        """One sentence's head outputs, in ``HEADS`` order, and its encoding."""
        with self._encoding:
            enc = encode_tokens(self._tokenizer, turn, earlier, sentence, self.layout)
        values = (enc.input_ids, enc.attention_mask, enc.sentence_mask)
        feeds = {name: np.array([v], dtype=np.int64) for name, v in zip(INPUTS, values)}
        return [h[0] for h in self._session.run(None, feeds)], enc

    def predict(self, turn: TurnContext, earlier: list[str], sentence: str) -> SentenceLabels:
        heads, enc = self.heads(turn, earlier, sentence)
        classes = (self.config["deliveries"], self.config["sounds"])
        return decode(heads, enc.offsets, sentence, turn.space, classes, self.config.get("sound_bias"))


class ModelTagger(Tagger):
    """One call's tagger on a model, which other calls may share. A model's config can cap a reply's breaths
    (``breaths_per_reply``): breaths past it are dropped. It can also give the kinds a breath is drawn as, with their
    shares (``breath_kinds``), among those the call's TTS has. A kind is drawn from the call so far, so a call tagged
    again sounds the same, and the model reads its last reply's breaths as breaths."""

    name = "model"

    def __init__(self, model: ToneTaggerModel):
        self._model = model
        self._turn: TurnContext | None = None
        self._reply: list[SentenceLabels] | None = None

    def start_call(self) -> None:
        self._turn, self._reply = None, None

    async def start_turn(self, turn: TurnContext) -> None:
        # The last reply's labels, which a model with previous_tags reads: the turn's own when it brings them, as
        # training rows do, else this tagger's.
        self._turn = turn if turn.previous is not None else replace(turn, previous=self._reply)
        self._reply = []
        self._draw = random.Random("\n".join(m.text for m in turn.messages))

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        if not self._turn:
            return SentenceLabels()
        # Off the event loop, as audio keeps flowing while the model runs.
        labels = await asyncio.to_thread(self._model.predict, self._turn, earlier, sentence)
        cap = self._model.config.get("breaths_per_reply")
        if labels.sound_before == "breath" and cap is not None:
            if sum(lab.sound_before == "breath" for lab in self._reply) >= cap:
                labels = replace(labels, sound_before=None)
        self._reply.append(labels)
        kinds = self._model.config.get("breath_kinds") or {}
        offered = {kind: share for kind, share in kinds.items() if kind in self._turn.space.breaths}
        if labels.sound_before == "breath" and offered:
            labels = replace(
                labels, sound_before=self._draw.choices(list(offered), list(offered.values()))[0]
            )
        return labels
