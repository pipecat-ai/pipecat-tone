"""Rows as model inputs and targets, encoded as the runtime encodes them."""

from dataclasses import dataclass

from lab.corpus.rows import Row
from pipecat_tone.tagger.features import Layout
from pipecat_tone.tagger.model import (
    BEGIN,
    INSIDE,
    OUTSIDE,
    Encoded,
    class_mask,
    delivery_classes,
    encode_tokens,
    sound_classes,
)

IGNORE = -100


@dataclass
class Example:
    """A row's model input, and its targets."""

    encoded: Encoded
    delivery: int
    sound_before: int
    sound_after: int
    spans: list[int]
    emphasis: list[int]
    delivery_mask: list[int]
    sound_mask: list[int]
    breath: int
    breath_offered: int


def make_example(row: Row, tokenizer, layout: Layout) -> Example:
    enc = encode_tokens(tokenizer, row.turn_context(), row.earlier, row.sentence, layout)
    labels, space = row.labels, row.space
    deliveries, sounds = delivery_classes(), sound_classes()
    # Breath has its own head, so the sound heads never learn it.
    breathes = labels.sound_before == "breath"
    return Example(
        encoded=enc,
        delivery=deliveries.index(labels.delivery or "none"),
        sound_before=sounds.index("none" if breathes else labels.sound_before or "none"),
        sound_after=sounds.index(labels.sound_after or "none"),
        spans=_bio(enc.offsets, [(s.start, s.end) for s in labels.spans if s.delivery == "slow"]),
        emphasis=_bio(enc.offsets, labels.emphasis) if space.emphasis else [IGNORE] * len(enc.offsets),
        delivery_mask=[int(m) for m in class_mask(deliveries, space.deliveries)],
        sound_mask=[int(m) for m in class_mask(sounds, [s for s in space.sounds if s != "breath"])],
        breath=int(breathes),
        breath_offered=int("breath" in space.sounds),
    )


def _bio(offsets: list[tuple[int, int] | None], ranges: list[tuple[int, int]]) -> list[int]:
    out = []
    previous: tuple[int, int] | None = None
    for off in offsets:
        if off is None:
            out.append(IGNORE)
            previous = None
            continue
        inside = next(((a, b) for a, b in ranges if off[0] < b and off[1] > a), None)
        out.append(OUTSIDE if inside is None else INSIDE if inside == previous else BEGIN)
        previous = inside
    return out
