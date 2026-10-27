"""Pipecat's expression markup: writing a sentence's labels in the call's TTS names, and reading words back.

Labels a call's :class:`LabelSpace` masks, or that would overlap, are dropped, so the markup always
parses and speaks exactly the sentence's words. Words are read back with Pipecat's own formatter.
"""

import re

from loguru import logger
from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import NEUTRAL_DELIVERY
from pipecat.expressive.formatter import EXPRESSION_TAGS

from .._shared import shown_words
from .labels import LabelSpace, Pause, SentenceLabels

# A TTS's own markup in words it reports: Cartesia's <emotion .../>, ElevenLabs' [sympathetic].
TTS_MARKUP = re.compile(r"<[^<>]*>|\[[^\[\]]*\]")
_OPENERS = tuple(start for start, _ in EXPRESSION_TAGS)


class ReplyRenderer:
    """Renders one reply's sentences in order: a delivery tag only when it changes, ``<<neutral>>`` back."""

    def __init__(self, space: LabelSpace):
        self._space = space
        self._delivery: str | None = None

    def render(self, sentence: str, labels: SentenceLabels) -> str:
        """One sentence of the reply as markup."""
        text = sentence.strip()
        if not text:
            return text
        kept = keep(text, labels, self._space)
        names = self._space.deliveries

        # (offset, order, tag): at one offset a close comes first, then a pause, then an open.
        inserts: list[tuple[int, int, str]] = []
        for span in kept.spans:
            inserts += [(span.start, 2, f"<<{names[span.delivery]}: "), (span.end, 0, ">>")]
        for start, end in kept.emphasis:
            inserts += [(start, 2, "^^"), (end, 0, "^^")]
        for pause in kept.pauses:
            inserts.append((pause.at, 1, f" [[pause {pause.ms}]]"))
        for at, _, tag in sorted(inserts, key=lambda i: (i[0], i[1]), reverse=True):
            text = text[:at] + tag + text[at:]

        if kept.sound_before:
            text = f"[[{self._space.sound_name(kept.sound_before)}]] {text}"
        if kept.sound_after:
            text = f"{text} [[{self._space.sound_name(kept.sound_after)}]]"
        delivery = names[kept.delivery] if kept.delivery else None
        if delivery != self._delivery:
            text = f"<<{delivery or NEUTRAL_DELIVERY}>> {text}"
            self._delivery = delivery
        return text


def render_reply(space: LabelSpace, sentences: list[str], labels: list[SentenceLabels]) -> str:
    """A whole reply as markup, its sentences joined by spaces."""
    renderer = ReplyRenderer(space)
    return " ".join(renderer.render(s, label) for s, label in zip(sentences, labels))


def keep(sentence: str, labels: SentenceLabels, space: LabelSpace) -> SentenceLabels:
    """The labels a renderer writes: allowed by the call, on whole words, not overlapping. Training targets too."""
    core = sentence.strip()
    taken: list[tuple[int, int]] = []
    spans = []
    for span in sorted(labels.spans, key=lambda s: s.start):
        if (
            span.delivery in space.deliveries
            and _words(core, span.start, span.end)
            and not _overlaps(taken, span.start, span.end)
        ):
            taken.append((span.start, span.end))
            spans.append(span)
        else:
            _dropped("span", span, core)
    emphasis = []
    if space.emphasis:
        for start, end in sorted(labels.emphasis):
            if _words(core, start, end) and not _overlaps(taken, start, end):
                taken.append((start, end))
                emphasis.append((start, end))
            else:
                _dropped("emphasis", (start, end), core)
    pauses = []
    for pause in labels.pauses:
        ms = space.clamp_pause(pause.ms)
        if (
            ms is not None
            and 0 < pause.at < len(core)
            and core[pause.at] == " "
            and not _inside(taken, pause.at)
        ):
            pauses.append(Pause(pause.at, ms))
        else:
            _dropped("pause", pause, core)
    return SentenceLabels(
        delivery=labels.delivery if labels.delivery in space.deliveries else None,
        spans=spans,
        emphasis=emphasis,
        pauses=pauses,
        sound_before=labels.sound_before if space.sound_name(labels.sound_before or "") else None,
        sound_after=labels.sound_after if space.sound_name(labels.sound_after or "") else None,
    )


def same_words(plain: str, markup: str, adapter: BaseExpressionAdapter) -> bool:
    """Whether markup speaks exactly the plain text's words, however it's spaced."""
    return " ".join(shown_words(markup, adapter).split()) == " ".join(plain.split())


def has_markup(text: str) -> bool:
    return any(opener in text for opener in _OPENERS)


def without_markup(text: str) -> str:
    """Text without expression tags, keeping only the whitespace at its ends."""
    return _with_ends(text, shown_words(text)) if has_markup(text) else text


def without_tts_markup(text: str) -> str:
    """Words a TTS reported, without its own markup or the tagger's."""
    text = without_markup(text)
    if "<" not in text and "[" not in text:
        return text
    return _with_ends(text, " ".join(TTS_MARKUP.sub("", text).split()))


def _with_ends(text: str, words: str) -> str:
    """Words with the whitespace at the ends of the text they came from."""
    if not words:
        return ""
    return text[: len(text) - len(text.lstrip())] + words + text[len(text.rstrip()) :]


def _words(core: str, start: int, end: int) -> bool:
    return (
        0 <= start < end <= len(core)
        and not core[start].isspace()
        and not core[end - 1].isspace()
        and _boundary(core, start)
        and _boundary(core, end)
    )


def _boundary(core: str, at: int) -> bool:
    return at in (0, len(core)) or not (core[at - 1].isalnum() and core[at].isalnum())


def _overlaps(taken: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < e and s < end for s, e in taken)


def _inside(taken: list[tuple[int, int]], at: int) -> bool:
    return any(s < at < e for s, e in taken)


def _dropped(kind: str, label: object, core: str):
    logger.debug(f"tone tagger: dropped {kind} {label} in {core!r}")
