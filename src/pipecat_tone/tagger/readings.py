"""Caller readings: parsed from the context, folded into the tagger's four tones, described in bands.

The tagger was trained on a ``tone`` signal (positive, neutral, sad, frustrated). Pipecat Tone
gives an ``emotion`` signal with nine labels; :func:`fold` maps them in, adding confidences.
"""

import re

from pipecat.expressive.perception import PerceptionSignal

CLEAR = 0.7
LIKELY = 0.5

TONE_LABELS = ("positive", "neutral", "sad", "frustrated")

EMOTION_TO_TONE = {
    "neutral": "neutral",
    "happy": "positive",
    "amused": "positive",
    "surprised": "neutral",
    "sad": "sad",
    "afraid": "sad",
    "angry": "frustrated",
    "disgusted": "frustrated",
    "contemptuous": "frustrated",
}

# What the user aggregator writes before a caller's words (pipecat.expressive.perception.format_perception).
_HEARD = re.compile(r"^\s*\[heard: ([^\]]*)\]\s*")
_LABEL = re.compile(r"(.+?) (-?\d+(?:\.\d+)?)$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def parse(content: str) -> tuple[list[PerceptionSignal] | None, str]:
    """A caller message's folded reading and its words: ``"[heard: tone sad 0.86] No."``."""
    match = _HEARD.match(content)
    if not match:
        return None, content
    signals = []
    for part in match[1].split("; "):
        name, _, rest = part.partition(" ")
        if _NUMBER.fullmatch(rest):
            signals.append(PerceptionSignal(name, value=float(rest)))
        else:
            labels = [_LABEL.match(label.strip()) for label in rest.split(", ")]
            signals.append(PerceptionSignal(name, labels=[(m[1], float(m[2])) for m in labels if m]))
    return fold(signals), content[match.end() :]


def fold(heard: list[PerceptionSignal] | None) -> list[PerceptionSignal] | None:
    """An ``emotion`` signal as a ``tone`` signal; other signals unchanged."""
    if not heard:
        return heard
    out = []
    for signal in heard:
        if signal.name == "emotion" and signal.labels:
            totals: dict[str, float] = {}
            for label, confidence in signal.labels:
                if tone := EMOTION_TO_TONE.get(label):
                    totals[tone] = totals.get(tone, 0.0) + confidence
            labels = sorted(((t, round(min(c, 1.0), 2)) for t, c in totals.items()), key=lambda p: -p[1])[:2]
            out.append(PerceptionSignal("tone", labels=labels))
        else:
            out.append(signal)
    return out


def levels(heard: list[PerceptionSignal] | None) -> tuple[dict[str, float], float]:
    """Tone label -> confidence, and the sarcasm measure."""
    tone: dict[str, float] = {}
    sarcasm = 0.0
    for signal in heard or []:
        if signal.name == "tone":
            tone = dict(signal.labels)
        elif signal.name == "sarcasm" and signal.value is not None:
            sarcasm = signal.value
    return tone, sarcasm


def describe(heard: list[PerceptionSignal] | None) -> str:
    """The reading in bands, as the model reads it: ``"tone sad clear, neutral guess; sarcasm low"``."""
    if not heard:
        return "no reading"
    parts = []
    for signal in heard:
        if signal.labels:
            parts.append(signal.name + " " + ", ".join(f"{label} {_band(c)}" for label, c in signal.labels))
        elif signal.value is not None:
            parts.append(f"{signal.name} {_level(signal.value)}")
    return "; ".join(parts)


def _band(confidence: float) -> str:
    return "clear" if confidence >= CLEAR else "likely" if confidence >= LIKELY else "guess"


def _level(value: float) -> str:
    return "high" if value >= CLEAR else "some" if value >= LIKELY else "low"
