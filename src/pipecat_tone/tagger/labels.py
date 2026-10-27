"""Canonical labels, the names each TTS performs them by (``labels.yaml``), and a sentence's labels."""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml
from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.adapters.tts.services.minimax_adapter import MiniMaxExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities
from pipecat.expressive.policy import ExpressionPolicy

from .presets import BREATHE

TTS_ADAPTERS: dict[str, type[BaseExpressionAdapter]] = {
    "cartesia": CartesiaExpressionAdapter,
    "elevenlabs": ElevenLabsAudioTagExpressionAdapter,
    "fish": FishAudioS2ExpressionAdapter,
    "minimax": MiniMaxExpressionAdapter,
}


@dataclass(frozen=True)
class LabelMap:
    """Each canonical label's candidate names, per TTS, best first."""

    deliveries: dict[str, dict[str, list[str]]]
    sounds: dict[str, dict[str, list[str]]]
    breaths: dict[str, dict[str, list[str]]]


@cache
def label_map() -> LabelMap:
    raw = yaml.safe_load(Path(__file__).with_name("labels.yaml").read_text())
    return LabelMap(
        deliveries={label: dict(names) for label, names in raw["deliveries"].items()},
        sounds={label: dict(names) for label, names in raw["sounds"].items()},
        breaths={label: dict(names) for label, names in raw.get("breaths", {}).items()},
    )


def is_breath(sound: str | None) -> bool:
    """Whether a sound is a breath: the one a model chooses, or a kind it's drawn as."""
    return sound == "breath" or sound in label_map().breaths


def tts_key(service_name: str) -> str | None:
    """The labels.yaml key in a TTS service's name, such as "CartesiaTTSService#0"."""
    name = service_name.lower()
    return next((key for key in TTS_ADAPTERS if key in name), None)


@dataclass(frozen=True)
class LabelSpace:
    """The labels one call can use, each with the name its TTS performs.

    Parameters:
        tts: The labels.yaml key of the TTS.
        deliveries: Canonical delivery -> name to write.
        sounds: Canonical sound -> name to write.
        emphasis: Whether the TTS stresses ``^^words^^``.
        pause_range_ms: The shortest and longest pause, or None.
        breaths: Kind of breath -> name to write, where breath is offered. A model chooses only breath, so the
            kinds stay out of everything it reads; a tagger draws which kind a breath is.
    """

    tts: str
    deliveries: dict[str, str]
    sounds: dict[str, str]
    emphasis: bool
    pause_range_ms: tuple[int, int] | None
    breaths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        tts: str,
        capabilities: ExpressionCapabilities,
        policy: ExpressionPolicy | None = None,
        vocabulary: dict[str, list[str]] | None = None,
    ) -> "LabelSpace":
        """Each canonical label's first name the TTS performs and the policy offers; the rest are masked.
        Breath is offered only with the ``BREATHE`` preset in the policy's guidance.

        Names are tried in order: the vocabulary's for the label, then labels.yaml's for the TTS. For a TTS
        labels.yaml doesn't list, they're the canonical name and every name it has for the label.
        """
        policy = policy or ExpressionPolicy()
        vocabulary = vocabulary or {}
        labels = label_map()
        breathes = BREATHE in policy.guidance
        return cls(
            tts=tts,
            deliveries=_resolve_all(
                labels.deliveries,
                tts,
                {t.name for t in capabilities.delivery},
                capabilities.free_text_delivery,
                policy.delivery,
                vocabulary,
            ),
            sounds=_resolve_all(
                {k: v for k, v in labels.sounds.items() if k != "breath" or breathes},
                tts,
                {t.name for t in capabilities.sounds},
                capabilities.free_text_sounds,
                policy.sounds,
                vocabulary,
            ),
            emphasis=capabilities.emphasis,
            pause_range_ms=capabilities.pause_range_ms,
            breaths=_resolve_all(
                labels.breaths if breathes else {},
                tts,
                {t.name for t in capabilities.sounds},
                capabilities.free_text_sounds,
                policy.sounds,
                vocabulary,
            ),
        )

    @classmethod
    def for_tts(
        cls, tts: str, policy: ExpressionPolicy | None = None, vocabulary: dict[str, list[str]] | None = None
    ) -> "LabelSpace":
        """Resolved against the TTS's own adapter."""
        if tts not in TTS_ADAPTERS:
            raise ValueError(f"unknown tts {tts!r}, expected one of {list(TTS_ADAPTERS)}")
        return cls.resolve(tts, TTS_ADAPTERS[tts]().capabilities(), policy, vocabulary)

    def sound_name(self, sound: str) -> str | None:
        """The name to write for a sound or a kind of breath, if the call offers it."""
        return self.sounds.get(sound) or self.breaths.get(sound)

    def clamp_pause(self, ms: int) -> int | None:
        if self.pause_range_ms is None:
            return None
        low, high = self.pause_range_ms
        return min(max(ms, low), high)

    def to_dict(self) -> dict:
        return {
            "tts": self.tts,
            "deliveries": self.deliveries,
            "sounds": self.sounds,
            "emphasis": self.emphasis,
            "pause_range_ms": list(self.pause_range_ms) if self.pause_range_ms else None,
            "breaths": self.breaths,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSpace":
        def names(value) -> dict[str, str]:
            # Corpus v0 rows list the canonical labels only.
            return dict(value) if isinstance(value, dict) else {label: label for label in value}

        pause = d.get("pause_range_ms")
        return cls(
            d["tts"],
            names(d["deliveries"]),
            names(d["sounds"]),
            d["emphasis"],
            tuple(pause) if pause else None,
            dict(d.get("breaths", {})),
        )


@dataclass(frozen=True)
class Span:
    """A delivery for some words, ``<<delivery: words>>``. Offsets count in the stripped sentence."""

    start: int
    end: int
    delivery: str


@dataclass(frozen=True)
class Pause:
    """A pause at the space at ``at``, ``[[pause N]]``."""

    at: int
    ms: int


@dataclass
class SentenceLabels:
    """What a tagger chose for one sentence, in canonical labels; usually nothing.

    Parameters:
        delivery: The sentence's delivery, or None for the voice's baseline.
        spans: Words in a delivery of their own, mainly slow details.
        emphasis: Words to stress, as ``(start, end)`` offsets.
        pauses: Pauses between words.
        sound_before: A sound before the sentence, or None.
        sound_after: A sound after the sentence, or None.
    """

    delivery: str | None = None
    spans: list[Span] = field(default_factory=list)
    emphasis: list[tuple[int, int]] = field(default_factory=list)
    pauses: list[Pause] = field(default_factory=list)
    sound_before: str | None = None
    sound_after: str | None = None

    def to_dict(self) -> dict:
        return {
            "delivery": self.delivery,
            "spans": [[s.start, s.end, s.delivery] for s in self.spans],
            "emphasis": [list(r) for r in self.emphasis],
            "pauses": [[p.at, p.ms] for p in self.pauses],
            "sound_before": self.sound_before,
            "sound_after": self.sound_after,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceLabels":
        return cls(
            delivery=d["delivery"],
            spans=[Span(a, b, delivery) for a, b, delivery in d["spans"]],
            emphasis=[tuple(r) for r in d["emphasis"]],
            pauses=[Pause(at, ms) for at, ms in d.get("pauses", [])],
            sound_before=d["sound_before"],
            sound_after=d["sound_after"],
        )


def _resolve_all(
    labels: dict[str, dict[str, list[str]]],
    tts: str,
    declared: set[str],
    free_text: bool,
    offered: list[str] | None,
    vocabulary: dict[str, list[str]],
) -> dict[str, str]:
    resolved = {}
    for label, by_tts in labels.items():
        listed = (
            by_tts[tts] if tts in by_tts else [label, *(name for names in by_tts.values() for name in names)]
        )
        # A fixed list performs only what it declares; free text performs any name, the canonical one last.
        for name in dict.fromkeys([*vocabulary.get(label, []), *listed, *([label] if free_text else [])]):
            if (free_text or name in declared) and (offered is None or name in offered):
                resolved[label] = name
                break
    return resolved
