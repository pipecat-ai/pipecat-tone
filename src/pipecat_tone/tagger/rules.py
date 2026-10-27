"""Hand-written rules after Pipecat's MATCH_STYLE and SLOW_FOR_DETAILS: no model, no download.

Details go in slow spans. A frustrated caller (clear) gets calm, a sad one (likely) sympathetic, a
positive one (clear) warm; sarcasm (clear) turns positive into calm. Apologies and bad news are
sympathetic, good news warm, and a joke gets a chuckle.
"""

import re

from . import patterns, readings
from .labels import LabelSpace, SentenceLabels, Span
from .tagger import Tagger, TurnContext

# What to fall back to when the TTS or policy masks a delivery.
_FALLBACK = {
    "sympathetic": ["sympathetic", "sad", "calm"],
    "apologetic": ["apologetic", "sympathetic", "calm"],
    "warm": ["warm", "excited"],
    "calm": ["calm"],
}


class RulesTagger(Tagger):
    name = "rules"

    def __init__(self):
        self._space: LabelSpace | None = None
        self._mood: str | None = None
        self._joke = False

    def start_call(self) -> None:
        self._space = None

    async def start_turn(self, turn: TurnContext) -> None:
        self._space = turn.space
        tone, sarcasm = readings.levels(turn.heard)
        self._mood = None
        if sarcasm >= readings.CLEAR or tone.get("frustrated", 0) >= readings.CLEAR:
            self._mood = "calm"
        elif tone.get("sad", 0) >= readings.LIKELY:
            self._mood = "sympathetic"
        elif tone.get("positive", 0) >= readings.CLEAR:
            self._mood = "warm"
        caller = turn.caller.text if turn.caller else ""
        self._joke = bool(patterns.JOKE.search(caller)) and sarcasm < readings.LIKELY

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        if not self._space:
            return SentenceLabels()
        if patterns.SORRY.search(sentence):
            wanted = "apologetic" if re.search(r"apologi", sentence, re.IGNORECASE) else "sympathetic"
        elif patterns.GOOD_NEWS.search(sentence) and self._mood not in ("calm", "sympathetic"):
            wanted = "warm"
        else:
            wanted = self._mood
        spans = []
        if "slow" in self._space.deliveries:
            spans = [Span(m.start(), m.end(), "slow") for m in patterns.DETAILS.finditer(sentence)]
        sound = None
        if self._joke and not earlier:
            sound = next((s for s in ("chuckle", "laugh") if s in self._space.sounds), None)
        return SentenceLabels(delivery=self._pick(wanted), spans=spans, sound_before=sound)

    def _pick(self, wanted: str | None) -> str | None:
        return next((label for label in _FALLBACK.get(wanted, []) if label in self._space.deliveries), None)
