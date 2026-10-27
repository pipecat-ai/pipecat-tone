"""The model's input: a text pair per sentence, shared by training and the runtime.

The first text is the end of the call. The second holds the TTS and its labels, the policy presets,
the caller's reading in bands, state hints from the whole call, the previous sentence, and the
sentence, last. State hints stand in for a long context window, which would triple the sentence
pass's time.
"""

from dataclasses import dataclass

from pipecat.expressive.policy import ExpressionGuidance

from . import patterns, readings
from .labels import SentenceLabels
from .presets import BREATHE
from .tagger import Message, TurnContext

PRESET_NAMES = {
    ExpressionGuidance.MATCH_STYLE: "match style",
    ExpressionGuidance.SLOW_FOR_DETAILS: "slow for details",
    BREATHE: "breathe",
}


@dataclass(frozen=True)
class Layout:
    """What the model reads: messages of context, earlier sentences, guidance, state hints, the labels of the agent's
    previous reply (``previous_tags``), token budget.

    Over budget, ``cut_call_first`` cuts the call's oldest tokens (and, if the rest alone is over, the
    longer text); off, the longer text is cut from its start, which can drop the label space. A model's
    config records the layout it was trained with.
    """

    context_turns: int = 1
    earlier: int = 1
    custom_guidance: bool = False
    state: bool = True
    max_tokens: int = 160
    cut_call_first: bool = False
    previous_tags: bool = False


COMPACT = Layout()


def state_hints(messages: list[Message], sentence: str) -> list[str]:
    """Facts from the whole call that a short context window misses."""
    agent = [m.text for m in messages if m.role != "user"]
    caller = [m.text for m in messages if m.role == "user"]
    hints = []
    if any(patterns.APOLOGY.search(text) for text in agent):
        hints.append("apologized before")
    if any(patterns.SLOW_OR_REPEAT.search(text) for text in caller):
        hints.append("caller asked to slow down or repeat")
    details = [patterns.normalize(m[0]) for m in patterns.DETAILS.finditer(sentence)]
    if details:
        said = patterns.normalize(" ".join(agent))
        last = patterns.normalize(caller[-1]) if caller else ""
        if any(d in last for d in details):
            hints.append("caller just read this detail back")
        elif any(d in said for d in details):
            hints.append("detail given before")
        else:
            hints.append("new detail")
    return hints


def last_reply(labels: list[SentenceLabels] | None) -> str:
    """The deliveries and sounds the agent's previous reply used, in order: "none" before the first reply."""
    if labels is None:
        return "none"
    used = []
    for lab in labels:
        used += [lab.sound_before, lab.delivery, *(s.delivery for s in lab.spans), lab.sound_after]
    return " ".join(dict.fromkeys(u for u in used if u)) or "plain"


def encode(turn: TurnContext, earlier: list[str], sentence: str, layout: Layout) -> tuple[str, str, int]:
    """The text pair for one sentence, and where the sentence starts in the second text."""
    lines = []
    for m in turn.messages[-layout.context_turns :]:
        lines.append(
            f"caller ({readings.describe(m.heard)}): {m.text}" if m.role == "user" else f"agent: {m.text}"
        )

    guidance = turn.policy.guidance
    presets = [PRESET_NAMES[g] for g in guidance if g in PRESET_NAMES]
    custom = [g for g in guidance if g not in PRESET_NAMES] if layout.custom_guidance else []
    space = turn.space
    header = [
        f"tts {space.tts}",
        "can " + (" ".join(space.deliveries) or "nothing"),
        "sounds " + (" ".join(space.sounds) or "none"),
        f"stress {'yes' if space.emphasis else 'no'}",
        "policy " + (", ".join(presets) or "none") + ("; " + " ".join(custom) if custom else ""),
        f"caller sounded {readings.describe(turn.heard)}",
    ]
    if layout.previous_tags:
        header.append("last reply " + last_reply(turn.previous))
    if layout.state:
        header.append("state " + (", ".join(state_hints(turn.messages, sentence)) or "none"))
    before = " ".join(earlier[-layout.earlier :]) if layout.earlier else ""
    prefix = " | ".join(header) + f" | reply so far: {before or '-'} | sentence: "
    return "\n".join(lines), prefix + sentence, len(prefix)
