"""The teacher: Claude labels fixed agent replies, never changing a word.

It sees the call, each caller reading, the TTS's labels as that TTS speaks them, and the policy.
A fixed schema lists every canonical label, so it compiles once; the label space drops the rest.
"""

from dataclasses import dataclass, field

import anthropic
from pipecat.expressive.perception import PerceptionSignal, format_perception
from pipecat.expressive.policy import ExpressionPolicy

from lab import llm
from pipecat_tone.tagger.labels import LabelSpace, Pause, SentenceLabels, Span, label_map

# Bumped when the guidance changes, so older labels can be told apart and redone. v4 only adds breaths, so a call
# whose policy offers none is labeled as v3 labeled it.
TEACHER_VERSION = 4
BREATHLESS_VERSION = 3
TEACHER_EFFORT = "high"

# When a breath goes in, for a call that asks to breathe. The breath pass draws its own, at people's rates.
BREATH_RULE = """\
- A breath is offered only when the developer asks for breathing. Then it goes in sound_before: an \
audible breath before a sentence that is long, carries hard news, or starts a careful explanation, as a \
person on the phone breathes before it. At most one per reply, and most replies need none: never before \
a short or routine sentence, never by habit at the start of every reply, and never where another sound \
starts the sentence."""

SYSTEM = f"""\
You direct how a voice agent's replies are spoken on a customer service phone call. A text-to-speech \
service (TTS) speaks the agent's words. The words are fixed: you never change them. For each sentence \
of each agent reply you are asked about, you choose labels the TTS will perform.

What you choose for each sentence:
- delivery: how the whole sentence is spoken, or null for the voice's normal delivery. Give consecutive \
sentences the same delivery when a mood carries across them.
- slow: exact substrings of the sentence to say slowly. Only what the caller may need to write down or \
check character by character: a reference or confirmation code, an order, account or ticket number, a \
phone number, an email address, a street address to note down. Just the detail itself, never the whole \
sentence. Not prices, dates, times, amounts or small counts.
- emphasis: exact words to stress, rarely: only where stress carries meaning, such as correcting a \
misunderstanding. Only when the TTS stresses words.
- pause_after: rarely, a short pause after an exact word, in milliseconds, such as before a code the \
caller is getting ready to write down. Only when the TTS pauses.
- sound_before, sound_after: a sound at the start or end of the sentence, such as a chuckle when the \
caller has said something funny. Rare.

How to choose:
- Most sentences need no delivery. Routine sentences (asking for details, confirming, explaining steps) \
are spoken normally. Use a delivery where the moment clearly calls for one.
- Match the caller's mood and energy: warm or excited for their good news, sympathetic for their bad \
news or worry. When the caller clearly jokes, start the reply with a laugh or chuckle, when the TTS has one.
- Each label is listed with how this TTS speaks it, such as "warm (spoken as happy)". Choose by how it \
will sound: a happy or cheerful voice never fits a problem, disappointment, worry or sarcasm.
- When the caller is hostile, stay calm. Never answer anger with anger, and never sound sarcastic.
- The caller's reading, such as [heard: tone frustrated 0.86, neutral 0.10; sarcasm 0.04], is how they \
sounded, with how sure it is from 0 to 1. A label at 0.7 or above is clear; below 0.5 it is a guess: \
don't act on a guess. Weigh the reading against the caller's words. When sarcasm is 0.7 or above, the \
caller means the opposite of their words: "great, just great" is not good news.
- When the agent apologizes or gives bad news, a sympathetic or apologetic delivery fits that sentence. \
Don't give every sentence of a reply that delivery. Once the agent has apologized or sympathized for a \
problem earlier in the call, a later sentence that apologizes again for the same problem stays normal: \
the words already carry it.
- After an emotional sentence, routine sentences that follow go back to normal unless the mood carries.
- Slow every detail the caller may need to write down or check, each time the agent says it: the first \
time and whenever it is repeated. The one exception: when the caller has just read it back and the agent \
only confirms it, it can stay normal.
{BREATH_RULE}
- Use only the labels listed as available for this call. When the label you want isn't available, use \
the closest available one, or none.
- Follow the developer's guidance for this call.
- Label each reply from the call up to and including that reply. Never use what is said after it.

Return an entry for every sentence of every reply you are asked about, numbered as given. Quote \
substrings exactly as they appear in the sentence."""


@dataclass
class Exchange:
    """A caller message, its reading, and the agent's reply in the sentences the tagger sees."""

    caller: str
    heard: list[PerceptionSignal] | None
    sentences: list[str]


@dataclass
class Teaching:
    labels: list[list[SentenceLabels]]
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None


class Teacher:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self._client = client

    async def label(self, exchanges: list[Exchange], space: LabelSpace, policy: ExpressionPolicy) -> Teaching:
        """Labels for every reply; a reply or sentence the teacher leaves out gets none."""
        reply = await llm.claude_json(
            self._client, SYSTEM, _prompt(exchanges, space, policy), _schema(), TEACHER_EFFORT
        )
        labels = _parse(reply.data, exchanges) if reply.data else _empty(exchanges)
        return Teaching(labels, reply.usage, reply.cost_usd, reply.error and f"teacher {reply.error}")


def _prompt(exchanges: list[Exchange], space: LabelSpace, policy: ExpressionPolicy) -> str:
    guidance = policy.guidance or ["(none)"]
    turns = [
        (f"{format_perception(e.heard)} {e.caller}" if e.heard else e.caller, e.sentences) for e in exchanges
    ]
    return "\n".join(
        [
            f"TTS: {space.tts}",
            f"Available deliveries: {_spoken(space.deliveries) or 'none'}",
            f"Available sounds: {_spoken(space.sounds) or 'none'}",
            f"Stresses words: {'yes' if space.emphasis else 'no'}",
            f"Pauses: {'yes' if space.pause_range_ms else 'no'}",
            "",
            *llm.call_lines(guidance, turns),
            f"Label every sentence of the agent's replies in turns 1 to {len(exchanges)}.",
        ]
    )


def _spoken(names: dict[str, str]) -> str:
    return ", ".join(
        label if name == label else f"{label} (spoken as {name})" for label, name in names.items()
    )


def _schema() -> dict:
    labels = label_map()
    sentence = llm.obj(
        sentence=llm.INTEGER,
        delivery=llm.enum_or_null(list(labels.deliveries)),
        slow=llm.array(llm.STRING),
        emphasis=llm.array(llm.STRING),
        pause_after=llm.array(llm.obj(word=llm.STRING, ms=llm.INTEGER)),
        sound_before=llm.enum_or_null(list(labels.sounds)),
        sound_after=llm.enum_or_null(list(labels.sounds)),
    )
    return llm.obj(replies=llm.array(llm.obj(turn=llm.INTEGER, sentences=llm.array(sentence))))


def _parse(data: dict, exchanges: list[Exchange]) -> list[list[SentenceLabels]]:
    labels = _empty(exchanges)
    for reply in data.get("replies", []):
        r = reply.get("turn", 0) - 1
        if not 0 <= r < len(labels):
            continue
        sentences = exchanges[r].sentences
        for item in reply.get("sentences", []):
            n = item.get("sentence", 0) - 1
            if 0 <= n < len(sentences):
                labels[r][n] = _sentence_labels(item, sentences[n])
    return labels


def _sentence_labels(item: dict, sentence: str) -> SentenceLabels:
    pauses = []
    for p in item.get("pause_after", []):
        found = _find(sentence, p.get("word", ""))
        if found and found[1] < len(sentence) and sentence[found[1]] == " ":
            pauses.append(Pause(found[1], int(p.get("ms", 300))))
    return SentenceLabels(
        delivery=item.get("delivery"),
        spans=[
            Span(s, e, "slow") for s, e in filter(None, (_find(sentence, q) for q in item.get("slow", [])))
        ],
        emphasis=[r for r in (_find(sentence, q) for q in item.get("emphasis", [])) if r],
        pauses=pauses,
        sound_before=item.get("sound_before"),
        sound_after=item.get("sound_after"),
    )


def _find(sentence: str, quote: str) -> tuple[int, int] | None:
    """Where a quoted substring sits, trimmed of spaces and end punctuation; case-blind as a fallback."""
    quote = quote.strip().rstrip(".,!?;:")
    if not quote:
        return None
    at = sentence.find(quote)
    if at < 0:
        at = sentence.lower().find(quote.lower())
    return (at, at + len(quote)) if at >= 0 else None


def _empty(exchanges: list[Exchange]) -> list[list[SentenceLabels]]:
    return [[SentenceLabels() for _ in e.sentences] for e in exchanges]
