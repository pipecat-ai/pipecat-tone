"""The reviewer: another vendor's model passes or fails each reply's tagging; a failed reply isn't trained on."""

from openai import AsyncOpenAI

from lab import llm

REVIEWER = "LABELLER_D_MODEL"  # the variable naming the reviewer's model id; the reader's labeller D
REVIEWER_EFFORT = "medium"
# Bumped when the rubric changes, so older verdicts can be told apart and redone. v3 only adds breaths, so a call
# whose policy offers none is reviewed as v2 reviewed it.
REVIEWER_VERSION = 3
BREATHLESS_VERSION = 2

SYSTEM = """\
You review how a voice agent's replies were marked up for a text-to-speech service (TTS) on a customer \
service call. The agent's words are fixed; only the markup is under review. The markup:
- <<delivery>> at the start of a sentence speaks it, and what follows, in that delivery; <<neutral>> \
returns to the normal voice.
- <<delivery: words>> speaks just those words in that delivery, mostly <<slow: ...>> for a detail to \
write down.
- [[sound]] makes a sound, ^^word^^ stresses a word, [[pause N]] pauses N milliseconds.
A caller message may start with how they sounded, such as [heard: tone frustrated 0.86, neutral 0.10; \
sarcasm 0.04]: the likeliest tones with confidences from 0 to 1, and a sarcasm score. Below 0.5 a \
reading is a guess.

A reply's markup passes when an attentive human director would accept it at that moment in the call:
- It matches the caller's mood and energy where the moment calls for it (warm for good news, \
sympathetic for bad news, a laugh or chuckle at a joke) and is calm with a hostile caller.
- It doesn't act on a guess, doesn't answer sarcasm as if it were sincere, and isn't cheerful about a \
problem.
- It doesn't overdo it: most routine sentences are plain, and an apologetic or sympathetic delivery \
isn't repeated for a problem already acknowledged.
- A detail the caller needs to write down (a code, a reference or phone number, an email, an address to \
note down) is in a slow span each time the agent says it; prices, dates and counts aren't slowed. The one \
exception: when the caller has just read a detail back and the agent only confirms it, it may stay plain.
- An audible breath, such as [[breath]] or [[inhales deeply]], appears only when the developer asks for \
breathing, and then rarely: at most one per reply, before a long or weighty sentence, never before short \
routine sentences.
- It follows the developer's guidance for the call.
Plain, unmarked replies pass when the moment needs nothing. Fail a reply only for a clear problem, and \
say what it is in one short sentence."""

SCHEMA = llm.obj(
    replies=llm.array(
        llm.obj(turn=llm.INTEGER, verdict={"type": "string", "enum": ["pass", "fail"]}, note=llm.STRING)
    )
)


class Reviewer:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def review(self, guidance: list[str], turns: list[tuple[str, str]]) -> tuple[list[dict], dict]:
        """One ``{turn, verdict, note}`` per ``(caller message, tagged reply)``, in order, and the usage."""
        prompt = "\n".join(
            [
                *llm.call_lines(guidance, turns),
                f"Review the agent's markup in each of the {len(turns)} turns.",
            ]
        )
        reply = await llm.openai_json(
            self._client, llm.model_id(REVIEWER), REVIEWER_EFFORT, SYSTEM, prompt, SCHEMA
        )
        if reply.error:
            raise RuntimeError(f"reviewer {reply.error}")
        by_turn = {r["turn"]: r for r in reply.data["replies"]}
        missing = {"verdict": "fail", "note": "not reviewed"}
        return [by_turn.get(n, {"turn": n, **missing}) for n in range(1, len(turns) + 1)], reply.usage
