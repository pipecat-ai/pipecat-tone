"""The breath pass: where a call breathes when its policy asks it to. A call that asks to breathe takes these breaths
in place of its teacher's; a call taught without them gets a breath variant, its policy with the BREATHE preset and
breath offered where its TTS has one (mostly: some keep a sound list without it).

Which replies breathe is drawn at people's rate for their length: asked for a rate, the teacher breathes by its own
sense of where breaths belong, and the reviewer fails breaths it finds unneeded, both in long replies far more than
people do. So the teacher chooses only the sentence each breath starts, every other label is the call's own, and so
are the verdicts.
"""

import random
from dataclasses import replace

import anthropic
from pipecat.expressive.perception import format_perception
from pipecat.expressive.policy import ExpressionPolicy

from lab import llm
from lab.corpus.scenario import load
from lab.corpus.speech_breaths import TURN_LENGTHS, band
from pipecat_tone.tagger import BREATHE, LabelSpace, SentenceLabels
from pipecat_tone.tagger.labels import label_map
from pipecat_tone.tagger.markup import render_reply, same_words
from pipecat_tone.tagger.model import words

# Bumped when the pass changes, so older breaths can be told apart and redone.
BREATH_VERSION = 2
EFFORT = "medium"
# As often as v2's BREATHE specs offer breath in a sound list.
OFFERED_SHARE = 0.85

# How often people breathe in a turn of each of TURN_LENGTHS, before or inside a sentence: results/breaths/icsi.json.
REPLY_RATES = (0.045, 0.071, 0.145, 0.261, 0.412)

SYSTEM = """\
You direct how a voice agent's replies are spoken on a customer service phone call. A text-to-speech \
service (TTS) speaks the agent's words, which are fixed. The developer has asked for audible breaths. \
People breathe audibly in about one reply in five, more often in long ones, so the replies that breathe \
are chosen for you, and you choose only where in each the breath goes: the sentence it starts.

- Put it where a person on the phone would take a breath: before a new thought, a long or careful \
explanation, or hard news. A long sentence takes it more often than a short one, and a reply's first \
sentence no more often than the rest.
- Never choose a sentence that another sound starts.
- Choose from the call up to and including that reply. Never use what is said after it.

Return each reply you are asked about, numbered as given, with the number of the sentence the breath \
starts."""

SCHEMA = llm.obj(replies=llm.array(llm.obj(turn=llm.INTEGER, sentence=llm.INTEGER)))


def variant_policy(policy: ExpressionPolicy, tts: str, call_id: str) -> ExpressionPolicy:
    """The call's policy with BREATHE, and its sound list, if it has one, mostly offering breath."""
    sounds = policy.sounds
    names = label_map().sounds["breath"].get(tts) or []
    if sounds is not None and names and random.Random(f"breath-variant-{call_id}").random() < OFFERED_SHARE:
        sounds = sorted({*sounds, names[0]})
    return ExpressionPolicy(guidance=[*policy.guidance, BREATHE], delivery=policy.delivery, sounds=sounds)


def needs_breaths(call: dict) -> bool:
    """Whether a built call lacks a current breath pass."""
    return "turns" in call and (call.get("breaths") or {}).get("version") != BREATH_VERSION


def breathing_replies(call_id: str, turns: list[dict]) -> list[int]:
    """The replies that take a breath, each drawn at people's rate for its length, the same every time for a call."""
    draw = random.Random(f"breath-replies-{call_id}")
    return [
        t
        for t, turn in enumerate(turns)
        if draw.random() < REPLY_RATES[band(sum(map(words, turn["sentences"])), TURN_LENGTHS)]
    ]


def without_breaths(labels: list[SentenceLabels]) -> list[SentenceLabels]:
    """A reply's labels without the teacher's breaths, for the pass's to go in."""
    return [replace(lab, sound_before=None) if lab.sound_before == "breath" else lab for lab in labels]


def with_breaths(labels: list[SentenceLabels], chosen: list[int]) -> list[SentenceLabels]:
    """A reply's labels with a breath before each chosen sentence that has no sound before it."""
    return [
        replace(lab, sound_before="breath") if n in chosen and not lab.sound_before else lab
        for n, lab in enumerate(labels)
    ]


async def breathe_call(call: dict) -> dict:
    """The call with its breaths: the policy, each reply's breaths, markup, checks and verdicts."""
    from bench.format_check import check_reply
    from bench.profiles import tts_adapter

    scenario = load(call["scenario"])
    policy = (
        scenario.policy
        if BREATHE in scenario.policy.guidance
        else variant_policy(scenario.policy, scenario.tts, call["spec"]["id"])
    )
    space = LabelSpace.for_tts(scenario.tts, policy)
    record = {
        "version": BREATH_VERSION,
        "policy": {"guidance": policy.guidance, "delivery": policy.delivery, "sounds": policy.sounds},
        "offered": "breath" in space.sounds,
    }
    if not record["offered"]:
        return {**call, "breaths": record}

    turns = call["turns"]
    chosen: dict[int, list[int]] = {}
    if breathing := breathing_replies(call["spec"]["id"], turns):
        exchanges = [
            (f"{format_perception(st.heard)} {st.user}" if st.heard else st.user, t["sentences"])
            for st, t in zip(scenario.turns, turns)
        ]
        prompt = "\n".join(
            [
                f"TTS: {space.tts}",
                *llm.call_lines(policy.guidance, exchanges),
                f"These replies take one breath each: turns {', '.join(str(t + 1) for t in breathing)}. For each, "
                "give the sentence the breath starts.",
            ]
        )
        reply = await llm.claude_json(anthropic.AsyncAnthropic(), SYSTEM, prompt, SCHEMA, EFFORT)
        record["teacher"] = {"usage": reply.usage, "cost_usd": reply.cost_usd, "error": reply.error}
        for item in (reply.data or {}).get("replies", []):
            if item.get("turn", 0) - 1 in breathing:
                chosen[item["turn"] - 1] = [item.get("sentence", 0) - 1]

    adapter = tts_adapter(scenario.tts)
    variant = []
    for t, turn in enumerate(turns):
        original = without_breaths([SentenceLabels.from_dict(d) for d in turn["labels"]])
        labels = with_breaths(original, chosen.get(t, []))
        markup = render_reply(space, turn["sentences"], labels)
        variant.append(
            {
                "breaths": [n for n, (a, b) in enumerate(zip(original, labels)) if a != b],
                "markup": markup,
                "format_issues": check_reply(markup, adapter, policy, slow=scenario.slow).issues,
                "same_words": same_words(" ".join(turn["sentences"]), markup, adapter),
            }
        )
    record["turns"] = [
        {**v, "verdict": t.get("verdict"), "note": t.get("note")} for v, t in zip(variant, turns)
    ]
    return {**call, "breaths": record}
