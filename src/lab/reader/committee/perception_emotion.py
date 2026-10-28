"""The emotion question under the Realtime Context Standard's perception (perception 0.1): the emotion a turn to a voice
agent shows and its intensity, the act toward the agent and its intensity, and the backchannel flag, read from the
turn's words and the agent's line before it, as the reader reads them. Its answers are written as line tags 0.1's, one
act per target, so they join the label sets as the act and emotion question's do."""

import json

from lab.reader.committee import act_emotion, clients, guide, user_message
from lab.reader.data import previous_words
from lab.reader.schemas import ACTS, EMOTIONS, INTENSITIES, ActTag, LineInput, LineTags

GUIDE = "perception-guide-0.1.md"
SETTING = "a support call: a person talking to an AI voice agent"


def view(inp: LineInput) -> dict:
    """The turn as the committee sees it under the standard, as unsorted JSON: the setting, the agent's line before it
    in its words only, and the turn."""
    before = previous_words(inp.previous)
    return {"setting": SETTING, **({"agent_line_before": before} if before else {}), "turn": inp.text}


# Worked examples A and B see: (the agent's line before or None, the turn, emotion, intensity, act, act intensity,
# backchannel).
EXAMPLES = (
    (
        "Can I take the order number?",
        "Yeah, it's four eight two one three.",
        "neutral",
        "low",
        "none",
        "low",
        False,
    ),
    (
        "Your parcel was delivered this morning.",
        "Oh, brilliant, thanks so much.",
        "happy",
        "medium",
        "thanks",
        "low",
        False,
    ),
    (
        "I'm afraid the earliest visit is next Thursday.",
        "Oh, perfect. Just perfect.",
        "angry",
        "medium",
        "none",
        "low",
        False,
    ),
    (
        "It's still being processed, so it hasn't shipped yet.",
        "Great, love that. So it'll miss the birthday.",
        "angry",
        "medium",
        "none",
        "low",
        False,
    ),
    (
        "I've added a note to your account.",
        "Wow, a note. That fixes everything.",
        "contemptuous",
        "medium",
        "insult",
        "low",
        False,
    ),
    (
        "It looks like the card was used abroad this morning.",
        "What? I haven't left the country. Is my money gone?",
        "afraid",
        "high",
        "none",
        "low",
        False,
    ),
    (
        "I'm sorry, the concert has been cancelled.",
        "Oh no. I was really looking forward to that.",
        "sad",
        "medium",
        "none",
        "low",
        False,
    ),
    ("So I'll set that up for you now.", "mm-hm", "neutral", "low", "none", "low", True),
    (
        "You'll need to call the bank yourself.",
        "Seriously? You lot are useless.",
        "angry",
        "medium",
        "insult",
        "medium",
        False,
    ),
    (
        "Is there anything else I can help with?",
        "No, that's everything. Thanks, bye.",
        "neutral",
        "low",
        "thanks",
        "low",
        False,
    ),
    (
        "I've reset it from here, so it should work now.",
        "Ha, so it was just off and on again. Classic.",
        "amused",
        "low",
        "none",
        "low",
        False,
    ),
    (None, "hi um i'm calling about my bill", "neutral", "low", "none", "low", False),
)


def example_input(e: tuple) -> LineInput:
    prev, text = e[0], e[1]
    return LineInput(
        speaker="player",
        speaker_kind="player",
        to="Agent",
        targets=["Agent"],
        text=text,
        previous=f"Agent: {prev}" if prev else None,
    )


def _answer(e: tuple) -> dict:
    _, _, emotion, intensity, act, act_intensity, backchannel = e
    return {
        "emotion": emotion,
        "emotion_intensity": intensity,
        "act": act,
        "act_intensity": act_intensity,
        "backchannel": backchannel,
    }


def system_prompt() -> str:
    """The guide, then worked examples as input and answer pairs. Fixed text, so it caches."""
    shots = "\n\n".join(
        f"{user_message(view(example_input(e)))}\nAnswer:\n{json.dumps(_answer(e))}" for e in EXAMPLES
    )
    return (
        "You label what people say to AI voice agents on support calls, for a reader whose readings tell the agent's "
        "LLM how each turn was meant. Follow this guide exactly.\n\n"
        + guide(GUIDE)
        + "\n\n## Worked examples\n\n"
        + shots
        + "\n\nAnswer with the JSON object only: the emotion `turn` shows and its intensity, read after "
        "`agent_line_before` when there is one, the act toward the agent and its intensity, and the backchannel flag."
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["emotion", "emotion_intensity", "act", "act_intensity", "backchannel"],
    "properties": {
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "emotion_intensity": {"type": "string", "enum": list(INTENSITIES)},
        "act": {"type": "string", "enum": list(ACTS)},
        "act_intensity": {"type": "string", "enum": list(INTENSITIES)},
        "backchannel": {"type": "boolean"},
    },
}


def tags_from_answer(answer: dict, inp: LineInput) -> LineTags:
    """A member's answer as LineTags: the act toward every target (the agent), `none` and `neutral` at low intensity,
    and a backchannel that carries an act is not a backchannel."""
    act, emotion = answer["act"], answer["emotion"]
    tag = ActTag(act=act, intensity="low" if act == "none" else answer["act_intensity"])
    return LineTags(
        emotion=emotion,
        emotion_intensity="low" if emotion == "neutral" else answer["emotion_intensity"],
        acts={t: tag for t in inp.targets},
        backchannel=bool(answer["backchannel"]) and act == "none",
    )


FORM = clients.Form(view, OUTPUT_SCHEMA, "perception_emotion", 4_000, tags_from_answer)


# --- D ---------------------------------------------------------------------------------------------------------------

D_PREFACE = (
    "One turn a person says to an AI voice agent on a support call, to be labelled for the feeling its words show and "
    "what it does to the agent. The state below is JSON; the questions refer to its fields by name in backticks.\n"
)
EMOTION_CRITERIA = {
    "neutral": "No clear feeling in the words: plain information, routine answers, questions and requests. Neutral "
    "means the words show none, not that the person feels none.",
    "happy": "Pleasure, warmth, relief, gratitude or excitement, meant.",
    "amused": "Finds something funny: joking, laughing at themself, wry, or teasing in fun.",
    "angry": "Irritation up to rage, at someone or something, said straight or through sarcasm.",
    "sad": "Sorrow, disappointment, regret or a weary letdown.",
    "afraid": "Fear, worry, nervousness or alarm.",
    "surprised": "Astonishment, disbelief or a start, good or bad.",
    "disgusted": "Revulsion at something physical or moral.",
    "contemptuous": "Looking down on someone: scorn, disdain, sneering, or sarcasm that mocks the agent or their help.",
}
ACT_CRITERIA = {
    **act_emotion.ACT_CRITERIA,
    "insult": act_emotion.ACT_CRITERIA["insult"]
    + " Sarcasm that mocks the agent or their help is an insult.",
    "accusation": "Says the agent, or the company they speak for, did something wrong, or asks in a way that presumes "
    "it.",
    "none": "Nothing social toward the agent: plain information, answers, questions asking for information, complaints "
    "about the situation, and sarcasm about the situation or the person's own luck.",
}


def d_request(inp: LineInput, model: str) -> dict:
    """One Decisions request per turn; the act is asked as act_0, so ``act_emotion.parse_d`` reads the answer."""

    def levels(texts):
        return [{"label": INTENSITIES[i], "description": d} for i, d in enumerate(texts)]

    questions = [
        {
            "type": "choice",
            "name": "emotion",
            "instructions": "Which emotion do the words of `turn` show, read after `agent_line_before` when there is "
            "one? Go by what the words say; do not guess feelings they do not show. A sarcastic turn shows the "
            "feeling it means, never the pleasure its words pretend.",
            "choices": [{"value": k, "description": v} for k, v in EMOTION_CRITERIA.items()],
        },
        {
            "type": "score",
            "name": "emotion_intensity",
            "instructions": "How strongly does `turn` show that emotion?",
            "levels": levels(act_emotion.EMOTION_LEVELS),
        },
        {
            "type": "predicate",
            "name": "backchannel",
            "instructions": "Is `turn` only a short sound of attention or agreement, such as mm, uh-huh, yeah, right "
            "or okay, that does not take a turn in the conversation?",
        },
        {
            "type": "choice",
            "name": "act_0",
            "instructions": "What does `turn` do to the agent? Most turns do nothing social to the agent: none.",
            "choices": [{"value": k, "description": v} for k, v in ACT_CRITERIA.items()],
        },
        {
            "type": "score",
            "name": "act_intensity_0",
            "instructions": "How strong is what `turn` does to the agent?",
            "levels": levels(act_emotion.ACT_LEVELS),
        },
    ]
    state = json.dumps(view(inp), indent=1, sort_keys=True, ensure_ascii=False)
    return {"model": model, "input": D_PREFACE + state, "questions": questions}


def parse_d(data: dict, inp: LineInput) -> tuple[LineTags | None, str | None, dict]:
    """A Decisions response as (LineTags, error, usage), the act toward every target."""
    one = inp.model_copy(update={"targets": inp.targets[:1]})
    tags, error, usage = act_emotion.parse_d(data, one)
    if tags is None:
        return None, error, usage
    act = tags.acts[inp.targets[0]]
    return tags.model_copy(update={"acts": {t: act for t in inp.targets}}), None, usage
