"""The act and emotion question (line tags 0.1): the speaker's emotion and its intensity, an act and its intensity
toward each person present, and the backchannel flag."""

import json
from dataclasses import dataclass
from functools import partial

from lab.reader.committee import clients, guide, majority, user_message, view, worked_examples
from lab.reader.schemas import ACTS, EMOTIONS, INTENSITIES, PLAYER, ActTag, LineInput, LineTagRecord, LineTags

GUIDE = "line-tags-guide-0.1.md"

# Worked examples A and B see.
FEW_SHOT_IDS = (
    "guide-01-insult",
    "guide-06-accusation-third-party",
    "guide-09-threat-in-jest",
    "guide-10-insult-not-tease",
    "guide-12-demand-low",
    "guide-15-refusal",
    "guide-21-warning-room",
    "guide-22-warning-cut-off",
    "guide-25-backchannel",
    "guide-27-impatient",
    "guide-29-sarcasm",
    "guide-30-comparison",
)


# --- what A and B read -----------------------------------------------------------------------------------------------


def _answer_json(rec: LineTagRecord) -> dict:
    t = rec.tags
    return {
        "emotion": t.emotion,
        "emotion_intensity": t.emotion_intensity,
        "acts": [{"target": n, "act": a.act, "intensity": a.intensity} for n, a in t.acts.items()],
        "backchannel": t.backchannel,
    }


def system_prompt() -> str:
    """The guide, then worked examples as input and answer pairs. Fixed text, so it caches."""
    examples = worked_examples("0.1")
    shots = "\n\n".join(
        f"{user_message(view(examples[i].input, previous_first=False))}\n"
        f"Answer:\n{json.dumps(_answer_json(examples[i]), ensure_ascii=False)}"
        for i in FEW_SHOT_IDS
    )
    return (
        "You label lines of dialogue from games for a classifier that drives characters' faces and feelings. "
        "Follow this guide exactly.\n\n"
        + guide(GUIDE)
        + "\n\n## Worked examples\n\n"
        + shots
        + "\n\nAnswer with the JSON object only: the speaker's emotion and its intensity, one act with an intensity for "
        "every person in `people_in_scene` (use their names exactly), and the backchannel flag."
    )


# The answer's strict schema, the same for every line so it never breaks the prompt cache; `tags_from_answer` checks the
# names.
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["emotion", "emotion_intensity", "acts", "backchannel"],
    "properties": {
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "emotion_intensity": {"type": "string", "enum": list(INTENSITIES)},
        "acts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "act", "intensity"],
                "properties": {
                    "target": {"type": "string"},
                    "act": {"type": "string", "enum": list(ACTS)},
                    "intensity": {"type": "string", "enum": list(INTENSITIES)},
                },
            },
        },
        "backchannel": {"type": "boolean"},
    },
}


def tags_from_answer(answer: dict, inp: LineInput) -> LineTags:
    """A member's JSON answer as LineTags: one act per target (first answer wins, missing targets fail), `none` and
    `neutral` at low intensity, and a backchannel that carries an act is not a backchannel."""
    acts: dict[str, ActTag] = {}
    by_lower = {t.lower(): t for t in inp.targets}
    by_lower.update({"the player": PLAYER, "you": inp.to} if inp.to else {"the player": PLAYER})
    for a in answer["acts"]:
        a = {**a, "target": by_lower.get(a["target"].strip().lower(), a["target"])}
        if a["target"] in inp.targets and a["target"] not in acts:
            intensity = "low" if a["act"] == "none" else a["intensity"]
            acts[a["target"]] = ActTag(act=a["act"], intensity=intensity)
    missing = [t for t in inp.targets if t not in acts]
    if missing:
        raise ValueError(f"no act for {missing}")
    emotion = answer["emotion"]
    backchannel = bool(answer["backchannel"]) and all(a.act == "none" for a in acts.values())
    return LineTags(
        emotion=emotion,
        emotion_intensity="low" if emotion == "neutral" else answer["emotion_intensity"],
        acts=acts,
        backchannel=backchannel,
    )


FORM = clients.Form(partial(view, previous_first=False), OUTPUT_SCHEMA, "line_tags", 4_000, tags_from_answer)


# --- D: the definitions as Decisions questions -----------------------------------------------------------------------

EMOTION_CRITERIA = {
    "neutral": "No clear feeling: plain information, routine questions and requests. Sympathy alone is neutral.",
    "happy": "Pleasure, warmth, relief, gratitude or excitement.",
    "amused": "Finds something funny: joking, teasing, wry.",
    "angry": "Irritation up to rage, at someone or something.",
    "sad": "Sorrow, disappointment, regret or loneliness.",
    "afraid": "Fear, worry, nervousness or alarm.",
    "surprised": "Astonishment, disbelief or a start, good or bad.",
    "disgusted": "Revulsion at something physical or moral.",
    "contemptuous": "Looking down on someone: scorn, disdain, sneering.",
}

ACT_CRITERIA = {
    "insult": "Demeans them: name-calling, mockery of their worth, looks, wits or work. Includes sarcasm and praise that is really mockery, and rhetorical put-downs like 'Are you stupid?'.",
    "threat": "Says harm will come to them, open or veiled, now or if they do something.",
    "accusation": "Says they did something wrong, or asks in a way that presumes they did.",
    "tease": "A playful jab that the words themselves mark as play: a laugh, absurd exaggeration, a joking tag. Without such a marker a jab is an insult.",
    "demand": "Orders them to do or stop something, without asking politely.",
    "dismiss": "Brushes them off: ends or refuses the exchange, or says they do not matter.",
    "refusal": "Declines something they asked for or offered.",
    "request": "Asks them for a favour, an item, help or an action, with a softener such as please or could you.",
    "thanks": "Thanks them.",
    "praise": "Says something sincerely good about them or their work.",
    "apology": "Says sorry to them, or owns a fault toward them.",
    "comfort": "Reassures, consoles or encourages them.",
    "warning": "Tells them of a danger to them, for their own good.",
    "none": "Nothing social toward them: they are a bystander the line does not address, name or talk about, or the line is a greeting, small talk, plain information or a question asking for information.",
}

ACT_RULES = [
    "Label the words, not the friendship: you do not know how these people feel about each other.",
    "The line touches the person it was said to, anyone it names or points at, and anyone present it talks about. "
    "A line about a present person acts on that person, not on the listener. Everyone else is a bystander: none.",
    "A line to a group or the room ('you lot', 'everyone', a shouted warning) acts on every person in that group.",
    "If the line does several things to this person, pick the weightiest in this order: threat, insult, accusation, "
    "demand, dismiss, refusal, apology, thanks, comfort, praise, warning, tease, request, none.",
    "Player lines may come from speech to text with no punctuation or misheard words: read what was meant.",
]

EMOTION_LEVELS = [
    "No feeling shown, or only a trace of it",
    "The feeling is clearly felt",
    "The feeling is overwhelming: shouting, sobbing, terror or rapture",
]
ACT_LEVELS = [
    "Mild, softened or only implied",
    "Clear and direct",
    "Strong words, swearing, violence, or shouted",
]

D_PREFACE = (
    "One line of dialogue from a video game, to be labelled for the feeling its speaker shows and what it does to "
    "each person present. The state below is JSON; the questions refer to its fields by name in backticks.\n"
)
# D's own notes beside the shared definitions.
D_ACT_RULES_V2 = [
    "Most lines do nothing social to most people. Plain replies, okays, agreement, small talk, questions, complaints "
    "about things rather than people, and talk about the game are none.",
    "Pick comfort, praise, thanks, dismiss or insult only when the words plainly do that to this person.",
]
D_EMOTION_V2 = {
    "contemptuous": EMOTION_CRITERIA["contemptuous"]
    + " Sarcasm and mock praise aimed at someone are contemptuous, not "
    "amused or happy, and not angry unless the words also show open irritation.",
    "amused": EMOTION_CRITERIA["amused"] + " Not sarcasm that mocks someone: that is contemptuous.",
}
D_ACT_RULES_V3 = [
    "A put-down, accusation or threat aimed at someone else does nothing hostile to the person it was said to: it acts on "
    "the one it is aimed at.",
]


def _target_label(name: str) -> str:
    return "the player" if name == PLAYER else name


def _questions(inp: LineInput) -> dict:
    """The definitions as questions, with D's notes: the emotion, its intensity, the backchannel, and an act and its
    intensity per person."""
    questions: dict = {
        "emotion": {
            "type": "choice",
            "instructions": {
                "question": "Which emotion does the speaker show in `line`, as the words show it?",
                "not_for": "Do not guess feelings the words do not show.",
            },
            "criteria": {**EMOTION_CRITERIA, **D_EMOTION_V2},
        },
        "emotion_intensity": {
            "type": "score",
            "instructions": "How strongly does `line` show the speaker's emotion?",
            "criteria": EMOTION_LEVELS,
        },
        "backchannel": {
            "type": "noul",
            "instructions": "Is `line` only a short sound of attention or agreement, such as mm, uh-huh, "
            "yeah, right or go on, that does not take a turn in the conversation?",
        },
    }
    for i, target in enumerate(inp.targets):
        who = _target_label(target)
        questions[f"act_{i}"] = {
            "type": "choice",
            "instructions": {
                "question": f"What does `line` do to {who}?",
                "rules": [*ACT_RULES, *D_ACT_RULES_V2, *D_ACT_RULES_V3],
            },
            "criteria": ACT_CRITERIA,
        }
        questions[f"act_intensity_{i}"] = {
            "type": "score",
            "instructions": f"How strong is what `line` does to {who}?",
            "criteria": ACT_LEVELS,
        }
    return questions


def _instructions(ins) -> str:
    if isinstance(ins, str):
        return ins
    text = ins["question"]
    if ins.get("not_for"):
        text += " " + ins["not_for"]
    if ins.get("rules"):
        text += "\nRules:\n" + "\n".join(f"- {r}" for r in ins["rules"])
    return text


def d_request(inp: LineInput, model: str) -> dict:
    """One Decisions request per line."""
    questions = []
    for name, q in _questions(inp).items():
        item = {
            "type": {"choice": "choice", "score": "score", "noul": "predicate"}[q["type"]],
            "name": name,
            "instructions": _instructions(q["instructions"]),
        }
        if q["type"] == "choice":
            item["choices"] = [{"value": k, "description": v} for k, v in q["criteria"].items()]
        elif q["type"] == "score":
            item["levels"] = [
                {"label": INTENSITIES[i], "description": d} for i, d in enumerate(q["criteria"])
            ]
        questions.append(item)
    state = json.dumps(view(inp), indent=1, sort_keys=True, ensure_ascii=False)
    return {"model": model, "input": D_PREFACE + state, "questions": questions}


def _level(score: float) -> str:
    return INTENSITIES[min(2, max(0, round(score)))]


def parse_d(data: dict, inp: LineInput) -> tuple[LineTags | None, str | None, dict]:
    """A Decisions response as (LineTags, error, usage). A refused question means no D vote on the line."""
    ans, refused = {}, False
    for a in data.get("answers") or []:
        t = a.get("type")
        if t == "choice":
            ans[a["name"]] = {
                "choice": a["choice"],
                "probabilities": {p["value"]: p["probability"] for p in a.get("probabilities") or []},
            }
        elif t == "score":
            ans[a["name"]] = {"score": a["score"]}
        elif t == "predicate":
            ans[a["name"]] = {"noul": a["probability"]}
        else:
            refused = True
    try:
        acts = {}
        for i, target in enumerate(inp.targets):
            a = ans[f"act_{i}"]
            act = a["choice"]
            intensity = "low" if act == "none" else _level(ans[f"act_intensity_{i}"]["score"])
            acts[target] = ActTag(act=act, intensity=intensity, confidence=round(a["probabilities"][act], 4))
        emo = ans["emotion"]["choice"]
        backchannel = ans["backchannel"]["noul"] > 0.5 and all(a.act == "none" for a in acts.values())
        tags = LineTags(
            emotion=emo,
            emotion_intensity="low" if emo == "neutral" else _level(ans["emotion_intensity"]["score"]),
            emotion_confidence=round(ans["emotion"]["probabilities"][emo], 4),
            acts=acts,
            backchannel=backchannel,
        )
        return tags, None, data.get("usage") or {}
    except (KeyError, TypeError, ValueError):
        return None, clients.REFUSAL if refused else clients.INVALID, data.get("usage") or {}


# --- majority of three -----------------------------------------------------------------------------------------------


@dataclass
class HeadVote:
    votes: dict[str, str]
    label: str | None  # None without a majority
    intensity: str | None = None
    agree: int = 0


@dataclass
class LineVote:
    emotion: HeadVote
    acts: dict[str, HeadVote]
    backchannel: HeadVote

    @property
    def complete(self) -> bool:
        return (
            self.emotion.label is not None
            and self.backchannel.label is not None
            and all(h.label is not None for h in self.acts.values())
        )


def _intensity(values: list[str]) -> str:
    """Median of the matching votes' intensities; the lower one when two differ."""
    ranks = sorted(INTENSITIES.index(v) for v in values)
    return INTENSITIES[ranks[(len(ranks) - 1) // 2]]


def _head(votes: dict[str, str], intensities: dict[str, str] | None = None) -> HeadVote:
    label, agree = majority(votes)
    if label is None:
        return HeadVote(votes=votes, label=None, agree=agree)
    inten = _intensity([intensities[m] for m, v in votes.items() if v == label]) if intensities else None
    return HeadVote(votes=votes, label=label, intensity=inten, agree=agree)


def vote(inp: LineInput, labels: dict[str, LineTags | None]) -> LineVote:
    """The majority per head over the members that answered."""
    got = {m: t for m, t in labels.items() if t is not None}
    emotion = _head({m: t.emotion for m, t in got.items()}, {m: t.emotion_intensity for m, t in got.items()})
    acts = {
        tg: _head(
            {m: t.acts[tg].act for m, t in got.items()}, {m: t.acts[tg].intensity for m, t in got.items()}
        )
        for tg in inp.targets
    }
    back = _head({m: str(t.backchannel).lower() for m, t in got.items()})
    return LineVote(emotion=emotion, acts=acts, backchannel=back)
