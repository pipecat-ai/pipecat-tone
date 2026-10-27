"""The sarcasm question (line tags 0.3): whether a line means something other than, usually the opposite of, what its
words say, read with the line before it."""

import json

from lab.reader.committee import clients, guide_section, majority, user_message, view
from lab.reader.schemas import LineInput
from lab.reader.scoring import f1_sarcastic

GUIDE = "line-tags-guide-0.3.md"

# The worked examples A and B see: (id, previous line, speaker, said to, line, sarcastic).
EXAMPLES = (
    (
        "s01-thanks-after-trouble",
        "Mara: Sorry, we're out of rooms tonight.",
        "player",
        "Mara",
        "oh great, thanks a lot",
        True,
    ),
    (
        "s02-thanks-after-help",
        "Mara: I fixed your sword, no charge.",
        "player",
        "Mara",
        "oh great, thanks a lot",
        False,
    ),
    (
        "s03-mock-agreement",
        "Tomas: I'll pay you back tomorrow, I swear.",
        "player",
        "Tomas",
        "Sure you will.",
        True,
    ),
    (
        "s05-mock-impressed",
        "Wren: I locked the keys inside the cart.",
        "player",
        "Wren",
        "Wow. Genius.",
        True,
    ),
    ("s06-weather", "Mara: Storm's rolling in again.", "player", "Mara", "Lovely. Rain again.", True),
    (
        "s08-friendly",
        "Wren: I beat the bear with my bare hands, you know.",
        "player",
        "Wren",
        "Oh yes, a real hero, you are. Ha!",
        True,
    ),
    ("s09-tease-not-sarcasm", "Wren: Watch this throw!", "player", "Wren", "Missed again, ha!", False),
    (
        "s10-polite-fiction",
        "Mara: Could you watch the stall for an hour?",
        "player",
        "Mara",
        "I'd love to, but I can't today.",
        False,
    ),
    ("s11-plain-insult", "Tomas: I lost the map.", "player", "Tomas", "You useless lump.", False),
    (
        "s15-npc-mock",
        "player: I'm the best swordsman in the realm.",
        "Tomas",
        "player",
        "Of course you are. And I'm the queen.",
        True,
    ),
)


def example_input(e: tuple) -> LineInput:
    _, prev, speaker, to, text, _ = e
    return LineInput(
        speaker=speaker,
        speaker_kind="player" if speaker == "player" else "npc",
        to=to,
        targets=[to],
        text=text,
        previous=prev,
    )


def system_prompt() -> str:
    shots = "\n\n".join(
        f"{user_message(view(example_input(e)))}\nAnswer:\n{json.dumps({'sarcastic': e[5]})}"
        for e in EXAMPLES
    )
    return (
        "You label lines of dialogue from games for a classifier that tells a game whether a line is sarcastic. "
        "Follow this guide exactly.\n\n"
        + guide_section(GUIDE, "## Sarcasm")
        + "\n\n## Worked examples\n\n"
        + shots
        + "\n\nAnswer with the JSON object only: whether `line` is sarcastic, read after `previous_line` when there is one. "
        "`speaker`, `said_to` and `people_in_scene` say who is talking to whom; `the player` is the person playing the game."
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sarcastic"],
    "properties": {"sarcastic": {"type": "boolean"}},
}


def label_from_answer(answer: dict) -> bool:
    v = answer["sarcastic"]
    if not isinstance(v, bool):
        raise ValueError(f"not a boolean: {str(v)[:40]}")
    return v


FORM = clients.Form(view, OUTPUT_SCHEMA, "sarcasm", 2_000, lambda answer, _: label_from_answer(answer))


# --- D ---------------------------------------------------------------------------------------------------------------

D_PREFACE = (
    "One line of dialogue from a video game, to be labelled for whether it is sarcastic. The state below is JSON; "
    "the question refers to its fields by name in backticks.\n"
)
D_PREDICATE = (
    "Is `line` sarcastic: does it mean something other than, usually the opposite of, what its words say, read "
    "after `previous_line` when there is one? Mock praise, mock thanks, mock agreement, mock sympathy, pretending "
    "to be impressed, and irony about a situation or about oneself all count, whether hostile or friendly. A "
    "joke, a tease or an exaggeration that means what it says, a polite fiction, a plain insult and a plain "
    "rhetorical question do not. If the words could easily be meant, it is not sarcastic."
)


def d_request(inp: LineInput, model: str) -> dict:
    return {
        "model": model,
        "input": D_PREFACE + json.dumps(view(inp), indent=1, sort_keys=True, ensure_ascii=False),
        "questions": [{"type": "predicate", "name": "sarcastic", "instructions": D_PREDICATE}],
    }


def parse_d(data: dict, inp: LineInput) -> tuple[bool | None, str | None, dict, float | None]:
    """(vote, error, usage, probability): the vote is the probability at or above one half."""
    answers = data.get("answers") or []
    for a in answers:
        if (
            a.get("name") == "sarcastic"
            and a.get("type") == "predicate"
            and isinstance(a.get("probability"), (int, float))
        ):
            p = float(a["probability"])
            return p >= 0.5, None, data.get("usage") or {}, p
    refused = any(a.get("type") != "predicate" for a in answers)
    return None, clients.REFUSAL if refused else clients.INVALID, data.get("usage") or {}, None


# --- the majority ----------------------------------------------------------------------------------------------------


def vote(labels: dict[str, bool | None]) -> dict:
    """The majority of the members that answered; `agree` is 0 without one."""
    votes = {m: v for m, v in labels.items() if v is not None}
    label, agree = majority(votes)
    return {"label": label, "votes": votes, "agree": agree if label is not None else 0}


def committee_level(rows: list[dict]) -> dict:
    """Labellers A and B on the same lines: F1 of the sarcastic class with each as gold, averaged."""
    both = [
        (r["sarcasm"]["votes"]["a"], r["sarcasm"]["votes"]["b"])
        for r in rows
        if r["sarcasm"]["label"] is not None and "a" in r["sarcasm"]["votes"] and "b" in r["sarcasm"]["votes"]
    ]
    a = [int(x) for x, _ in both]
    b = [int(y) for _, y in both]
    return {
        "lines": len(both),
        "f1_a_as_gold": round(f1_sarcastic(a, b), 4),
        "f1_b_as_gold": round(f1_sarcastic(b, a), 4),
        "f1": round((f1_sarcastic(a, b) + f1_sarcastic(b, a)) / 2, 4),
        "agreement": round(sum(x == y for x, y in both) / max(1, len(both)), 4),
    }
