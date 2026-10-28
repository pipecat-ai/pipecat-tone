"""The sarcasm question under the Realtime Context Standard's perception (perception 0.1): whether a turn to a voice
agent means something other than, usually the opposite of, what its words say, read with the agent's line before it
as context, not as the turn."""

import json

from lab.reader.committee import clients, guide_section, user_message
from lab.reader.committee.perception_emotion import GUIDE, example_input, view
from lab.reader.committee.sarcasm import OUTPUT_SCHEMA, committee_level, label_from_answer, parse_d, vote  # noqa: F401

# The worked examples A and B see: (the agent's line before or None, the turn, sarcastic).
EXAMPLES = (
    (
        "Your package is out for delivery and should arrive by eight tonight.",
        "Oh great, thanks so much.",
        False,
    ),
    ("I'm afraid the earliest technician visit is next Thursday.", "Oh, perfect.", True),
    (
        "Your order is packed and waiting for the courier to collect it.",
        "Fantastic, so it's sitting in a box while I wait. Love that.",
        True,
    ),
    (
        "Can you confirm the email address on the account?",
        "Sure, it's the same one I've given you three times now. Love doing this.",
        True,
    ),
    ("Happy to help! I've added a note to your account.", "Wow, a note. That changes everything.", True),
    (
        "The refund was approved and will reach your card in three to five days.",
        "Okay, perfect, I'm not in a hurry.",
        False,
    ),
    (
        "It's still being processed, so it hasn't shipped yet.",
        "Oh no, so it won't be here by Friday. That's really annoying.",
        False,
    ),
    ("I'm sorry, I can't waive that fee.", "Right, of course you can't.", True),
    (None, "yeah because that worked so well last time", True),
    (
        "I've reset the router from here, so it should come back on in a minute.",
        "Oh nice, that was quick, thank you.",
        False,
    ),
    (
        "Is there anything else I can help you with today?",
        "No, you've been a huge help. I still don't have a working phone, but sure.",
        True,
    ),
    ("You'll get an email with a link to reschedule.", "Great, thanks, I'll look out for it.", False),
)


def system_prompt() -> str:
    shots = "\n\n".join(
        f"{user_message(view(example_input(e)))}\nAnswer:\n{json.dumps({'sarcastic': e[2]})}"
        for e in EXAMPLES
    )
    return (
        "You label what people say to AI voice agents on support calls, for a reader whose readings tell the agent's "
        "LLM whether a turn is sarcastic. Follow this guide exactly.\n\n"
        + guide_section(GUIDE, "## The axioms")
        + "\n\n"
        + guide_section(GUIDE, "## Sarcasm")
        + "\n\n## Worked examples\n\n"
        + shots
        + "\n\nAnswer with the JSON object only: whether `turn` is sarcastic, read after `agent_line_before` when there "
        "is one."
    )


FORM = clients.Form(
    view, OUTPUT_SCHEMA, "perception_sarcasm", 2_000, lambda answer, _: label_from_answer(answer)
)


# --- D ---------------------------------------------------------------------------------------------------------------

D_PREFACE = (
    "One turn a person says to an AI voice agent on a support call, to be labelled for whether it is sarcastic. The "
    "state below is JSON; the question refers to its fields by name in backticks.\n"
)
D_PREDICATE = (
    "Is `turn` sarcastic: does it mean something other than, usually the opposite of, what its words say, read after "
    "`agent_line_before` when there is one? Mock pleasure, mock thanks, mock praise, mock agreement, mock sympathy, "
    "pretending to be impressed, and irony about the situation or about oneself all count. Words that are sarcastic "
    "on their own, such as pleased or grateful words said together with the bad consequence or the complaint itself, "
    "stay sarcastic whatever the agent said before them, a plain update or a kind word included. A short pleased "
    "answer with nothing of its own is sarcastic only when the agent's line brings something the person would plainly "
    "not welcome. A plain complaint, taking a downside in good spirit, thanks after help, a joke that means what it "
    "says, a polite fiction and a plain insult are not. If the words could easily be meant, it is not sarcastic."
)


def d_request(inp, model: str) -> dict:
    return {
        "model": model,
        "input": D_PREFACE + json.dumps(view(inp), indent=1, sort_keys=True, ensure_ascii=False),
        "questions": [{"type": "predicate", "name": "sarcastic", "instructions": D_PREDICATE}],
    }
