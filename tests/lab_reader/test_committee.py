"""The committee asks exactly what labelled rc1's lines."""

import hashlib
import json

import pytest

from lab.reader.committee import act_emotion, clients, move, sarcasm
from lab.reader.schemas import LineInput

QUESTIONS = {"committee": act_emotion, "exchange": move, "sarcasm": sarcasm}

# The system prompts' sha256, as recorded in the summaries of the label sets rc1 was trained and tested on.
RC1_PROMPTS = {
    "committee": "386663145c439eb276c7ed16f41fdc4550d3468a753d7eac64475e12af6cd1e2",
    "exchange": "2996c87b935a35c69e97bcbd618b4e8b18a0f326b548d287d10241f05c6440ea",
    "sarcasm": "f388f9a3b4cb8012a956daf2b4c9b9cfb30a0ca4c531538fa55970952733284c",
}

LINES = [
    dict(
        speaker="player",
        speaker_kind="player",
        to="Ava",
        targets=["Ava"],
        text="Oh great, another hold. Love that.",
        previous="Ava: I'll need to put you on hold for a moment.",
    ),
    dict(
        speaker="Ava",
        speaker_kind="npc",
        to="player",
        targets=["player"],
        text="Can I take the order number?",
    ),
    dict(
        speaker="player",
        speaker_kind="player",
        to=None,
        targets=["Ava"],
        text="um yeah it's four four two",
        previous="Ava: What's the number?",
    ),
]

# Each member's requests for LINES, as sent: checked equal, byte for byte, to the requests that labelled rc1's
# data.
RC1_REQUESTS = {
    "committee": {
        "a": "a2f0ed6dd0716f32e0d5a0c527aaef1a64a8c3f5cfe1566d1334174708965d4e",
        "b": "1d84d43c99c4f56ac5342509688de43a4784167f7a276e244fce2232ec2fdb3b",
        "d": "5e1e261e3223d2fe4676d0eb62e6787c6349042c7c322f9ab0bf0d7b0349b422",
    },
    "exchange": {
        "a": "5d54026f4d001957e693dabfd98ca521ff4b9c837ee7ef5fb8feba52bad7fbfa",
        "b": "cc2c07ab9f445ed3e41ada9f1a2c9ee8a3d8c3553ff72c008acfefbfdb7364a7",
        "d": "4c7a3fc2171af63bd142395b021a7ff524f0cf58561cd7a8cf9153f98f4e6af5",
    },
    "sarcasm": {
        "a": "68785e4d82b9c98e96c925599e8bef0d8c21ad01e7cd23a736717eca9478d9ec",
        "b": "e2d082784cda042407c165fc3f0f85445b5332efc02c0322113ba19ef5931d7b",
        "d": "20cea62fbf0167d5262cc9c1a63cd2205663724bd79c237999a0087905476f69",
    },
}


def sha256(*texts: str) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def test_the_system_prompts_are_rc1s():
    assert {q: sha256(m.system_prompt()) for q, m in QUESTIONS.items()} == RC1_PROMPTS


@pytest.mark.parametrize("question", QUESTIONS)
def test_each_member_asks_what_it_asked_for_rc1(question):
    m = QUESTIONS[question]
    system, lines = m.system_prompt(), [LineInput(**kw) for kw in LINES]
    requests = {
        "a": [clients.a_request(m.FORM, "line-1", inp, system, "model-a") for inp in lines],
        "b": [clients.b_request(m.FORM, inp, system, "model-b") for inp in lines],
        "d": [m.d_request(inp, "model-d") for inp in lines],
    }
    sent = {who: sha256(*(json.dumps(r, ensure_ascii=False) for r in rs)) for who, rs in requests.items()}
    assert sent == RC1_REQUESTS[question]
