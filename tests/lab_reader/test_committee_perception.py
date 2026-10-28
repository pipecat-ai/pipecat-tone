"""The committee's questions under the Realtime Context Standard's perception: what A, B and D see and how their
answers become votes. No network."""

import json

import pytest

from lab.reader.committee import MODEL_ENV, clients, perception_emotion, perception_sarcasm
from lab.reader.committee import label as committee_label
from lab.reader.schemas import ActTag, LineInput, LineTags
from lab.reader.support import PROBES

QUESTIONS = {"committee": perception_emotion, "sarcasm": perception_sarcasm}
TURN = LineInput(
    speaker="player",
    speaker_kind="player",
    to="Ava",
    targets=["Ava"],
    text="Great, love that. So it'll miss the birthday.",
    previous="Ava: It's still being processed, so it hasn't shipped yet.",
)


@pytest.mark.parametrize("q", QUESTIONS)
def test_the_prompts_teach_the_standards_axioms_for_voice_agents(q):
    system = QUESTIONS[q].system_prompt()
    assert "AI voice agent" in system and "## The axioms" in system
    assert "The agent's line is context, not the turn" in system
    assert "game" not in system.lower() and "player" not in system.lower()
    assert not any(v in system for v in MODEL_ENV.values())


@pytest.mark.parametrize("q", QUESTIONS)
def test_the_worked_examples_are_not_the_probes(q):
    probes = {t.lower() for _, t, _, _ in PROBES}
    assert not probes & {e[1].lower() for e in QUESTIONS[q].EXAMPLES}


@pytest.mark.parametrize("q", QUESTIONS)
def test_the_committee_sees_what_the_reader_reads_and_no_names(q):
    shown = QUESTIONS[q].FORM.view(TURN)
    assert shown["agent_line_before"] == "It's still being processed, so it hasn't shipped yet."
    assert shown["turn"] == TURN.text and "Ava" not in json.dumps(shown)
    d = QUESTIONS[q].d_request(TURN, "model-d")
    assert d["model"] == "model-d" and "Ava" not in d["input"]
    alone = QUESTIONS[q].FORM.view(TURN.model_copy(update={"previous": None}))
    assert "agent_line_before" not in alone


def test_an_emotion_answer_becomes_line_tags_with_one_act_toward_the_agent():
    form = perception_emotion.FORM
    tags = form.read(
        {
            "emotion": "angry",
            "emotion_intensity": "medium",
            "act": "none",
            "act_intensity": "high",
            "backchannel": True,
        },
        TURN,
    )
    assert tags == LineTags(
        emotion="angry",
        emotion_intensity="medium",
        acts={"Ava": ActTag(act="none", intensity="low")},
        backchannel=True,
    )
    neutral = form.read(
        {
            "emotion": "neutral",
            "emotion_intensity": "high",
            "act": "thanks",
            "act_intensity": "low",
            "backchannel": True,
        },
        TURN,
    )
    assert neutral.emotion_intensity == "low" and neutral.backchannel is False


def test_d_answers_the_emotion_and_the_act_toward_the_agent():
    names = [q["name"] for q in perception_emotion.d_request(TURN, "m")["questions"]]
    assert names == ["emotion", "emotion_intensity", "backchannel", "act_0", "act_intensity_0"]
    data = {
        "answers": [
            {
                "type": "choice",
                "name": "emotion",
                "choice": "angry",
                "probabilities": [{"value": "angry", "probability": 0.8}],
            },
            {"type": "score", "name": "emotion_intensity", "score": 1.2},
            {"type": "predicate", "name": "backchannel", "probability": 0.1},
            {
                "type": "choice",
                "name": "act_0",
                "choice": "insult",
                "probabilities": [{"value": "insult", "probability": 0.6}],
            },
            {"type": "score", "name": "act_intensity_0", "score": 0.2},
        ],
        "usage": {"input_tokens": 10},
    }
    tags, error, usage = perception_emotion.parse_d(data, TURN)
    assert error is None and usage == {"input_tokens": 10}
    assert tags.emotion == "angry" and tags.emotion_intensity == "medium"
    assert tags.acts == {"Ava": ActTag(act="insult", intensity="low", confidence=0.6)}
    assert perception_emotion.parse_d({"answers": [], "usage": {}}, TURN)[0] is None


def test_sarcasm_reads_a_boolean_and_ds_probability():
    assert perception_sarcasm.FORM.read({"sarcastic": True}, TURN) is True
    with pytest.raises(ValueError):
        perception_sarcasm.FORM.read({"sarcastic": "yes"}, TURN)
    data = {"answers": [{"type": "predicate", "name": "sarcastic", "probability": 0.7}], "usage": {}}
    assert perception_sarcasm.parse_d(data, TURN)[0] is True
    assert perception_sarcasm.d_request(TURN, "m")["questions"][0]["name"] == "sarcastic"


@pytest.mark.parametrize("q", QUESTIONS)
def test_each_member_asks_the_standards_question(q):
    m = QUESTIONS[q]
    system = m.system_prompt()
    a = clients.a_request(m.FORM, "l00000", TURN, system, "model-a")
    b = clients.b_request(m.FORM, TURN, system, "model-b")
    assert a["body"]["input"][0]["content"] == system and b["system"][0]["text"] == system
    assert a["body"]["text"]["format"]["name"] == m.FORM.name


def test_a_question_set_without_the_move_is_prepared_and_asked_alone():
    systems, meta = committee_label.prepare("pilot", QUESTIONS, lines_key="k")
    assert set(systems) == set(meta["system_prompt_sha256"]) == set(QUESTIONS)
    runners = committee_label.runners(systems, {"a": "A", "b": "B", "d": "D"}, 4, QUESTIONS)
    assert set(runners) == {(q, m) for q in QUESTIONS for m in ("a", "b", "d")}
    work = committee_label.Work("tonetext-test-work", QUESTIONS, ("committee", "sarcasm"))
    assert work.batched == ("committee", "sarcasm") and work.questions is QUESTIONS
