"""The committee's logic on made-up answers: its parsers, votes, retry delays and checkpoints. No network."""

import email.utils
import json
import time

import httpx
import pytest

from lab import llm
from lab.reader import storage
from lab.reader.committee import MEMBERS, MODEL_ENV, act_emotion, clients, majority, move, sarcasm, view
from lab.reader.committee import label as committee_label
from lab.reader.schemas import ActTag, LineInput, LineTags

from .test_committee import RC1_PROMPTS

USAGE = {"input_tokens": 100, "output_tokens": 10}
# The player speaks to Ava with Bo present.
TO_AVA = LineInput(
    speaker="player",
    speaker_kind="player",
    to="Ava",
    targets=["Ava", "Bo"],
    text="Hand it over.",
    previous="Ava: No.",
)
# Mara speaks to the player with Tomas present.
TO_PLAYER = LineInput(
    speaker="Mara", speaker_kind="npc", to="player", targets=["player", "Tomas"], text="Get out."
)
QUESTIONS = {"committee": act_emotion, "exchange": move, "sarcasm": sarcasm}


# --- how the tests reach the committee ------------------------------------------------------------------------------


def parse_a(q: str, line: dict, inp: LineInput) -> tuple:
    return clients.parse_a(QUESTIONS[q].FORM, line, inp)


def parse_b(q: str, message: dict, inp: LineInput) -> tuple:
    return clients.parse_b(QUESTIONS[q].FORM, message, inp)


def parse_d(q: str, data: dict, inp: LineInput) -> tuple:
    return QUESTIONS[q].parse_d(data, inp)


def b_rows(*messages) -> list:
    """A Message Batch's results as b_results reads them."""
    return [(f"l{i}", m) for i, m in enumerate(messages)]


# --- made-up answers -------------------------------------------------------------------------------------------------


def a_line(text: str | None = None, content: list | None = None, **body) -> dict:
    """A's Batch-format output line: a completed message with `text`, or with `content`."""
    content = content if content is not None else [{"type": "output_text", "text": text}]
    out = [{"type": "reasoning", "summary": []}, {"type": "message", "content": content}]
    body = {"status": "completed", "usage": USAGE, "output": out, **body}
    return {"custom_id": "l00000", "response": {"status_code": 200, "body": body}}


def b_message(text: str = "", stop_reason: str = "end_turn", **extra) -> dict:
    """B's Messages API response as a dict: a thinking block, then the text."""
    content = [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}]
    return {"stop_reason": stop_reason, "usage": USAGE, "content": content, **extra}


ANSWERS = {
    "committee": (
        {
            "emotion": "angry",
            "emotion_intensity": "medium",
            "acts": [
                {"target": "Ava", "act": "demand", "intensity": "high"},
                {"target": "Bo", "act": "none", "intensity": "low"},
            ],
            "backchannel": False,
        },
        LineTags(
            emotion="angry",
            emotion_intensity="medium",
            acts={"Ava": ActTag(act="demand", intensity="high"), "Bo": ActTag(act="none")},
        ),
    ),
    "exchange": ({"exchange": "asks"}, "asks"),
    "sarcasm": ({"sarcastic": False}, False),
}
# An answer each reader rejects.
WRONG = {
    "committee": {"emotion": "angry", "emotion_intensity": "low", "acts": [], "backchannel": False},
    "exchange": {"exchange": "shrugs"},
    "sarcasm": {"sarcastic": "yes"},
}


# --- A's output lines ------------------------------------------------------------------------------------------------


def test_an_output_line_gives_its_text_and_usage():
    assert clients.parse_output_line(a_line('{"exchange": "asks"}')) == ('{"exchange": "asks"}', None, USAGE)


@pytest.mark.parametrize(
    "line, error",
    [
        ({"custom_id": "l1", "error": {"code": "batch_expired"}}, "no result"),
        (
            {"custom_id": "l1", "response": {"status_code": 429, "body": {"error": "gpt-x is busy"}}},
            "HTTP 429",
        ),
        (a_line("{", status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "max_tokens"),
        (a_line("{", status="incomplete", incomplete_details={"reason": "content_filter"}), "refusal"),
        (a_line(content=[{"type": "refusal", "refusal": "I won't label 'Hand it over.'"}]), "refusal"),
        (a_line(content=[]), "no result"),
    ],
)
def test_an_output_line_without_an_answer_says_why_by_kind_only(line, error):
    text, why, _ = clients.parse_output_line(line)
    assert text is None and why == error


# --- A, B and D's answers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", QUESTIONS)
def test_a_and_b_read_each_questions_answer(q):
    answer, vote = ANSWERS[q]
    assert parse_a(q, a_line(json.dumps(answer)), TO_AVA) == (vote, None, USAGE)
    assert parse_b(q, b_message(json.dumps(answer)), TO_AVA) == (vote, None, USAGE)


@pytest.mark.parametrize("q", QUESTIONS)
def test_an_answer_that_is_not_a_vote_is_an_invalid_answer(q):
    # a list or a number where an object belongs fails the line, not the whole call
    for text in ("not json", json.dumps(WRONG[q]), "[1, 2]", "7"):
        for answer in (parse_a(q, a_line(text), TO_AVA), parse_b(q, b_message(text), TO_AVA)):
            assert answer == (None, "invalid answer", USAGE)


@pytest.mark.parametrize("q", QUESTIONS)
def test_a_line_without_an_answer_keeps_its_error(q):
    line = {"custom_id": "l1", "response": {"status_code": 500, "body": {}}}
    assert parse_a(q, line, TO_AVA) == (None, "HTTP 500", {})


@pytest.mark.parametrize("q", QUESTIONS)
def test_b_stopping_on_a_refusal_or_max_tokens_gives_no_vote(q):
    refused = b_message("", "refusal", stop_details={"category": "cyber", "explanation": "the line"})
    assert parse_b(q, refused, TO_AVA) == (None, "refusal", USAGE)
    assert parse_b(q, b_message('{"exchange": "asks"}', "max_tokens"), TO_AVA) == (None, "max_tokens", USAGE)


def act_emotion_d(**answers) -> dict:
    """D's answers to the act and emotion questions on TO_AVA: one act each for Ava and Bo."""
    base = {
        "emotion": {
            "type": "choice",
            "name": "emotion",
            "choice": "angry",
            "probabilities": [
                {"value": "angry", "probability": 0.81234},
                {"value": "neutral", "probability": 0.1},
            ],
        },
        "emotion_intensity": {"type": "score", "name": "emotion_intensity", "score": 1.6},
        "backchannel": {"type": "predicate", "name": "backchannel", "probability": 0.9},
        "act_0": {
            "type": "choice",
            "name": "act_0",
            "choice": "demand",
            "probabilities": [{"value": "demand", "probability": 0.7}],
        },
        "act_intensity_0": {"type": "score", "name": "act_intensity_0", "score": 0.4},
        "act_1": {
            "type": "choice",
            "name": "act_1",
            "choice": "none",
            "probabilities": [{"value": "none", "probability": 0.95}],
        },
        "act_intensity_1": {"type": "score", "name": "act_intensity_1", "score": 2.0},
    }
    base.update(answers)
    return {"answers": [a for a in base.values() if a is not None], "usage": USAGE}


def test_d_reads_act_and_emotion_with_its_confidences():
    tags, error, usage = parse_d("committee", act_emotion_d(), TO_AVA)
    assert error is None and usage == USAGE
    # scores round to levels; `none` is low; a backchannel that carries an act is not one
    assert tags == LineTags(
        emotion="angry",
        emotion_intensity="high",
        emotion_confidence=0.8123,
        acts={
            "Ava": ActTag(act="demand", intensity="low", confidence=0.7),
            "Bo": ActTag(act="none", intensity="low", confidence=0.95),
        },
        backchannel=False,
    )


def test_d_refusing_or_skipping_a_question_gives_no_vote():
    refused = act_emotion_d(act_1={"type": "refusal", "name": "act_1"})
    assert parse_d("committee", refused, TO_AVA) == (None, "refusal", USAGE)
    missing = act_emotion_d(act_1=None)
    assert parse_d("committee", missing, TO_AVA) == (None, "invalid answer", USAGE)


def move_d(**probs) -> dict:
    """D's seven predicates on the move, by move name."""
    qid = {m: q for q, m in move.D_QIDS.items()}
    answers = [{"type": "predicate", "name": qid[m], "probability": p} for m, p in probs.items()]
    return {"answers": answers, "usage": USAGE}


def test_d_reads_the_move_from_its_seven_predicates():
    probs = dict.fromkeys(move.D_QIDS.values(), 0.1) | {"asks": 0.9, "answers": 0.6}
    assert parse_d("exchange", move_d(**probs), TO_AVA) == ("answers", None, USAGE, probs)


def test_d_missing_a_move_fails_the_line():
    probs = dict.fromkeys(move.D_QIDS.values(), 0.1)
    del probs["other"]
    with pytest.raises(clients.DecisionsError, match="^invalid answer$"):
        parse_d("exchange", move_d(**probs), TO_AVA)
    refused = move_d(**probs)
    refused["answers"].append({"type": "refusal", "name": "q7"})
    with pytest.raises(clients.DecisionsError, match="^refusal$"):
        parse_d("exchange", refused, TO_AVA)


@pytest.mark.parametrize(
    "probs, label",
    [
        ({"asks": 0.9, "answers": 0.6}, "answers"),  # the guide's order, not the most probable
        ({"asks": 0.9, "closes": 0.5}, "asks"),
        ({"other": 0.5}, "other"),  # 0.5 reaches
        ({"asks": 0.3, "closes": 0.45, "other": 0.2}, "closes"),  # none reaches: the most probable
        ({}, "asks"),
    ],
)
def test_ds_move_is_the_first_in_the_guides_order_to_reach_one_half(probs, label):
    assert move.d_label(probs) == label


@pytest.mark.parametrize("p, vote", [(0.5, True), (0.49, False)])
def test_ds_sarcasm_vote_is_its_probability_at_one_half(p, vote):
    data = {"answers": [{"type": "predicate", "name": "sarcastic", "probability": p}], "usage": USAGE}
    assert parse_d("sarcasm", data, TO_AVA) == (vote, None, USAGE, p)


def test_d_without_a_sarcasm_probability_gives_no_vote():
    data = {"answers": [{"type": "refusal", "name": "sarcastic"}], "usage": USAGE}
    assert parse_d("sarcasm", data, TO_AVA) == (None, "refusal", USAGE, None)
    assert parse_d("sarcasm", {"answers": [], "usage": USAGE}, TO_AVA) == (
        None,
        "invalid answer",
        USAGE,
        None,
    )


# --- act and emotion: names --------------------------------------------------------------------------------------------


def test_acts_are_matched_to_the_people_present_by_name():
    answer = {
        "emotion": "angry",
        "emotion_intensity": "high",
        "acts": [
            {"target": "You", "act": "demand", "intensity": "high"},  # whoever it was said to
            {"target": " tomas ", "act": "none", "intensity": "high"},  # any case and spacing; none is low
            {"target": "the player", "act": "insult", "intensity": "low"},  # the player again: the first wins
            {"target": "Nobody", "act": "threat", "intensity": "high"},  # not present: dropped
        ],
        "backchannel": True,
    }
    tags = act_emotion.tags_from_answer(answer, TO_PLAYER)
    assert tags.acts == {"player": ActTag(act="demand", intensity="high"), "Tomas": ActTag(act="none")}
    assert tags.backchannel is False  # it carries an act


def test_the_player_is_named_and_you_needs_a_listener():
    room = LineInput(speaker="Mara", speaker_kind="npc", targets=["player"], text="Hm.")
    answer = {"emotion": "neutral", "emotion_intensity": "high", "backchannel": True}
    tags = act_emotion.tags_from_answer(
        {**answer, "acts": [{"target": "The Player", "act": "none", "intensity": "low"}]}, room
    )
    assert tags.acts == {"player": ActTag(act="none")}
    assert tags.emotion_intensity == "low" and tags.backchannel is True  # neutral is low
    with pytest.raises(ValueError, match="no act for"):
        act_emotion.tags_from_answer(
            {**answer, "acts": [{"target": "you", "act": "none", "intensity": "low"}]}, room
        )


# --- the votes ---------------------------------------------------------------------------------------------------------


def tags(emotion: str, intensity: str, ava: str, ava_intensity: str = "medium", bo: str = "none") -> LineTags:
    return LineTags(
        emotion=emotion,
        emotion_intensity="low" if emotion == "neutral" else intensity,
        acts={
            "Ava": ActTag(act=ava, intensity="low" if ava == "none" else ava_intensity),
            "Bo": ActTag(act=bo),
        },
    )


def head(h) -> tuple:
    return h.label, h.intensity, h.agree, h.votes


def test_act_and_emotion_take_two_matching_votes_and_the_lower_middle_intensity():
    v = act_emotion.vote(
        TO_AVA,
        {
            "a": tags("angry", "high", "demand", "high"),
            "b": tags("angry", "medium", "demand", "low"),
            "d": tags("sad", "low", "insult"),
        },
    )
    assert head(v.emotion) == ("angry", "medium", 2, {"a": "angry", "b": "angry", "d": "sad"})
    assert head(v.acts["Ava"]) == ("demand", "low", 2, {"a": "demand", "b": "demand", "d": "insult"})
    assert head(v.acts["Bo"]) == ("none", "low", 3, {"a": "none", "b": "none", "d": "none"})
    assert head(v.backchannel) == ("false", None, 3, {"a": "false", "b": "false", "d": "false"})
    assert v.complete


def test_three_matching_intensities_take_the_median():
    v = act_emotion.vote(
        TO_AVA,
        {
            "a": tags("sad", "low", "none"),
            "b": tags("sad", "high", "none"),
            "d": tags("sad", "medium", "none"),
        },
    )
    assert head(v.emotion)[:3] == ("sad", "medium", 3)


def test_act_and_emotion_without_a_majority_have_no_label():
    split = act_emotion.vote(
        TO_AVA, {"a": tags("sad", "low", "none"), "b": tags("happy", "low", "none"), "d": None}
    )
    assert head(split.emotion) == (None, None, 1, {"a": "sad", "b": "happy"})
    assert not split.complete
    alone = act_emotion.vote(TO_AVA, {"a": tags("sad", "low", "none"), "b": None, "d": None})
    assert head(alone.emotion) == (None, None, 1, {"a": "sad"})
    nobody = act_emotion.vote(TO_AVA, {"a": None, "b": None, "d": None})
    assert head(nobody.emotion) == (None, None, 0, {})


@pytest.mark.parametrize(
    "labels, vote",
    [
        ({"a": "asks", "b": "asks", "d": "answers"}, {"label": "asks", "agree": 2}),
        ({"a": "asks", "b": "asks", "d": "asks"}, {"label": "asks", "agree": 3}),
        ({"a": "asks", "b": "closes", "d": "answers"}, {"label": None, "agree": 1}),
        ({"a": "asks", "b": None, "d": "answers"}, {"label": None, "agree": 1}),
        ({"a": None, "b": None, "d": "answers"}, {"label": None, "agree": 1}),
        ({"a": None, "b": None, "d": None}, {"label": None, "agree": 0}),
    ],
)
def test_the_move_takes_two_matching_votes(labels, vote):
    votes = {m: v for m, v in labels.items() if v is not None}
    assert move.vote(labels) == {**vote, "votes": votes}
    assert list(move.vote(labels)) == ["label", "agree", "votes"]


@pytest.mark.parametrize(
    "labels, label, agree",
    [
        ({"a": True, "b": True, "d": False}, True, 2),
        ({"a": False, "b": False, "d": False}, False, 3),
        ({"a": False, "b": False, "d": None}, False, 2),
        ({"a": True, "b": False, "d": None}, None, 0),  # no majority: agree is 0
        ({"a": True, "b": None, "d": None}, None, 0),
        ({"a": None, "b": None, "d": None}, None, 0),
    ],
)
def test_sarcasm_takes_two_matching_votes(labels, label, agree):
    votes = {m: v for m, v in labels.items() if v is not None}
    assert sarcasm.vote(labels) == {"label": label, "votes": votes, "agree": agree}
    assert list(sarcasm.vote(labels)) == ["label", "votes", "agree"]


# --- D's retries ---------------------------------------------------------------------------------------------------------


def test_retry_after_reads_milliseconds_seconds_or_a_date():
    assert clients._retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert clients._retry_after({"retry-after-ms": "soon", "retry-after": "2"}) == 2.0
    assert clients._retry_after({"retry-after": "3"}) == 3.0
    assert clients._retry_after({"retry-after": "soon"}) is None
    assert clients._retry_after({}) is None
    later = email.utils.formatdate(time.time() + 60, usegmt=True)
    assert 55 < clients._retry_after({"retry-after": later}) <= 60
    assert clients._retry_after({"retry-after": email.utils.formatdate(time.time() - 60, usegmt=True)}) == 0.0


# --- checkpoints ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def volume(monkeypatch) -> dict:
    """The data volume as a dict of key to bytes."""
    files: dict[str, bytes] = {}
    monkeypatch.setattr(storage, "exists", lambda key: key in files)
    monkeypatch.setattr(storage, "get", lambda key: files[key])
    monkeypatch.setattr(storage, "put_json", lambda key, obj: files.update({key: json.dumps(obj).encode()}))
    return files


def test_checkpoints_keep_each_answer_and_its_extras(volume):
    work = committee_label.Work("tonetext-test-work")
    act = {"l1": (ANSWERS["committee"][1], None, USAGE), "l2": (None, "max_tokens", {})}
    work.save("committee", "a", act)
    assert set(volume) == {"runs/tonetext/labels/tonetext-test-work/committee-a.json"}
    assert work.load("committee", "a") == act
    probs = {"asks": 0.9}
    work.save("exchange", "d", {"l1": ("asks", None, USAGE, probs)})
    assert work.load("exchange", "d") == {"l1": ("asks", None, USAGE, probs)}
    assert work.load("sarcasm", "b") == {}


def test_a_restarted_ask_keeps_what_was_answered_and_saves_each_chunk(volume):
    work = committee_label.Work("tonetext-test-work")
    items = [(f"l{i}", TO_AVA) for i in range(6)]
    work.save("exchange", "b", {"l0": ("asks", None, USAGE), "l1": (None, "max_tokens", {})})
    asked_for, saves = [], []

    def runner(chunk):
        asked_for.append([cid for cid, _ in chunk])
        saves.append(len(work.load("exchange", "b")))
        return {cid: ("closes", None, USAGE) for cid, _ in chunk}

    have, asked = work.ask_chunked("exchange", "b", items, runner, chunk=2)
    assert asked_for == [["l1", "l2"], ["l3", "l4"], ["l5"]]
    # each chunk saved before the next is asked
    assert saves == [2, 3, 5]
    assert sorted(asked) == ["l1", "l2", "l3", "l4", "l5"]
    assert have["l0"] == ("asks", None, USAGE) and work.load("exchange", "b") == have


def test_a_message_batch_is_known_by_its_first_answer():
    act = b_message(json.dumps(ANSWERS["committee"][0]))
    assert committee_label.b_question(b_rows(None, b_message("not json"), act)) == "committee"
    assert committee_label.b_question(b_rows(b_message('{"exchange": "asks"}'))) == "exchange"
    assert committee_label.b_question(b_rows(b_message('{"sarcastic": true}'), act)) is None
    assert committee_label.b_question(b_rows(None)) is None


# --- the shared pieces ---------------------------------------------------------------------------------------------------


def test_a_majority_is_two_matching_votes_that_no_other_label_ties():
    assert majority({"a": "asks", "b": "asks", "d": "closes"}) == ("asks", 2)
    assert majority({"a": False, "b": False}) == (False, 2)
    assert majority({"a": "asks", "b": "closes"}) == (None, 1)
    assert majority({}) == (None, 0)


def test_the_view_puts_the_line_before_first_or_after_the_line():
    assert list(view(TO_AVA)) == [
        "speaker",
        "speaker_is",
        "said_to",
        "people_in_scene",
        "previous_line",
        "line",
    ]
    assert list(view(TO_AVA, previous_first=False))[-2:] == ["line", "previous_line"]
    assert "previous_line" not in view(TO_PLAYER) and view(TO_PLAYER)["said_to"] == "player"


def test_one_reader_takes_a_messages_answer_or_its_stop_reason():
    assert llm.read_message(b_message('{"a": 1}')) == ({"a": 1}, None)
    assert llm.read_message(b_message("", "refusal")) == (None, "refusal")
    assert llm.read_message(b_message("{", "max_tokens")) == (None, "max_tokens")
    with pytest.raises(ValueError):
        llm.read_message(b_message("not json"))


def test_the_labellers_ids_are_read_in_the_job_and_only_their_names_kept(monkeypatch):
    for m, name in MODEL_ENV.items():
        monkeypatch.setenv(name, f"secret-model-{m}")
    assert committee_label.models() == {m: f"secret-model-{m}" for m in MEMBERS}
    systems, meta = committee_label.prepare("label", lines_key="k")
    assert list(meta) == ["labellers", "system_prompt_sha256", "lines_key", "step"]
    assert meta["labellers"] == MODEL_ENV and meta["system_prompt_sha256"] == RC1_PROMPTS
    assert "secret-model" not in json.dumps(meta)
    monkeypatch.delenv("LABELLER_B_MODEL")
    with pytest.raises(RuntimeError, match="LABELLER_B_MODEL") as missing:
        committee_label.models()
    assert "secret-model" not in str(missing.value)


class Response:
    def __init__(self, status: int, body=None, headers: dict | None = None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


class HTTP:
    """Answers each POST with the next response, or raises it."""

    def __init__(self, *responses):
        self.responses = list(responses)

    def post(self, url, json):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def waits(monkeypatch) -> list:
    """Every retry delay, slept at once, without jitter."""
    slept: list[float] = []
    monkeypatch.setattr(clients.time, "sleep", slept.append)
    monkeypatch.setattr(clients.random, "random", lambda: 0.0)
    return slept


def test_one_retry_loop_backs_off_honours_retry_after_and_rejects_bad_bodies(waits):
    http = HTTP(
        httpx.ConnectError("down"),
        Response(429, {}, {"retry-after": "3"}),
        Response(200, None),
        Response(200, {"answers": []}),
    )
    assert clients.post(http, "url", {}, lambda data: isinstance(data, dict)) == (200, {"answers": []})
    assert waits == [0.5, 3.0, 2.0]


def test_a_refused_request_is_not_retried_and_retries_run_out(waits):
    assert clients.post(HTTP(Response(400, {"error": "no"})), "url", {}, bool) == (400, {"error": "no"})
    assert waits == []
    busy = [Response(503, {}) for _ in range(clients.RETRIES + 1)]
    assert clients.post(HTTP(*busy), "url", {}, bool) == (503, {})
    with pytest.raises(httpx.TransportError):
        clients.post(HTTP(*[httpx.ReadTimeout("slow")] * (clients.RETRIES + 1)), "url", {}, bool)


def test_a_cancelled_batch_is_waited_for_a_grace_then_left(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(clients.time, "time", lambda: now[0])
    monkeypatch.setattr(clients.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    cancelled = []
    clients._wait_or_cancel(lambda: False, lambda: cancelled.append(now[0]), 100.0)
    assert cancelled == [120.0] and now[0] == 100.0 + clients.GRACE_S + 20


def test_d_failing_gives_its_error_by_kind(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    probs = dict.fromkeys(move.D_QIDS.values(), 0.1) | {"closes": 0.8}
    outcomes = {
        "l0": (500, {"error": "gpt-secret failed on 'Hand it over.'"}),
        "l1": (200, {"no": "answers"}),
        "l2": httpx.ConnectError("down"),
        "l3": (200, move_d(**probs)),
        "l4": (200, {"answers": []}),
    }

    def post(http, url, body, ok):
        if isinstance(outcomes[body["cid"]], Exception):
            raise outcomes[body["cid"]]
        return outcomes[body["cid"]]

    monkeypatch.setattr(clients, "post", post)
    items = [(cid, TO_AVA.model_copy(update={"text": cid})) for cid in outcomes]
    got = clients.ask_d(items, lambda inp: {"cid": inp.text}, move.parse_d)
    assert got == {
        "l0": (None, "HTTP 500", {}),
        "l1": (None, "no result", {}),
        "l2": (None, "no result", {}),
        "l3": ("closes", None, USAGE, probs),
        "l4": (None, "invalid answer", {}),
    }


class Batches:
    """OpenAI's Batches endpoints for one finished batch: its output file answers l0 and refuses l1."""

    def __init__(self, api_key):
        pass

    def retrieve_batch(self, batch_id):
        return {"status": "completed", "output_file_id": "out", "error_file_id": None}

    def file_lines(self, file_id):
        refused = a_line(content=[{"type": "refusal", "refusal": "no"}])
        return [{**a_line('{"exchange": "asks"}'), "custom_id": "l0"}, {**refused, "custom_id": "l1"}]

    def close(self):
        pass


def test_a_batch_is_collected_through_its_wait_and_a_missing_line_has_no_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(clients, "BatchClient", Batches)
    items = [(f"l{i}", TO_AVA) for i in range(3)]
    assert clients.collect_a(move.FORM, "batch", items, 0.0) == {
        "l0": ("asks", None, USAGE),
        "l1": (None, "refusal", USAGE),
        "l2": (None, "no result", {}),
    }


def test_a_message_batchs_results_are_read_by_the_questions_form():
    items = [(f"l{i}", TO_AVA) for i in range(3)]
    rows = [("l0", b_message('{"sarcastic": true}')), ("l1", None), ("l9", b_message("{}"))]
    assert clients.b_answers(sarcasm.FORM, rows, items) == {
        "l0": (True, None, USAGE),
        "l1": (None, "no result", {}),
    }


# --- asking, the batch guard and the cost ------------------------------------------------------------------------------


SETS = {q: f"tonetext-test-{q}" for q in committee_label.QUESTIONS}
SYSTEMS = dict.fromkeys(committee_label.QUESTIONS, "system")
MODELS = {m: f"model-{m}" for m in MEMBERS}


def test_each_batch_is_sent_once_and_a_failed_one_is_kept_as_its_error_kind(volume, monkeypatch):
    sent = []

    def submit_a(form, items, system, model, label):
        sent.append(("a", label, [cid for cid, _ in items]))
        return f"batch_{len(sent)}"

    def submit_b(form, items, system, model):
        sent.append(("b", form.name, [cid for cid, _ in items]))
        if form.name == "exchange":
            raise clients.HTTPError(400, "model-b is not a model; the line was 'Hand it over.'")
        return f"msgbatch_{len(sent)}"

    monkeypatch.setattr(clients, "submit_a", submit_a)
    monkeypatch.setattr(clients, "submit_b", submit_b)
    work = committee_label.Work("tonetext-test-work")
    items = [(f"l{i}", TO_AVA) for i in range(5)]
    record = work.send_once(items, SYSTEMS, MODELS, ("a", "b"), 60.0, SETS, chunk=2)
    assert [s[1] for s in sent if s[0] == "a"] == [
        f"tonetext-test-{q}-{k}" for q in ("committee", "exchange") for k in range(3)
    ]
    assert record["a"]["committee"] == ["batch_1", "batch_2", "batch_3"]
    assert record["b"]["exchange"] == [{"error": "HTTP 400"}] * 3
    assert time.time() < record["deadline"] <= time.time() + 60 and record["chunk"] == 2
    assert work.send_once(items, SYSTEMS, MODELS, ("a", "b"), 60.0, SETS, chunk=2) == record
    assert len(sent) == 12  # the second call sent nothing


def test_one_batch_of_every_line_is_named_by_its_set(volume, monkeypatch):
    labels = []
    monkeypatch.setattr(
        clients, "submit_a", lambda form, items, system, model, label: labels.append(label) or "b"
    )
    committee_label.Work("tonetext-test-work").send_once(
        [("l0", TO_AVA)], SYSTEMS, MODELS, ("a",), 60.0, SETS
    )
    assert labels == ["tonetext-test-committee", "tonetext-test-exchange"]


def test_sent_batches_are_collected_until_the_deadline_keeping_answered_lines(volume, monkeypatch):
    calls = []

    def collect_a(form, batch_id, items, wait_s):
        calls.append((form.name, batch_id, [cid for cid, _ in items], round(wait_s)))
        vote = ANSWERS["committee" if form.name == "line_tags" else "exchange"][1]
        return {cid: (None, "no result", {}) if cid == "l1" else (vote, None, {}) for cid, _ in items}

    monkeypatch.setattr(clients, "collect_a", collect_a)
    work = committee_label.Work("tonetext-test-work")
    items = [(f"l{i}", TO_AVA) for i in range(4)]
    record = {
        "a": {"committee": ["b1", {"error": "HTTP 400"}], "exchange": ["b3", "b4"]},
        "deadline": time.time() + 100,
        "chunk": 2,
    }
    got = work.collect_batches(record, items)
    # the batch that was never made is skipped
    assert sorted(calls) == [
        ("exchange", "b3", ["l0", "l1"], 100),
        ("exchange", "b4", ["l2", "l3"], 100),
        ("line_tags", "b1", ["l0", "l1"], 100),
    ]
    assert sorted(got[("committee", "a")]) == ["l0"] and sorted(got[("exchange", "a")]) == ["l0", "l2", "l3"]
    assert work.load("exchange", "a-batch") == got[("exchange", "a")]


def test_what_a_batch_left_is_asked_synchronously_once(monkeypatch):
    asked = []

    def ask_member(q, m, items, system, model, concurrency=12):
        asked.append((m, [cid for cid, _ in items]))
        return {cid: ("closes", None, {}) for cid, _ in items}

    monkeypatch.setattr(committee_label, "ask_member", ask_member)
    monkeypatch.setattr(clients, "collect_a", lambda form, bid, items, wait_s: {"l0": ("asks", None, {})})
    items = [(f"l{i}", TO_AVA) for i in range(3)]
    batches = {"b": {"error": "HTTP 400"}, "a": "batch"}
    res, filled, ids = committee_label.ask("exchange", items, "system", MODELS, batches, 5.0)
    assert sorted(asked) == [("a", ["l1", "l2"]), ("b", ["l0", "l1", "l2"]), ("d", ["l0", "l1", "l2"])]
    assert res["a"] == {"l0": ("asks", None, {}), "l1": ("closes", None, {}), "l2": ("closes", None, {})}
    # a batch that was never made leaves every line to be asked, at Standard
    assert (
        sorted(filled["a"]) == ["l1", "l2"] and sorted(filled["b"]) == ["l0", "l1", "l2"] and ids is batches
    )


def test_a_checkpointed_run_takes_batch_answers_and_asks_only_the_rest(volume):
    work = committee_label.Work("tonetext-test-work")
    items = [(f"l{i}", TO_AVA) for i in range(3)]
    asked: dict = {}

    def runner(k):
        def run(chunk):
            asked.setdefault(k, []).extend(cid for cid, _ in chunk)
            return {cid: (ANSWERS[k[0]][1], None, {"by": "sync"}) for cid, _ in chunk}

        return run

    runners = {(q, m): runner((q, m)) for q in committee_label.QUESTIONS for m in MEMBERS}
    batched = [("committee", "a"), ("exchange", "a")]
    from_a_batch = (ANSWERS["committee"][1], None, {"by": "batch"})
    from_batches = {("committee", "a"): {"l0": from_a_batch, "l1": (None, "no result", {})}}
    done, from_batch = work.answer_all(items, runners, batched, lambda: from_batches)
    assert asked[("committee", "a")] == ["l1", "l2"] and asked[("exchange", "a")] == ["l0", "l1", "l2"]
    assert asked[("sarcasm", "b")] == ["l0", "l1", "l2"] and len(asked) == len(runners)
    assert from_batch == {("committee", "a"): {"l0": from_a_batch}, ("exchange", "a"): {}}
    answers = committee_label.answers(done, from_batch, {"committee": {"a": ["b1"], "b": "synchronous"}})
    res, filled, ids = answers["committee"]
    assert res["a"]["l0"] == from_a_batch and ids == {"a": ["b1"], "b": "synchronous"}
    assert list(filled) == ["a"] and sorted(filled["a"]) == ["l1", "l2"]
    assert answers["sarcasm"][1:] == ({}, {})
    asked.clear()
    work.answer_all(items, runners, batched, dict)
    assert asked == {}  # a restart asks nothing again


def test_batch_answers_cost_half_and_lines_filled_synchronously_full():
    use = {"input_tokens": 3500, "output_tokens": 300, "cache_read_input_tokens": 3000}
    res = {m: {f"l{i}": ("x", None, dict(use)) for i in range(10)} for m in MEMBERS}
    sar = {m: {cid: (*r, 0.1) if m == "d" else r for cid, r in res[m].items()} for m in MEMBERS}
    answers = {
        "committee": (res, {}, {}),
        "exchange": (res, {"a": {"l0": res["a"]["l0"]}}, {}),
        "sarcasm": (sar, {}, {}),
    }
    cost = committee_label.costs(answers, {"a": True, "b": True})
    # A at $2 and $10, B at Claude's $4, $20 and $0.20 for cache reads, D at $0.10 for input
    assert cost["committee"] == {"a": 0.05, "b": 0.103, "d": 0.0035, "filled_synchronously": {"a": 0, "b": 0}}
    assert cost["exchange"] == {"a": 0.055, "b": 0.103, "d": 0.0035, "filled_synchronously": {"a": 1, "b": 0}}
    assert cost["sarcasm"] == {"a": 0.1, "b": 0.206, "d": 0.0035}
    assert cost["total"] == 0.63
    assert committee_label.costs(answers, {"a": True})["committee"]["b"] == 0.206
