"""The voice-agent test set's prompts, parser and line picker, on made-up teacher output."""

import json
import random
from collections import Counter

from lab.reader import voice

OUTPUT = json.dumps(
    {
        "before": [
            {
                "agent": "Hi, thanks for calling Northline Broadband, this is Ava. How can I help?",
                "user": "Yeah, hi. My internet keeps dropping every night around eight.",
            },
            {
                "agent": "I'm sorry to hear that. Can I take the account holder's name?",
                "user": "It's under my wife's name, I think. Priya Shah.",
            },
        ],
        "moment": {
            "agent": "Thanks. I can see an outage in your area, but there's no time for a fix yet.",
            "user": "Oh, wonderful. Just what I needed this week.",
        },
        "after": [
            {"agent": "I understand. I can add a credit to your next bill.", "user": "Okay. Fine, do that."},
            {"agent": "Done. Is there anything else I can help with?", "user": "(sighs) No."},
        ],
        "agent_last": "Thanks for calling, have a good evening.",
    }
)


def conv(text=OUTPUT, asr=False, seed="sarcasm"):
    turns, problems = voice.parse_json_conversation(text, "agent")
    return {
        "prompt_id": "tv1-isp_support-000",
        "domain": "isp_support",
        "agent": "Ava",
        "company": "Northline Broadband",
        "scenario": "s",
        "persona": "p",
        "arc": "frustrated",
        "seed": seed,
        "seed_kind": "after_setback",
        "move_focus": None,
        "move_extra": "hedges",
        "asr": asr,
        "turns": turns,
        "problems": problems,
    }


def unmarked(c: dict) -> dict:
    """The conversation with its moment's mark lost."""
    return {**c, "turns": [{**t, "marked": False} for t in c["turns"]]}


def test_prompts_balance_seeds_and_speech_to_text():
    ps = voice.build_prompts(6)
    assert len(ps) == 6 * len(voice.DOMAINS)
    assert len({p["prompt_id"] for p in ps}) == len(ps)
    for d in voice.DOMAINS:
        mine = [p for p in ps if p["domain"] == d]
        assert Counter(p["seed"] for p in mine) == {"sarcasm": 2, "lookalike": 2, "none": 2}
        assert sum(p["asr"] for p in mine) == 2
    p = ps[0]
    text = p["messages"][1]["content"]
    assert (
        p["messages"][0]["role"] == "system"
        and "Write it as JSON" in text
        and "One round in the middle is the moment" in text
    )
    assert (
        "{agent}" not in text
        and "{company}" not in text
        and "{setback}" not in text
        and "{good_news}" not in text
    )
    first_user = [q for q in ps if q["first"] == "user"]
    assert first_user and all(q["domain"] in ("sales_call", "companion_chat") for q in first_user)
    assert (
        "opening" in voice.schema("user")["properties"]
        and "opening" not in voice.schema("agent")["properties"]
    )


def test_json_rounds_alternate_and_mark_the_moment():
    obj = {
        "before": [
            {"agent": "Hi, this is Ava. How can I help?", "user": "My internet keeps dropping."},
            {"agent": "Sorry. Whose name is the account under?", "user": "My wife's, I think."},
        ],
        "moment": {
            "agent": "There's an outage with no fix time yet.",
            "user": "Oh, wonderful. Just what I needed.",
        },
        "after": [{"agent": "I can add a credit to your bill.", "user": "Fine."}],
        "agent_last": "",
    }
    turns, problems = voice.parse_json_conversation(json.dumps(obj), "agent")
    assert [t["who"] for t in turns] == ["agent", "user"] * 4 and not problems
    assert [i for i, t in enumerate(turns) if t["marked"]] == [5]
    c = dict(conv(), turns=turns)
    assert voice.usable(c) and voice.marked_ok(c)
    obj2 = {**obj, "moment": {"agent": "", "user": "Oh, wonderful."}, "opening": "Hello?"}
    turns2, problems2 = voice.parse_json_conversation(json.dumps(obj2), "user")
    assert turns2[0] == {"who": "user", "text": "Hello?", "marked": False}
    assert "mark after a person's row" in problems2 and not any(t["marked"] for t in turns2)
    assert voice.parse_json_conversation("{not json", "agent") == ([], ["not json"])


def test_stage_directions_are_dropped_not_repaired():
    cands = voice.candidates(conv())
    # "(sighs) No." is never a line, nor a line whose line before is one.
    assert 9 not in [x["i"] for x in cands]
    assert 10 not in [x["i"] for x in cands]
    assert 8 in [x["i"] for x in cands]


def test_picked_lines_follow_tones_corpus_conventions():
    c = conv()
    picked = voice.pick_lines(c, random.Random(0), set(), Counter())
    lines = [ln for ln, _ in picked]
    assert len(lines) == 3
    user = [ln for ln in lines if ln.input.speaker == "player"]
    agent = [ln for ln in lines if ln.input.speaker == "Ava"]
    assert len(user) == 2 and len(agent) == 1
    seed = user[0]
    assert seed.input.text == "Oh, wonderful. Just what I needed this week."
    assert seed.input.previous.startswith("Ava: Thanks. I can see an outage")
    assert seed.input.to == "Ava" and seed.input.targets == ["Ava"] and seed.input.speaker_kind == "player"
    assert {"voice_agent", "isp_support", "user", "player", "sarcasm_seed"} <= set(seed.slices)
    assert (
        seed.generator == "sarcasm_seed:after_setback" and seed.split == "test" and seed.source == "generated"
    )
    assert seed.source_ref == "tv1-isp_support-000#5" and seed.group_key == "gen:tv1-isp_support-000"
    a = agent[0]
    assert a.input.to == "player" and a.input.targets == ["player"] and a.input.speaker_kind == "npc"
    assert a.input.previous is None or a.input.previous.startswith("player: ")
    assert {"voice_agent", "agent", "npc"} <= set(a.slices) and "asr_style" not in a.slices


def test_speech_to_text_style_reaches_the_persons_lines_and_their_use_as_the_line_before():
    c = conv(asr=True)
    picked = voice.pick_lines(c, random.Random(1), set(), Counter())
    for ln, _ in picked:
        if ln.input.speaker == "player":
            assert "asr_style" in ln.slices and ln.input.text == ln.input.text.lower()
            assert not any(ch in ln.input.text for ch in ",.!?")
        elif ln.input.previous:
            assert ln.input.previous == "player: " + voice.asr_text(ln.input.previous[len("player: ") :])


def test_a_text_is_never_taken_twice_with_the_same_line_before():
    seen, per = set(), Counter()
    first = voice.pick_lines(conv(), random.Random(0), seen, per)
    again = voice.pick_lines(conv(), random.Random(0), seen, per)
    keys = {(ln.input.text, ln.input.previous) for ln, _ in first}
    assert not keys & {(ln.input.text, ln.input.previous) for ln, _ in again}


def test_select_takes_marked_conversations_first_and_caps_each_domain():
    convs = [
        dict(unmarked(conv()), prompt_id="tv1-isp_support-000"),
        dict(conv(), prompt_id="tv1-isp_support-001"),
    ]
    lines, meta, counts = voice.select(convs, per_domain=1)
    assert counts["conversations"] == 1
    assert {m["prompt_id"] for m in meta} == {"tv1-isp_support-001"}
    assert len(lines) == 3 and len(meta) == 3


def test_select_fills_each_seed_quota_before_the_rest(monkeypatch):
    monkeypatch.setattr(voice, "QUOTA", {"sarcasm": 1, "lookalike": 1, "none": 1})
    convs = []
    for k in range(6):
        seed = ("sarcasm", "lookalike", "none")[k % 3]
        c = conv(OUTPUT.replace("eight", f"eight {k}").replace("Priya", f"Priya {k}"), seed=seed)
        # The first sarcasm prompt lost its mark.
        convs.append(dict(c if k else unmarked(c), prompt_id=f"tv1-isp_support-{k:03d}"))
    lines, meta, counts = voice.select(convs, per_domain=3)
    assert sorted({m["prompt_id"] for m in meta}) == [
        "tv1-isp_support-001",
        "tv1-isp_support-002",
        "tv1-isp_support-003",
    ]
