"""The support-call set's prompts, parser and line picker, on made-up teacher output."""

import json
from collections import Counter

from lab.reader import support
from lab.reader.voice import norm_key

OUTPUT = {
    "before": [
        {
            "agent": "Hi, thanks for calling Tidewater Supply, this is Ava. How can I help?",
            "user": "Hi, yeah. I ordered a jacket last week and it still hasn't shipped.",
        },
        {"agent": "I can look into that. What's the order number?", "user": "It's four eight two one three."},
        {"agent": "Thanks. And the name on the order?", "user": "Dana Reyes."},
    ],
    "moment": {
        "agent": "It's still being processed, so it hasn't shipped yet. What would you like to do?",
        "user": "Oh, brilliant. So it'll get here after my trip, then. Love that.",
        "user_instead": "That's fine, I'm not in a rush. I'll wait.",
    },
    "after": [
        {"agent": "I can ask the warehouse to prioritise it.", "user": "Yeah, do that, please."},
        {"agent": "Done. Is there anything else?", "user": "No, that's all."},
    ],
    "agent_last": "Thanks for calling, have a good day.",
}


def call(output=None, asr=False, domain="online_shop", prompt_id="ts1-online_shop-0000", moment="status"):
    turns, instead, problems = support.parse_call(json.dumps(output or OUTPUT))
    return {
        "prompt_id": prompt_id,
        "domain": domain,
        "agent": "Ava",
        "company": "Tidewater Supply",
        "scenario": "s",
        "persona": "p",
        "arc": "frustrated",
        "moment": moment,
        "main": "sarcastic_own",
        "instead": "sincere_accepting",
        "asr": asr,
        "turns": turns,
        "instead_text": instead,
        "problems": problems,
    }


def test_every_moment_pairs_a_sarcastic_answer_with_a_sincere_one():
    for m in support.MOMENTS.values():
        for main in m["main"]:
            for instead in m["instead"]:
                assert support.REPLIES[main]["sarcastic"] != support.REPLIES[instead]["sarcastic"]


def test_prompts_cover_every_domain_kind_and_speech_to_text():
    ps = support.build_prompts(9)
    assert len(ps) == 9 * len(support.DOMAINS)
    assert len({p["prompt_id"] for p in ps}) == len(ps)
    assert set(p["moment"] for p in ps) == set(support.MOMENTS)
    assert Counter(p["asr"] for p in ps)[True] == 3 * len(support.DOMAINS)
    for p in ps:
        text = p["messages"][1]["content"]
        assert '"user_instead"' in text and p["agent"] in text
        assert ("not bad news itself" in text) == support.MOMENTS[p["moment"]]["plain"]
    assert support.build_prompts(9) == ps  # fixed by the seed


def test_the_teachers_json_holds_both_answers():
    assert support.schema()["properties"]["moment"]["required"] == ["agent", "user", "user_instead"]
    c = call()
    assert c["instead_text"] == "That's fine, I'm not in a rush. I'll wait."
    marked = [t for t in c["turns"] if t["marked"]]
    assert len(marked) == 1 and marked[0]["text"].startswith("Oh, brilliant.")
    assert support.parse_call("not json") == ([], "", ["not json"])


def test_both_answers_follow_the_same_agent_line():
    main, instead = support.moment_lines(call())
    assert main["previous"] == instead["previous"] == OUTPUT["moment"]["agent"]
    assert (main["reply"], instead["reply"]) == ("sarcastic_own", "sincere_accepting")
    assert main["i"] == instead["i"]


def test_a_plain_line_that_carries_news_gives_no_moment():
    out = json.loads(json.dumps(OUTPUT))
    out["moment"]["agent"] = "Unfortunately it's still being processed."
    assert support.moment_lines(call(out)) is None
    assert support.moment_lines(call(out, moment="bad_news")) is not None  # news is what that kind carries


def test_a_moment_the_call_says_twice_gives_none():
    out = json.loads(json.dumps(OUTPUT))
    out["before"].append({"agent": out["moment"]["agent"], "user": out["moment"]["user"]})
    assert support.moment_lines(call(out)) is None


def test_stage_directions_and_twins_give_no_moment():
    out = json.loads(json.dumps(OUTPUT))
    out["moment"]["user_instead"] = "(sighs) Fine."
    assert support.moment_lines(call(out)) is None
    out["moment"]["user_instead"] = out["moment"]["user"]
    assert support.moment_lines(call(out)) is None


def test_lines_follow_tones_conventions_and_name_their_kinds():
    c = call(asr=True)
    main, instead = support.moment_lines(c)
    a, b = support.corpus_line(c, main, "train"), support.corpus_line(c, instead, "train")
    assert a.input.speaker == "player" and a.input.to == "Ava" and a.input.targets == ["Ava"]
    assert a.input.previous == "Ava: " + OUTPUT["moment"]["agent"]
    assert a.input.text == a.input.text.lower() and "," not in a.input.text  # speech to text
    assert "/instead" in b.source_ref and a.line_id != b.line_id and a.group_key == b.group_key
    assert {"sarcasm_seed", "reply_sarcastic_own", "moment_status", "asr_style"} <= set(a.slices)
    assert {"lookalike_seed", "reply_sincere_accepting"} <= set(b.slices)
    assert a.generator == "support:status:sarcastic_own"


def test_a_lines_meta_keeps_the_turns_before_it_without_names():
    c = call()
    main, _ = support.moment_lines(c)
    m = support.meta(c, main, support.corpus_line(c, main, "train"))
    assert [t["who"] for t in m["before"]] == ["agent", "user"] * 3 + ["agent"]
    assert m["before"][-1]["text"] == OUTPUT["moment"]["agent"]
    assert m["sarcastic_by_construction"] is True and m["answered_instead"] is False


def test_splits_are_by_call_and_the_held_domains_are_test_only():
    calls = [{"prompt_id": f"ts1-online_shop-{i:04d}", "domain": "online_shop"} for i in range(4000)]
    splits = Counter(support.split_of(c) for c in calls)
    assert 0.01 < splits["test"] / 4000 < 0.03 and 0.04 < splits["dev"] / 4000 < 0.06
    for d in support.HELD_DOMAINS:
        assert support.split_of({"prompt_id": f"ts1-{d}-0001", "domain": d}) == "test"


def test_select_takes_both_answers_and_keeps_bans_out_of_training():
    calls = [call(prompt_id=f"ts1-online_shop-{i:04d}") for i in range(3)]
    lines, metas, counts = support.select(calls, support.Keeper(), 100)
    assert counts["calls"] == 1  # the same two answers again are repeats
    assert len({ln.input.text for ln in lines}) == len(lines) >= 2
    banned = support.bans([(OUTPUT["moment"]["user_instead"], None)])
    lines, _, counts = support.select(calls[:1], support.Keeper(*banned), 100)
    assert support.split_of(calls[0]) != "train" or lines == []


def test_dev_and_test_texts_are_kept_out_of_training():
    keeper = support.Keeper()
    cand = {"text": "Oh, brilliant. So it'll get here after my trip, then.", "previous": "x"}
    keeper.take(cand, "test")
    assert not keeper.fresh({**cand, "previous": "y"}, "train")
    assert keeper.fresh({**cand, "previous": "y"}, "dev")


def test_short_texts_may_repeat_more_than_long_ones():
    keeper = support.Keeper()
    short, long = {"text": "Oh, wonderful."}, {"text": "Great, so it'll be late again, then."}
    for i in range(support.MAX_SHORT):
        assert keeper.fresh({**short, "previous": f"p{i}"}, "train")
        keeper.take({**short, "previous": f"p{i}"}, "train")
    assert not keeper.fresh({**short, "previous": "another"}, "train")
    for i in range(support.MAX_LONG):
        keeper.take({**long, "previous": f"p{i}"}, "train")
    assert not keeper.fresh({**long, "previous": "another"}, "train")


def test_the_probes_are_banned_from_training_but_short_ones_only_as_pairs():
    texts, pairs = support.probe_bans()
    assert norm_key("Well, that's just great, isn't it? It's gonna be late.") in texts
    assert norm_key("Oh, wonderful.") not in texts
    assert (norm_key("Oh, wonderful."), norm_key("Your delivery is delayed again.")) in pairs
    assert {want for *_, want in support.PROBES} == {"sarcastic", "sincere"}
