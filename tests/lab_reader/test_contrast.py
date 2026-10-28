"""r3c's contrast pairs, on made-up teacher output: their prompts, parser and line picker, the praise cases, the line
before the student keeps for them, and how they are read apart from the other sets."""

import json
import random
from collections import Counter

from lab.reader import data, model, sources, support
from lab.reader.data import CONTRAST_TAG, LinesR2, eval_groups_r2
from lab.reader.voice import norm_key

MOMENTS = [
    {
        "turn": "Oh, lovely, thanks.",
        "agent_meant": "I've waived the fee this time.",
        "agent_opposite": "There's a sixty dollar fee for that.",
    },
    {
        "turn": "Thanks, that's a big help.",
        "agent_meant": "I've sent the link to your email just now.",
        "agent_opposite": "You'll need to call the store directly about that.",
    },
    {
        "turn": "Brilliant.",
        "agent_meant": "It's arriving tomorrow, a day early.",
        "agent_opposite": "It's been pushed back another two weeks.",
    },
    {
        "turn": "Marvellous, cheers.",
        "agent_meant": "That's booked in for Thursday morning.",
        "agent_opposite": "The next free slot is in a month.",
    },
    {
        "turn": "Great, you've made my day.",
        "agent_meant": "Your claim's been approved this afternoon.",
        "agent_opposite": "I can't make an exception on that one.",
    },
    {
        "turn": "Superb.",
        "agent_meant": "I've put a ten dollar credit on your account.",
        "agent_opposite": "I don't have any information on that.",
    },
]
ITEM = {
    "turn_kind": "praise_thanks",
    "words": ["lovely", "thanks"],
    "meant": "favour",
    "opposite": "bad_news",
}


def row(moments=None, domain="online_shop", prompt_id="tc1-online_shop-0000", asr=False):
    ms, _ = support.parse_contrast(json.dumps({"moments": MOMENTS if moments is None else moments}))
    return {
        "prompt_id": prompt_id,
        "domain": domain,
        "agent": "Ava",
        "company": "Tidewater Supply",
        "scenario": "s",
        "items": [dict(ITEM) for _ in range(support.CONTRAST_ITEMS)],
        "asr": asr,
        "moments": ms,
    }


def test_prompts_fix_each_turns_kind_words_and_both_lines():
    ps = support.build_contrast_prompts(3)
    assert len(ps) == 3 * len(support.DOMAINS)
    assert ps == support.build_contrast_prompts(3)
    assert len({p["prompt_id"] for p in ps}) == len(ps) and all(p["prompt_id"].startswith("tc1-") for p in ps)
    assert all(len(p["items"]) == support.CONTRAST_ITEMS for p in ps)
    items = [it for p in ps for it in p["items"]]
    assert {it["turn_kind"] for it in items} == set(support.CONTRAST_TURNS)
    assert {it["meant"] for it in items} == set(support.CONTRAST_MEANT)
    assert {it["opposite"] for it in items} == set(support.CONTRAST_OPPOSITE)
    for p in ps:
        text = p["messages"][1]["content"]
        assert all(f'"{w}"' in text for it in p["items"] for w in it["words"])
    by_domain = Counter(p["domain"] for p in ps if p["asr"])
    assert set(by_domain.values()) == {1}


def test_the_examples_shown_are_none_of_the_cases():
    cases = [*support.PROBES, *support.PRAISE_PROBES]
    turns = {norm_key(e) for v in support.CONTRAST_TURNS.values() for e in v["examples"]}
    lines = {
        norm_key(e)
        for t in (support.CONTRAST_MEANT, support.CONTRAST_OPPOSITE)
        for v in t.values()
        for e in v["examples"]
    }
    assert not turns & {norm_key(t) for _, t, _, _ in cases}
    assert not lines & {norm_key(p) for _, _, p, _ in cases if p}


def test_the_parser_keeps_whole_moments_only():
    assert support.parse_contrast("not json") == ([], ["not json"])
    assert support.parse_contrast(json.dumps({"moments": "x"})) == ([], ["no moments"])
    ms, problems = support.parse_contrast(json.dumps({"moments": [MOMENTS[0], {"turn": "Hi."}, "x"]}))
    assert ms == [MOMENTS[0]] and problems == ["bad moment", "bad moment"]


def test_a_pair_is_one_turn_after_both_lines_the_meant_one_first():
    meant, opposite = support.contrast_pair(row(), 0)
    assert meant["text"] == opposite["text"] == "Oh, lovely, thanks."
    assert (meant["side"], opposite["side"]) == ("meant", "opposite")
    assert meant["previous"] == "I've waived the fee this time."
    assert opposite["previous"] == "There's a sixty dollar fee for that."
    assert (meant["agent_kind"], opposite["agent_kind"]) == ("favour", "bad_news")


def test_questions_long_turns_twin_lines_and_markup_give_no_pair():
    def one(**change):
        return support.contrast_pair(row([{**MOMENTS[0], **change}]), 0)

    assert one(turn="Oh, lovely, isn't it?") is None
    assert one(turn=" ".join(["lovely"] * (support.CONTRAST_MAX_WORDS + 1))) is None
    assert one(agent_opposite=MOMENTS[0]["agent_meant"]) is None
    assert one(turn="*sighs* Lovely.") is None
    assert support.contrast_pair(row([MOMENTS[0]]), 1) is None


def test_a_speech_to_text_turn_reads_as_one():
    meant, _ = support.contrast_pair(row(asr=True), 0)
    assert meant["text"] == "oh lovely thanks"
    assert meant["previous"] == "I've waived the fee this time."


def test_both_lines_of_a_pair_name_their_side_and_share_the_pair():
    meant, opposite = support.contrast_pair(row(domain="rail_travel", prompt_id="tc1-rail_travel-0000"), 0)
    a = support.contrast_line(row(domain="rail_travel"), meant, "test")
    b = support.contrast_line(row(domain="rail_travel"), opposite, "test")
    assert a.line_id != b.line_id and support.pair_of(a.source_ref) == support.pair_of(b.source_ref)
    assert {"contrast", "reply_contrast_meant", "lookalike_seed", "held_domain"} <= set(a.slices)
    assert {"contrast", "reply_contrast_opposite", "sarcasm_seed"} <= set(b.slices)
    assert (
        a.input.previous == "Ava: I've waived the fee this time."
        and a.generator == "contrast:meant:praise_thanks"
    )
    m = support.contrast_meta(row(domain="rail_travel"), opposite, b)
    assert m["sarcastic_by_construction"] is True and m["before"] == [
        {"who": "agent", "text": opposite["previous"]}
    ]


def test_select_takes_both_lines_of_a_pair_or_neither():
    banned = (norm_key(MOMENTS[1]["turn"]), norm_key(MOMENTS[1]["agent_opposite"]))
    keeper = support.Keeper(banned_pairs={banned}, max_short=support.CONTRAST_MAX_SHORT)
    rows = [row(prompt_id=f"tc1-online_shop-{j:04d}") for j in range(40)]
    train = next(r for r in rows if support.split_of(r) == "train")
    lines, metas, counts = support.select_contrast([train], keeper, 100)
    assert len(lines) == 2 * (support.CONTRAST_ITEMS - 1) and counts["pairs"] == support.CONTRAST_ITEMS - 1
    assert counts["pair_not_new"] == 1
    sides = Counter(support.pair_of(ln.source_ref) for ln in lines)
    assert set(sides.values()) == {2}
    assert [m["side"] for m in metas[:2]] == ["meant", "opposite"]


def test_dev_and_test_pairs_leave_out_turns_another_set_trains_on():
    seen = support.bans([(MOMENTS[1]["turn"], "something else")])
    held = row(domain="rail_travel", prompt_id="tc1-rail_travel-0000")
    lines, _, counts = support.select_contrast([held], support.Keeper(), 100, seen)
    assert counts["trained_elsewhere"] == 1 and len(lines) == 2 * (support.CONTRAST_ITEMS - 1)
    assert all(ln.split == "test" for ln in lines)


def test_a_short_turn_comes_up_in_at_most_four_pairs():
    rows = []
    for j in range(20):
        ms = [
            {
                **m,
                "turn": "Superb.",
                "agent_meant": f"{m['agent_meant']} Ref {j}{k}.",
                "agent_opposite": f"{m['agent_opposite']} Ref {j}{k}.",
            }
            for k, m in enumerate(MOMENTS)
        ]
        rows.append(row(ms, prompt_id=f"tc1-online_shop-{j:04d}"))
    keeper = support.Keeper(max_short=support.CONTRAST_MAX_SHORT)
    lines, _, _ = support.select_contrast(rows, keeper, 1000)
    assert len(lines) == support.CONTRAST_MAX_SHORT


def test_the_praise_cases_are_kept_out_of_contrast_training():
    ids = [pid for pid, *_ in support.PRAISE_PROBES]
    assert len(set(ids)) == len(ids)
    assert {w for *_, w in support.PRAISE_PROBES} == {"sarcastic", "sincere"}
    groups = {pid.rsplit("_", 1)[0] for pid in ids}
    assert groups == {"after_good", "alone", "after_bad", "own_words"}
    texts, pairs = support.praise_bans()
    for _, t, p, _ in support.PRAISE_PROBES:
        assert (norm_key(t), norm_key(p or "")) in pairs
        assert (norm_key(t) in texts) == (len(norm_key(t).split()) >= support.BAN_WORDS)


# --- the student's reading of them --------------------------------------------------------------------------------


def lines_with(*examples) -> LinesR2:
    lines = LinesR2()
    for ex in examples:
        lines.by_id[ex.line_id] = ex
    return lines


def ex(line_id, slices, sarc, previous="Ava's line", split="train", sets=()):
    e = data.Example(line_id, "Lovely.", previous, split, slices=tuple(slices), sets=set(sets))
    e.sarc = sarc
    return e


def test_only_lines_sarcastic_after_the_agents_line_keep_it():
    lines = lines_with(
        ex("a", ["reply_contrast_opposite"], 1),
        ex("b", ["reply_contrast_meant"], 0),
        ex("c", ["reply_sarcastic_news"], 1),
        ex("d", ["reply_sarcastic_own"], 1),
        ex("e", ["reply_contrast_opposite"], 0),
        ex("f", ["reply_contrast_opposite"], 1, previous=None),
    )
    assert lines.keep_needed_previous() == 2 and lines.counts["keep_previous"] == 2
    assert [k for k, e in lines.by_id.items() if e.keep_previous] == ["a", "c"]


def test_a_kept_line_before_is_never_dropped_and_the_others_drop_as_before():
    chunk = [ex(str(i), [], 1, previous="x" if i % 5 else None) for i in range(400)]
    plain = model.kept_previous(chunk, random.Random(3), 0.5)
    for e in chunk[::2]:
        e.keep_previous = True
    marked = model.kept_previous(chunk, random.Random(3), 0.5)
    assert all(marked[i] for i in range(0, 400, 2))
    assert [marked[i] for i in range(1, 400, 2)] == [plain[i] for i in range(1, 400, 2)]
    assert not all(plain)


def test_contrast_lines_are_read_apart_from_the_others():
    exs = [
        ex("t", [], 1, split="dev", sets={"act_emotion", "sarcasm"}),
        ex("c1", ["reply_contrast_opposite"], 1, split="dev", sets={CONTRAST_TAG, "act_emotion", "sarcasm"}),
        ex("c2", ["reply_contrast_meant", "held_domain"], 0, split="test", sets={CONTRAST_TAG}),
    ]
    for e in exs:
        e.emo = 0
    dev, test = eval_groups_r2(exs, "dev"), eval_groups_r2(exs, "test")
    assert dev["contrast_sarcasm"] == [1] and dev["contrast_emotion"] == [1]
    assert dev["sarcasm"] == [0] and dev["emotion"] == [0]
    assert test["contrast_sarcasm_held_domains"] == [2]


def test_r3c_keeps_r3s_ids_and_reads_r2s_teachers():
    from lab.reader.pipeline import support_label

    assert (
        support_label.SETS == sources.SUPPORT_SETS
        and support_label.sets_of(sources.CONTRAST_ID) == sources.CONTRAST_SETS
    )
    assert support_label.PILOT_SETS == {q: f"{sources.SUPPORT_ID}-pilot-{q}" for q in sources.SUPPORT_SETS}
    assert sources.contrast_key("lines.jsonl") == f"runs/tonetext/generated/{sources.CONTRAST_ID}/lines.jsonl"
    assert data.r2_soft_id(recipe="r3c") == sources.R2_SOFT_ID
