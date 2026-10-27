"""r2's committee labels: which open turns are chosen and how the committee's code reads them. Synthetic lines in the
open sets' shape; no volume, no Modal, no calls."""

import random
from collections import Counter

from lab.reader import open_sources, r2_labels
from lab.reader.committee import label as committee_label
from lab.reader.schemas import LineInput

from . import open_line


def line(
    src, conv, i, role, modality, split, text=None, previous=None, native=None, ref=None, participants=None
):
    if participants:
        speaker = participants[i % len(participants)]
        participants = [speaker] + [p for p in participants if p != speaker]
    tag = "asr-" if modality == "asr" and src == "harpervalley" else ""
    return open_line(
        f"{src}-{conv}-{tag}{i}",
        text or f"{src} {conv} turn {i} says something",
        src,
        role,
        modality,
        split,
        previous,
        conv,
        "dom_1",
        native,
        ref or f"f#{conv}/{i}",
        "x",
        participants,
    )


def voice_source(src, convs=40, turns=8, splits=("dev", "test", "train"), user_mod="spoken"):
    out = []
    for c in range(convs):
        split = splits[c % len(splits)]
        prev = None
        for i in range(turns):
            role = "user" if i % 2 == 0 else "agent"
            if src == "harpervalley":
                mod = "spoken"  # its human transcript, both sides; the machine transcript follows as asr
            else:
                mod = (
                    (user_mod if role == "user" else "written") if c % 5 else "written"
                )  # a fifth of conversations all written
            ln = line(src, f"c{c}", i, role, mod, split, previous=prev)
            out.append(ln)
            prev = f"{ln['input']['speaker']}: {ln['input']['text']}"
            if src == "harpervalley":
                out.append(
                    line(
                        src,
                        f"c{c}",
                        i,
                        role,
                        "asr",
                        split,
                        text=ln["input"]["text"] + " uh",
                        previous=ln["input"]["previous"],
                        ref=ln["source_ref"] + "/asr",
                    )
                )
    return out


def every_source(convs=60):
    return {s: voice_source(s, convs=convs) for s in r2_labels.VOICE_SOURCES}


def test_lines_are_valid_open_lines_with_digit_slices():
    for ln in voice_source("multiwoz22", convs=3) + voice_source("harpervalley", convs=3):
        open_sources.OpenLine.model_validate(ln)
        LineInput.model_validate(ln["input"])
    assert open_sources.role_of(line("taskmaster1", "c", 0, "agent", "written", "dev")) == "agent"
    assert open_sources.modality_of(line("harpervalley", "c", 0, "user", "asr", "dev")) == "asr"


def test_allocate_and_weights():
    assert sum(r2_labels.allocate(2000, r2_labels.HELD_QUOTA).values()) == 2000
    assert (
        r2_labels.allocate(800, r2_labels.HELD_QUOTA)["natcs"] == 160
        and sum(r2_labels.allocate(50_000, r2_labels.HELD_QUOTA).values()) == 50_000
    )
    assert sum(r2_labels.HELD_QUOTA.values()) == 1500


def test_held_sets_come_from_their_split_half_user_half_agent_spoken_users():
    by = every_source()
    p = r2_labels.Picker()
    counts = r2_labels.held_set(by, "dev", p)
    rows = p.rows
    assert all(r["line"]["split"] == "dev" and r["set"] == "dev" for r in rows)
    roles = Counter((r["line"]["source"], open_sources.role_of(r["line"])) for r in rows)
    # the synthetic sources are small: every source gives what it has up to its quota, both roles equally
    for src in r2_labels.VOICE_SOURCES:
        assert (
            roles[(src, "user")] == counts[f"{src}/user"]
            and abs(roles[(src, "user")] - roles[(src, "agent")]) <= 1
        )
    users = [
        r for r in rows if open_sources.role_of(r["line"]) == "user" and r["line"]["source"] != "harpervalley"
    ]
    spoken = [r for r in users if open_sources.modality_of(r["line"]) in r2_labels.SPOKEN]
    assert spoken and all(r["why"]["preferred"] == "spoken" for r in spoken)
    # written users only once the spoken ones of that source ran out
    for src in r2_labels.VOICE_SOURCES[:-1]:
        mine = [r for r in users if r["line"]["source"] == src]
        avail = [
            ln
            for ln in by[src]
            if ln["split"] == "dev"
            and open_sources.role_of(ln) == "user"
            and open_sources.modality_of(ln) == "spoken"
        ]
        if len(mine) > len(avail):
            assert sum(open_sources.modality_of(r["line"]) == "spoken" for r in mine) == len(avail)


def test_harper_valley_readings_never_both_and_half_machine_transcript():
    by = {"harpervalley": voice_source("harpervalley", convs=90)}
    p = r2_labels.Picker()
    r2_labels.held_set(by, "test", p)
    segs = [r2_labels.segment(r["line"]) for r in p.rows]
    assert len(segs) == len(set(segs))
    users = [r for r in p.rows if open_sources.role_of(r["line"]) == "user"]
    assert (
        sum(open_sources.modality_of(r["line"]) == "asr" for r in users)
        == r2_labels.HELD_QUOTA["harpervalley"] // 4
    )
    assert all(
        open_sources.modality_of(r["line"]) != "asr"
        for r in p.rows
        if open_sources.role_of(r["line"]) == "agent"
    )


def test_spread_takes_one_line_per_conversation_first():
    lines = voice_source("star", convs=30, splits=("dev",))
    p = r2_labels.Picker()
    r2_labels.spread(
        [ln for ln in lines if open_sources.role_of(ln) == "user"],
        30,
        random.Random(0),
        p,
        "dev",
        "voice_agent",
        {"rule": "x"},
    )
    assert len({r["line"]["group_key"] for r in p.rows}) == 30


def test_no_line_twice_pairs_once_and_text_cap():
    a = line("abcd", "c1", 0, "user", "written", "train", text="Okay.", previous="Agent: hi")
    b = dict(a, line_id="abcd-c2-0", group_key="abcd:c2")  # the same text and line before
    others = [
        line("abcd", f"d{k}", 0, "user", "written", "train", text="okay", previous=f"Agent: {k}")
        for k in range(6)
    ]
    p = r2_labels.Picker()
    taken = r2_labels.spread([a, b, *others], 10, random.Random(1), p, "train", "voice_agent", {"rule": "x"})
    assert taken == r2_labels.TEXT_CAP and len(p.ids) == r2_labels.TEXT_CAP
    assert not (a["line_id"] in p.ids and b["line_id"] in p.ids)
    assert not p.ok(a, "train") and p.ok(
        dict(a, line_id="new", source_ref="f#new/0", input={**a["input"], "text": "fine then"}), "train"
    )
    # the cap is per set: the same text may still go to dev
    assert p.ok(
        dict(
            others[0],
            line_id="dev-1",
            source_ref="f#dev/1",
            input={**others[0]["input"], "previous": "Agent: other"},
        ),
        "dev",
    )


def test_sarcasm_pairs_stay_together():
    lines = []
    for k in range(40):
        lines.append(
            line(
                "isarcasmeval",
                f"train-{k}",
                0,
                "user",
                "written",
                "train",
                text=f"oh great {k}",
                native={"label": 1},
            )
        )
        lines.append(
            line(
                "isarcasmeval",
                f"train-{k}",
                1,
                "user",
                "written",
                "train",
                text=f"I am sad {k}",
                native={"label": 0},
            )
        )
    lines += [
        line(
            "isarcasmeval",
            f"single-{k}",
            0,
            "user",
            "written",
            "train",
            text=f"nice day {k}",
            native={"label": 0},
        )
        for k in range(30)
    ]
    for ln in lines:
        ln["native"] = {"scheme": "isarcasmeval-sarcasm", "label": ln["native"]["label"]}
    p = r2_labels.Picker()
    out = r2_labels.train_sarcasm(lines, p)
    assert out["pairs"] == 40 and out["singles"] == 30  # fewer than the quotas here: all of them
    by_group = Counter(r["line"]["group_key"] for r in p.rows if r["why"]["rule"] == "train_sarcasm_pair")
    assert set(by_group.values()) == {2}
    first = [r for r in p.rows if r["why"]["rule"] == "train_sarcasm_pair"][0]
    assert first["why"]["native"] == 1  # the sarcastic post comes before its rephrase


def test_sentiment_is_even_and_a_short_label_passes_its_share_on():
    lines = []
    for lab, n in (("positive", 200), ("negative", 200), ("neutral", 200), ("mixed", 20)):
        lines += [
            line(
                "dynasent_r2",
                f"{lab}{k}",
                0,
                "user",
                "written",
                "train",
                text=f"{lab} sentence {k}",
                native={"scheme": "dynasent_r2-sentiment", "label": lab},
            )
            for k in range(n)
        ]
    lines.append(line("dynasent_r2", "nolabel", 0, "user", "written", "train", text="no label here"))
    p = r2_labels.Picker()
    out = r2_labels.train_sentiment(lines, p)
    assert out["by_label"]["mixed"] == 20 and sum(out["by_label"].values()) == r2_labels.DYNASENT
    assert max(out["by_label"].values()) - min(v for k, v in out["by_label"].items() if k != "mixed") <= 1
    assert "dynasent_r2-nolabel-0" not in p.ids


def test_unsure_takes_each_criterion_in_turn_per_source():
    pool = {
        s: [ln for ln in voice_source(s, convs=300) if ln["split"] == "train"]
        for s in r2_labels.VOICE_SOURCES
    }
    p = r2_labels.Picker()
    cands = r2_labels.unsure_candidates(pool, p)
    assert len(cands) == sum(len(v) for v in pool.values())  # fewer than 50,000 here: every free line
    rng = random.Random(3)
    scores = {
        ln["line_id"]: {
            "emotion_confidence": rng.random(),
            "move_confidence": rng.random(),
            "sarcasm_raw": rng.random(),
        }
        for ln in cands
    }
    counts = r2_labels.pick_unsure(cands, scores, 0.53, p)
    rows = [r for r in p.rows if r["group"] == "unsure"]
    assert len(rows) == min(r2_labels.UNSURE, len(cands))
    for r in rows:
        w = r["why"]
        if w["criterion"] == "sarcasm":
            assert abs(w["sarcasm_raw"] - 0.53) <= r2_labels.SARCASM_BAND and w["sarcasm_near_threshold"]
        if w["criterion"] == "move":
            assert w["move_confidence"] < r2_labels.MOVE_UNSURE
    crit = Counter(r["why"]["criterion"] for r in rows if r["line"]["source"] == "natcs")
    assert (
        max(crit.values()) - min(crit.values()) <= 1
        and sum(crit.values()) == r2_labels.allocate(r2_labels.UNSURE, r2_labels.HELD_QUOTA)["natcs"]
    )
    assert counts["natcs"]["candidates"] == len(pool["natcs"])


def test_choose_keeps_every_rule_across_sets():
    by = every_source(convs=90)
    pool = {s: [ln for ln in v if ln["split"] == "train"] for s, v in by.items()}
    pool["ami"] = [
        line("ami", f"m{m}", i, "participant", "spoken", "train", participants=["Alex", "Blair", "Casey"])
        for m in range(5)
        for i in range(100)
    ]
    pool["dipco"] = [
        line("dipco", "s1", i, "participant", "spoken", "train", participants=["Alex", "Blair"])
        for i in range(150)
    ]
    p, counts = r2_labels.choose(
        by,
        pool,
        lambda cs: {
            ln["line_id"]: {"emotion_confidence": 0.5, "move_confidence": 0.4, "sarcasm_raw": 0.5}
            for ln in cs
        },
        0.53,
    )
    ch = p.chosen()
    assert len({r["line_id"] for r in ch}) == len(ch)
    assert len({(r["line"]["input"]["text"], r["line"]["input"]["previous"]) for r in p.rows}) == len(ch)
    assert all(r["set"] == r["split"] for r in ch)  # dev from dev, test from test, training from train
    assert {r["set"] for r in ch} == {"dev", "test", "train"}
    for r in p.rows:
        LineInput.model_validate(r["line"]["input"])
    assert counts["train_multi_party"]["ami"] == 400 and counts["train_multi_party"]["dipco"] == 100
    t = r2_labels.tally(ch)
    assert t["lines"] == len(ch) and sum(t["by_set"].values()) == len(ch)
    ids = r2_labels.pilot_ids(ch)
    assert len(ids) == 30 and all(ch[i]["set"] != "test" for i in ids)


def labelled_rows(n=60, seed=0):
    """Synthetic labelled rows in Tone's formats over r2's splits and roles."""
    rng = random.Random(seed)
    act, xch, votes, sar = [], [], [], []
    for i in range(n):
        split = ("dev", "test", "train")[i % 3]
        role = (
            ("user", "agent", "participant")[(i // 3) % 3]
            if split == "train"
            else ("user", "agent")[(i // 3) % 2]
        )
        ln = line(
            "multiwoz22" if role != "participant" else "ami",
            f"c{i}",
            i,
            role,
            "written",
            split,
            participants=["Alex", "Blair"] if role == "participant" else None,
        )
        e = {m: rng.choice(["neutral", "happy", "angry"]) for m in "abd"}
        lab = Counter(e.values()).most_common(1)[0]
        act.append(
            {
                "line": ln,
                "complete": True,
                "emotion": {
                    "label": lab[0] if lab[1] >= 2 else None,
                    "intensity": "low",
                    "agree": lab[1],
                    "votes": e,
                },
                "acts": {
                    t: {"label": "none", "intensity": "low", "agree": 3, "votes": {m: "none" for m in "abd"}}
                    for t in ln["input"]["targets"]
                },
                "backchannel": {
                    "label": "false",
                    "intensity": None,
                    "agree": 3,
                    "votes": {m: "false" for m in "abd"},
                },
                "label_source": "committee",
            }
        )
        mv = {m: rng.choice(["answers", "asks", "other"]) for m in "abd"}
        top = Counter(mv.values()).most_common(1)[0]
        v = {"label": top[0] if top[1] >= 2 else None, "agree": top[1], "votes": mv}
        votes.append({"line_id": ln["line_id"], **v, "d_probabilities": None})
        xch.append({"line": ln, "exchange": v, "label_source": "committee"})
        sv = {m: rng.random() < 0.2 for m in "abd"}
        yes = sum(sv.values())
        sar.append(
            {
                "line": ln,
                "sarcasm": {
                    "label": True if yes >= 2 else False,
                    "votes": sv,
                    "agree": max(yes, 3 - yes),
                    "d_probability": 0.3,
                },
                "label_source": "committee",
            }
        )
    return act, xch, votes, sar


def test_r2_summaries_read_each_split_and_role():
    act, xch, votes, sar = labelled_rows()
    fns = r2_labels.summaries({r["line"]["line_id"]: "voice_agent" for r in act})
    c, x, s = fns["committee"](act), fns["exchange"](xch, votes), fns["sarcasm"](sar)
    assert set(c["by_split"]) == {"dev", "test", "train"} and set(x["by_split"]) == set(s["by_split"])
    assert set(c["by_split"]["train"]["emotion_by_role"]) == {"user", "agent", "participant"}
    assert set(c["by_split"]["dev"]["emotion_by_role"]) == {"user", "agent"}
    assert "emotion_participant_lines" in c["by_split"]["train"]["a_b"]
    assert x["by_split"]["test"]["a_b"]["committee_level"]["of"] == sum(
        r["line"]["split"] == "test" for r in xch
    )
    assert x["all"]["a_b"]["committee_level"]["of"] == len(xch)
    assert "agent_lines" in s["by_split"]["dev"]["a_b"] and "sarcasm_seed_lines" not in s["all"]
    assert c["by_group"]["dev/voice_agent"]["lines"] == sum(r["line"]["split"] == "dev" for r in act)


def test_the_voice_sets_defaults_are_unchanged():
    act, xch, votes, sar = labelled_rows()
    two = [r for r in act if r2_labels.row_role(r) != "participant"]
    assert list(committee_label.act_emotion_summary(two)["a_b"]) == [
        "emotion_all_lines",
        "emotion_majority_lines",
        "emotion_user_lines",
        "emotion_agent_lines",
        "act_all_lines",
        "definition",
    ]
    xs = [r for r in xch if r2_labels.row_role(r) != "participant"]
    assert list(committee_label.exchange_summary(xs, votes)["a_b"]) == [
        "committee_level",
        "user_lines",
        "agent_lines",
        "definition",
    ]
    s = committee_label.sarcasm_summary([r for r in sar if r2_labels.row_role(r) != "participant"])
    assert list(s) == [
        "lines",
        "majority",
        "sarcastic_by_role",
        "sarcasm_seed_lines",
        "lookalike_seed_lines",
        "other_lines_sarcastic",
        "a_b",
        "member_answered",
    ] and list(s["a_b"]) == ["committee_level", "all_lines", "user_lines", "definition"]
