"""The speech round trip's pure parts: normalisation, word error rate, the draw and twin building. No models, no
network, no volume. Every text here is made up for the test."""

import pytest

from lab.reader import open_sources, speech

from . import open_line


def ids_where(pred, prefix: str = "line", n: int = 1) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        lid = f"{prefix}-{i}"
        if pred(lid):
            out.append(lid)
        i += 1
    return out


def ami_line(lid: str, text: str) -> dict:
    return {
        "line_id": lid,
        "source": "ami",
        "source_ref": "a.xml#1",
        "licence": "CC BY 4.0",
        "split": "train",
        "group_key": "ami:ES2002a",
        "input": {
            "speaker": "Alex",
            "speaker_kind": "npc",
            "to": None,
            "targets": ["Blair", "Casey"],
            "text": text,
            "previous": None,
        },
        "slices": ["ami", "participant", "spoken", "scenario_meeting"],
        "twin_of": None,
        "generator": None,
        "native": None,
    }


def pool_line(
    lid: str, text: str, kind: str = "player", twin_of: str | None = None, source: str = "generated"
) -> dict:
    speaker = "player" if kind == "player" else "Mira"
    target = "Mira" if kind == "player" else "player"
    return {
        "line_id": lid,
        "source": source,
        "source_ref": "gen:1",
        "licence": "ours",
        "split": "train",
        "group_key": "w1",
        "input": {
            "speaker": speaker,
            "speaker_kind": kind,
            "to": target,
            "targets": [target],
            "text": text,
            "previous": "Mira: Welcome back.",
        },
        "slices": [kind, "style-casual_typer"],
        "twin_of": twin_of,
        "generator": "style-casual_typer",
    }


# --- normalisation and word error rate ----------------------------------------------------------------------------------


def test_norm():
    assert speech.norm("Hi, I'd like to MOVE it -- to 3:30?") == "hi i'd like to move it to 3 30"
    assert speech.norm("It’s fine.") == "it's fine"
    assert speech.norm("e-mail_me / now!!") == "e mail me now"
    assert speech.norm(None) == "" and speech.norm("?!...") == ""
    assert speech.norm("  café  au   lait ") == "café au lait"


def test_edit_distance_and_wer():
    assert speech.edit_distance([], []) == 0
    assert speech.edit_distance(["a"], []) == 1
    assert speech.edit_distance([], ["a", "b"]) == 2
    assert speech.edit_distance("the cat sat".split(), "the cat sat".split()) == 0
    # One substitution, one insertion.
    assert speech.edit_distance("the cat sat".split(), "a cat sat down".split()) == 2
    assert speech.wer("The cat sat.", "the cat sat") == 0.0
    assert speech.wer("I want two tickets", "I want 2 tickets") == pytest.approx(0.25)
    assert speech.wer("yes", "yes yes yes") == pytest.approx(2.0)  # insertions can pass 1
    assert speech.wer("...", "") == 0.0 and speech.wer("...", "hello") == 1.0
    assert speech.wer_counts("Okay, so um, yes.", "okay so yes") == (1, 4)


def test_speakable():
    assert speech.speakable("Yes") and not speech.speakable("No") and not speech.speakable("?!?!")
    assert speech.speakable("a" * 200) and not speech.speakable("a" * 201)
    assert not speech.speakable(None)


def test_hash_choices_are_fixed_and_spread():
    ids = [f"natcs-c{i}-3" for i in range(4000)]
    share = sum(speech.lowercased(i) for i in ids) / len(ids)
    assert 0.45 < share < 0.55
    voices = {speech.voice_for(i)[0] for i in ids}
    assert voices == {v for v, _ in speech.VOICES}
    assert all(speech.voice_for(i) == speech.voice_for(i) for i in ids[:50])
    assert {lang for _, lang in speech.VOICES} == {"a", "b"}
    assert all(v.startswith(("af_", "am_")) == (lang == "a") for v, lang in speech.VOICES)


def test_channel_split():
    ids = [f"abcd-{i}-2" for i in range(4000)]
    share = sum(speech.channel_for(i) == "phone" for i in ids) / len(ids)
    assert 0.45 < share < 0.55 and {speech.channel_for(i) for i in ids} == {"phone", "wideband"}
    both = sum(speech.channel_for(i) == "phone" and speech.lowercased(i) for i in ids) / len(ids)
    assert 0.2 < both < 0.3  # independent of the lowercasing


def test_unwrap_quotes():
    assert speech.unwrap_quotes("\"'Uff, whatever, I don't care, you know that.'") == (
        "Uff, whatever, I don't care, you know that.",
        True,
    )
    assert speech.unwrap_quotes('"Did you see that?"') == ("Did you see that?", True)
    assert speech.unwrap_quotes("'Oh, shut up, all you jerk!'") == ("Oh, shut up, all you jerk!", True)
    assert speech.unwrap_quotes("\u201cFine.\u201d") == ("Fine.", True)
    assert speech.unwrap_quotes("I don't know.") == ("I don't know.", False)
    assert speech.unwrap_quotes("'cause I said so.") == ("'cause I said so.", False)
    # Spaces collapse; only quotes count.
    assert speech.unwrap_quotes("  two   spaces ") == ("two spaces", False)
    assert speech.unwrap_quotes(None) == ("", False)


# --- twins ------------------------------------------------------------------------------------------------------------


def test_make_twin_kept_both_writings():
    low, keep = ids_where(speech.lowercased, "lw")[0], ids_where(lambda x: not speech.lowercased(x), "kp")[0]
    for lid, lowered in ((low, True), (keep, False)):
        clean = open_line(lid, "I'd like to book two tickets for Friday.")
        twin, outcome, w = speech.make_twin(clean, "  I'd like to book 2 tickets for Friday!  ")
        assert outcome == "kept" and w == pytest.approx(1 / 8)
        assert twin["line_id"] == f"{lid}-asr" and twin["twin_of"] == lid
        assert twin["input"]["text"] == (
            "i'd like to book 2 tickets for friday" if lowered else "I'd like to book 2 tickets for Friday!"
        )
        assert twin["input"]["previous"] == clean["input"]["previous"]
        assert twin["slices"] == [*clean["slices"], "speech_twin"]
        assert (
            twin["generator"] == speech.GENERATOR
            and "Kokoro-82M" in speech.GENERATOR
            and "whisper-large-v3-turbo" in speech.GENERATOR
        )
        assert {
            k: v for k, v in twin.items() if k not in ("line_id", "twin_of", "input", "slices", "generator")
        } == {
            k: v for k, v in clean.items() if k not in ("line_id", "twin_of", "input", "slices", "generator")
        }
        # Not mutated.
        assert (
            clean["input"]["text"] == "I'd like to book two tickets for Friday." and clean["twin_of"] is None
        )


def test_make_twin_drops():
    clean = open_line("d-1", "Can you check my balance, please?")
    assert speech.make_twin(clean, "") == (None, "empty", None)
    assert speech.make_twin(clean, "  ... ") == (None, "empty", None)
    twin, outcome, w = speech.make_twin(clean, "can you check my balance please")
    assert (twin, outcome, w) == (None, "same_as_clean", 0.0)
    twin, outcome, w = speech.make_twin(clean, "Thank you.")
    assert twin is None and outcome == "wer_over_0.6" and w > 0.6
    # Over 400 characters.
    twin, outcome, _ = speech.make_twin(clean, "Can you check my balance? " + "Please. " * 60)
    assert twin is None and outcome in ("invalid_input", "wer_over_0.6")
    long_clean = open_line("d-2", " ".join(["word"] * 30))
    twin, outcome, w = speech.make_twin(long_clean, " ".join(["word"] * 30) + " x" * 3 + " " + "y" * 380)
    assert twin is None and outcome == "invalid_input" and w <= 0.6


def test_make_twin_unwraps_quotes():
    keep = ids_where(lambda x: not speech.lowercased(x), "uq")[0]
    clean = open_line(keep, "Did you see the look on his face?")
    twin, outcome, w = speech.make_twin(clean, "\"'Did you see the look on his face? We owe him.'")
    assert outcome == "kept" and twin["input"]["text"] == "Did you see the look on his face? We owe him."
    assert w == pytest.approx(3 / 8)
    assert speech.make_twin(clean, "\"'Did you see the look on his face?'")[1] == "same_as_clean"


def test_make_twin_line_id_limit():
    clean = open_line("a" * 118, "Where is my parcel?")
    twin, outcome, _ = speech.make_twin(clean, "Where's my parcel?")
    assert twin is None and outcome == "invalid_input"


def test_meeting_and_pool_twins_keep_their_shape():
    t, outcome, _ = speech.make_twin(
        ami_line("ami-es2002a-12", "So we need, uh, a remote control."), "So we need a remote control."
    )
    assert outcome == "kept" and t["input"]["to"] is None and t["input"]["targets"] == ["Blair", "Casey"]
    assert t["slices"] == ["ami", "participant", "spoken", "scenario_meeting", "speech_twin"]
    t, outcome, _ = speech.make_twin(
        pool_line("ge-abc", "lol you really thought that would work"),
        "LOL, you really thought that would work.",
    )
    assert outcome == "same_as_clean" and t is None
    t, outcome, _ = speech.make_twin(
        pool_line("ge-abd", "Give me the sword, Mira."), "Give me the sword, Mirror."
    )
    assert outcome == "kept" and t["generator"] == speech.GENERATOR and t["source"] == "generated"


def test_outcome_of_results():
    clean = open_line("o-1", "My card was charged twice.")
    assert speech.outcome_of(clean, None)[1] == "no_result"
    assert (
        speech.outcome_of(clean, {"line_id": "o-1", "error": "tts: boom", "seconds": 0.0})[1] == "tts_error"
    )
    assert speech.outcome_of(clean, {"line_id": "o-1", "seconds": 0.0, "transcript": None})[1] == "no_audio"
    assert (
        speech.outcome_of(
            clean, {"line_id": "o-1", "seconds": 1.4, "transcript": "My card was charged to us."}
        )[1]
        == "kept"
    )


# --- the draw ---------------------------------------------------------------------------------------------------------


def test_draw_quotas_roles_and_filters():
    lines = [open_line(f"natcs-u{i}", f"User turn number {i} here.") for i in range(40)]
    lines += [open_line(f"natcs-a{i}", f"Agent turn number {i} here.", role="agent") for i in range(40)]
    lines += [open_line("natcs-asr", "A machine transcript line.", modality="asr")]
    lines += [open_line("natcs-dev", "A dev line here.", split="dev")]
    lines += [open_line("natcs-short", "Ok"), open_line("natcs-sym", "?!?!")]
    lines += [open_line("tm1-x", "Another source, not drawn.", source="dipco", role="participant")]
    lines += [ami_line(f"ami-{i}", f"Meeting turn {i} goes here.") for i in range(10)]
    pool = [pool_line(f"pp-{i}", f"Player line {i} in the pool.") for i in range(20)]
    pool += [pool_line(f"pn-{i}", f"Character line {i} in the pool.", kind="npc") for i in range(20)]
    pool += [pool_line("pp-twin", "player line in the pool", twin_of="pp-0")]
    drawn, counts = speech.draw(
        lines, pool, open_quotas={"natcs": 9, "ami": 4}, pool_quotas={"player": 6, "npc": 2}
    )
    ids = [d["line_id"] for d in drawn]
    assert len(ids) == len(set(ids)) == 9 + 4 + 6 + 2
    assert counts["drawn"] == {
        "ami/participant": 4,
        "natcs/agent": 3,
        "natcs/user": 6,
        "pool/generated/npc": 2,
        "pool/generated/player": 6,
    }
    assert not {"natcs-asr", "natcs-dev", "natcs-short", "natcs-sym", "tm1-x", "pp-twin"} & set(ids)
    assert counts["passed_over"]["machine transcript (not spoken again)"] == 1
    assert counts["passed_over"]["not train"] == 1 and counts["passed_over"]["twin (not drawn)"] == 1
    assert counts["by_group_and_kind"] == {
        "open/agent": 3,
        "open/participant": 4,
        "open/user": 6,
        "pool/npc": 2,
        "pool/player": 6,
    }
    again, _ = speech.draw(
        list(reversed(lines)),
        list(reversed(pool)),
        open_quotas={"natcs": 9, "ami": 4},
        pool_quotas={"player": 6, "npc": 2},
    )
    assert [d["line_id"] for d in again] == ids  # fixed by the hashes, not the order read


def test_draw_tops_up_and_caps_repeated_texts():
    lines = [open_line(f"u{i}", f"User says {i}.") for i in range(10)]
    lines += [open_line(f"a{i}", "Okay.", role="agent") for i in range(10)]  # one text, ten times
    drawn, counts = speech.draw(lines, [], open_quotas={"natcs": 9}, pool_quotas={})
    roles = [open_sources.role_of(d) for d in drawn]
    # The agent's side is short: users top it up.
    assert len(drawn) == 9 and roles.count("agent") == speech.MAX_PER_TEXT
    assert counts["repeated_text_skipped"] == 10 - speech.MAX_PER_TEXT
    few_users = [open_line(f"u{i}", f"User says {i}.") for i in range(2)] + [
        open_line(f"a{i}", f"Agent says {i}.", role="agent") for i in range(10)
    ]
    drawn, _ = speech.draw(few_users, [], open_quotas={"natcs": 9}, pool_quotas={})
    assert len(drawn) == 9 and [open_sources.role_of(d) for d in drawn].count("user") == 2


def test_pick_samples_and_task():
    lines = [
        open_line(f"s{i}", f"A short sample line number {i}.", source=("natcs" if i % 2 else "abcd"))
        for i in range(40)
    ]
    lines += [open_line("long", "x" * 150)]
    picked = speech.pick_samples(lines, n=6, per_source=3)
    assert len(picked) == 6 and "long" not in picked
    t = speech.task(lines[0], sample=True)
    assert (
        t
        == {
            "line_id": "s0",
            "text": lines[0]["input"]["text"],
            "voice": speech.voice_for("s0")[0],
            "lang": speech.voice_for("s0")[1],
            "bucket": t["bucket"],
            "sample": True,
        }
        and 0 <= t["bucket"] < 1000
    )


# --- the counts ---------------------------------------------------------------------------------------------------------


def test_summarize():
    low, keep = ids_where(speech.lowercased, "sl")[0], ids_where(lambda x: not speech.lowercased(x), "sk")[0]
    drawn = [
        open_line(low, "I want to cancel my order."),
        open_line(keep, "Where is the nearest branch?", role="agent"),
        open_line("same", "Thanks so much."),
        open_line("bad", "Could you repeat that?"),
        pool_line("pp", "Hand me the lantern."),
        pool_line("none", "No result for this one."),
    ]
    results = {
        low: {
            "line_id": low,
            "voice": "af_heart",
            "seconds": 1.8,
            "transcript": "I want to cancel my order, uh.",
        },
        keep: {
            "line_id": keep,
            "voice": "bm_george",
            "seconds": 1.5,
            "transcript": "Where's the nearest branch?",
        },
        "same": {"line_id": "same", "voice": "af_bella", "seconds": 0.9, "transcript": "Thanks so much!"},
        "bad": {"line_id": "bad", "voice": "am_puck", "seconds": 1.1, "transcript": "Thank you."},
        "pp": {"line_id": "pp", "voice": "bf_emma", "seconds": 0.0, "transcript": None},
    }
    twins, summ, details = speech.summarize(drawn, results)
    assert [t["twin_of"] for t in twins] == [low, keep]
    assert summ["lines_drawn"] == 6 and summ["results"] == 5 and summ["twins_kept"] == 2
    assert summ["dropped"] == {"same_as_clean": 1, "wer_over_0.6": 1, "no_audio": 1, "no_result": 1}
    assert summ["by_source"]["natcs"] == {"drawn": 4, "kept": 2, "same_as_clean": 1, "wer_over_0.6": 1}
    assert summ["by_group_and_kind"]["pool/player"] == {"drawn": 2, "kept": 0, "no_audio": 1, "no_result": 1}
    assert summ["lowercased"] == {"twins": 1, "share": 0.5}
    assert summ["voices_kept"] == {"af_heart": 1, "bm_george": 1}
    assert summ["wer"]["all"]["lines"] == 4 and summ["wer"]["kept"]["lines"] == 2
    assert summ["wer"]["all"]["corpus_wer"] == pytest.approx((1 + 2 + 0 + 3) / (6 + 5 + 3 + 4), abs=1e-4)
    assert summ["wer"]["all"]["bins"]["0"] == 1 and summ["wer"]["all"]["bins"]["0.6-1"] == 1
    assert [d["outcome"] for d in details] == [
        "kept",
        "kept",
        "same_as_clean",
        "wer_over_0.6",
        "no_audio",
        "no_result",
    ]
    assert all("text" not in d and "transcript" not in d for d in details)
    assert summ["audio_seconds"]["total"] == pytest.approx(5.3)
    assert sum(c["drawn"] for c in summ["by_channel"].values()) == 6
    assert set(summ["wer"]["by_channel"]) <= {"phone", "wideband"} and summ["quotation_marks_removed"] == 0
    assert [d["channel"] for d in details] == [speech.channel_for(d["line_id"]) for d in details]
    results["same"]["transcript"] = "\"'Thanks so much, mate.'"
    results["same"]["channel"] = "phone"
    twins, summ, _ = speech.summarize(drawn, results)
    assert summ["quotation_marks_removed"] == 1 and summ["twins_kept"] == 3
    assert summ["wer_over_0.6_numerals_one_side"]["lines"] == 0
    drawn.append(open_line("num", "It's five three four seven one zero eight."))
    results["num"] = {"line_id": "num", "voice": "af_heart", "seconds": 2.0, "transcript": "It's 5347108."}
    _, summ, _ = speech.summarize(drawn, results)
    assert summ["dropped"]["wer_over_0.6"] == 2 and summ["wer_over_0.6_numerals_one_side"]["lines"] == 1
    assert summ["by_channel"]["phone"]["kept"] >= 1
