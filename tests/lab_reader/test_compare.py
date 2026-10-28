"""The scorecard's paired comparisons, as the compare and precision stages read them: `scoring.paired_f1` over every
class and over the classes kept at the floor, the per-class reading, the test-line rules, and one column of items."""

import random

import pytest

from lab.reader import schemas, scoring


def _lines(seed: int, n: int = 400):
    rng = random.Random(seed)
    gold = [rng.choice(("neutral",) * 6 + ("happy",) * 3 + ("sad",)) for _ in range(n)]
    ref = [g if rng.random() < 0.8 else rng.choice(schemas.EMOTIONS) for g in gold]
    var = [g if rng.random() < 0.85 else rng.choice(schemas.EMOTIONS) for g in gold]
    return gold, ref, var


def test_every_class_kept_is_the_plain_reading():
    gold, ref, var = _lines(0)
    a, b = [[(g, p)] for g, p in zip(gold, ref)], [[(g, p)] for g, p in zip(gold, var)]
    whole = scoring.paired_f1(a, b, schemas.EMOTIONS)
    kept = scoring.paired_f1(a, b, schemas.EMOTIONS, schemas.EMOTIONS)
    assert list(whole) == ["ref", "var", "diff", "ci", "lines"]
    assert (kept["ref"], kept["var"], kept["diff"], kept["ci"]) == (
        whole["ref"],
        whole["var"],
        whole["diff"],
        whole["ci"],
    )


def test_kept_classes_read_as_the_floor():
    gold, ref, var = _lines(1)
    a, b = [[(g, p)] for g, p in zip(gold, ref)], [[(g, p)] for g, p in zip(gold, var)]
    res = scoring.paired_f1(a, b, schemas.EMOTIONS, ("happy", "neutral"))
    assert "sad" not in res["classes"]
    # sad has about 40 gold lines, under a floor of 100
    assert res["ref"] == pytest.approx(
        scoring.macro_f1_floor(gold, ref, schemas.EMOTIONS, floor=100), abs=1e-3
    )
    assert res["var"] == pytest.approx(
        scoring.macro_f1_floor(gold, var, schemas.EMOTIONS, floor=100), abs=1e-3
    )
    assert res["ci"][0] <= res["diff"] <= res["ci"][1]


def test_by_class():
    trip = [
        ("angry", "angry", "neutral"),
        ("angry", "angry", "angry"),
        ("neutral", "neutral", "neutral"),
        ("neutral", "angry", "neutral"),
    ]
    out = scoring.by_class(trip, schemas.EMOTIONS)
    assert set(out) == {"angry", "neutral"}
    assert (
        out["angry"]["gold"] == 2
        and out["angry"]["ref"]["recall"] == 1.0
        and out["angry"]["var"]["recall"] == 0.5
    )
    assert out["angry"]["ref"]["said"] == 3 and out["angry"]["ref"]["f1"] == pytest.approx(
        2 * 2 / (2 + 3), abs=1e-4
    )
    assert out["neutral"]["var"]["f1"] == pytest.approx(2 * 2 / (2 + 3), abs=1e-4)


def test_each_side_has_its_interval():
    gold, ref, var = _lines(2)
    a, b = [[(g, p)] for g, p in zip(gold, ref)], [[(g, p)] for g, p in zip(gold, var)]
    res = scoring.paired_f1(a, b, schemas.EMOTIONS, ("happy", "neutral"))
    assert (
        res["ref_ci"][0] <= res["ref"] <= res["ref_ci"][1]
        and res["var_ci"][0] <= res["var"] <= res["var_ci"][1]
    )


def test_macro_f1_floor():
    gold = ["neutral"] * 40 + ["happy"] * 30 + ["sad"] * 2
    pred = ["neutral"] * 40 + ["happy"] * 30 + ["neutral"] * 2
    # per-class F1 rounds to 3 places
    assert scoring.macro_f1_floor(gold, pred, schemas.EMOTIONS) == pytest.approx(
        (2 * 40 / (2 * 40 + 2) + 1.0) / 2, abs=1e-3
    )
    assert scoring.macro_f1_floor(["sad"] * 5, ["sad"] * 5, schemas.EMOTIONS) == 0.0


def test_the_test_line_rules():
    def line(split="test", twin_of=None, *slices):
        return {"split": split, "twin_of": twin_of, "slices": list(slices)}

    assert scoring.main_test(line()) and scoring.move_test(line("test", None, "player_speech"))
    assert not scoring.main_test(line("test", None, "player_speech"))
    assert not scoring.move_test(line("test", None, "unseen_genre"))
    assert scoring.in_slice(line("test", None, "unseen_genre"), "unseen_genre")
    assert not any(
        rule(line("test", "x1")) for rule in (scoring.clean_test, scoring.main_test, scoring.move_test)
    )
    assert scoring.noisy_twin(line("test", "x1")) and not scoring.noisy_twin(line("dev", "x1"))


class Fixed:
    """A tagger that reads every turn the same way, counting its reads."""

    def __init__(self, emotion: str, move: str, sarcastic: bool):
        self.tags = {
            "emotion": {"label": emotion},
            "move": {"label": move},
            "sarcasm": {"sarcastic": sarcastic},
        }
        self.reads = 0

    def tag(self, text: str, previous: str | None) -> dict:
        self.reads += 1
        return self.tags


def test_a_column_reads_each_input_once_and_keeps_its_items_in_order():
    rows = [
        {
            "line": {"input": {"text": f"line {i}", "previous": "Agent: How can I help?"}},
            "emotion": {"label": "happy"},
            "exchange": {"label": "answers"},
            "sarcasm": {"label": 0},
        }
        for i in range(40)
    ]
    ref, var = Fixed("neutral", "withholds", False), Fixed("happy", "answers", True)
    c = scoring.Paired(ref, var).column(keep=True)
    c.emotion("emotion", rows, floor=True)
    c.move("move", rows)
    c.sarcasm("sarcasm", rows, f1=True, false_rate=True)
    assert list(c.items) == [
        "emotion",
        "emotion_30",
        "move_macro_f1",
        "move_answered_or_open",
        "move_asks_or_not",
        "sarcasm_f1",
        "sarcasm_sincere_called_sarcastic",
    ]
    assert list(c.classes) == ["emotion", "move_macro_f1"] and ref.reads == var.reads == 40
    assert c.worse() == [] and c.rises() == ["sarcasm_sincere_called_sarcastic"]
    assert c.better() == ["emotion", "emotion_30", "move_macro_f1", "move_answered_or_open"]


class _Model:
    """A reader that calls everything sarcastic, or nothing."""

    def __init__(self, sarcastic: bool):
        self.sarcastic = sarcastic

    def tag(self, text, previous=None):
        p = 0.9 if self.sarcastic else 0.1
        return {
            "emotion": {"probabilities": {"angry": 0.7, "happy": 0.2, "neutral": 0.1}},
            "sarcasm": {"probability": p, "sarcastic": self.sarcastic},
        }


def test_the_probes_are_read_by_both_and_kept_without_their_text():
    from lab.reader.pipeline.compare import probe_reads
    from lab.reader.support import PROBES

    out = probe_reads(_Model(False), _Model(True))
    want = sum(w == "sarcastic" for *_, w in PROBES)
    assert out["_right"] == {"ref": len(PROBES) - want, "var": want} and out["_cases"] == len(PROBES)
    assert out["great_late_after_status"]["var"] == {
        "emotion": [["angry", 0.7], ["happy", 0.2]],
        "sarcasm": 0.9,
        "sarcastic": True,
        "right": True,
    }
    assert not any(t in str(out) for _, t, _, _ in PROBES)


def test_the_table_shows_each_item_read_and_the_probes():
    from lab.reader.pipeline.compare import markdown, probe_reads

    item = {"ref": 0.5, "var": 0.6, "diff": 0.1, "ci": [0.05, 0.15], "lines": 300}
    res = {
        "columns": {
            "with_previous": {"items": {"sarcasm_support_f1": item}},
            "without_previous": {"items": {}},
        },
        "probes": probe_reads(_Model(False), _Model(True)),
    }
    table = markdown(res, "rc1", "cand")
    assert (
        "| Support-call test split | Sarcasm F1, committee labels | 0.5 | 0.6 | +0.1 (+0.05 to +0.15) | 300 |"
        in table
    )
    assert "Sarcasm F1, turn alone" not in table  # an item not read is left out
    assert "Cases read right: rc1 4, cand 6 of 10." in table


class _AfterBad:
    """A reader that calls a turn sarcastic after a line with "fee" in it and nowhere else."""

    def tag(self, text, previous=None):
        sarcastic = "fee" in (previous or "")
        return {
            "emotion": {"label": "angry" if sarcastic else "happy"},
            "sarcasm": {"probability": 0.9 if sarcastic else 0.1, "sarcastic": sarcastic},
        }


def _pair_rows(n: int, opposite_sarcastic: bool = True, prompt: str = "tc1-online_shop-0000") -> list[dict]:
    rows = []
    for k in range(n):
        for side, prev, label in (
            ("meant", "Agent: I've waived it.", 0),
            ("opposite", "Agent: There's a fee.", int(opposite_sarcastic)),
        ):
            rows.append(
                {
                    "line": {
                        "source_ref": f"{prompt}#{k}/{side}",
                        "input": {"text": f"Lovely {k}.", "previous": prev},
                    },
                    "sarcasm": {"label": label},
                }
            )
    return rows


def test_contrast_pairs_are_read_on_both_lines_and_alone():
    from lab.reader.pipeline.compare import contrast_items

    items = contrast_items(
        scoring.Paired(_Model(True), _AfterBad()),
        _pair_rows(30) + _pair_rows(2, False, "tc1-online_shop-0001"),
    )
    assert items["contrast_pairs_right"]["lines"] == 30  # pairs the committee reads alike are left out
    assert (items["contrast_pairs_right"]["ref"], items["contrast_pairs_right"]["var"]) == (0.0, 1.0)
    assert items["contrast_pairs_right_at_act"]["var"] == 1.0
    alone = items["contrast_alone_sincere_called_sarcastic"]
    assert (alone["ref"], alone["var"]) == (1.0, 0.0)
    assert scoring.counts_mistakes("contrast_alone_sincere_called_sarcastic")
    assert contrast_items(scoring.Paired(_Model(True), _AfterBad()), []) == {}


def test_the_praise_cases_are_read_by_group_without_their_text():
    from lab.reader.pipeline.compare import markdown, praise_reads
    from lab.reader.support import PRAISE_PROBES

    out = praise_reads(_Model(False), _Model(True))
    g = out["groups"]
    assert g["after_good"]["lines"] == 13 and g["after_good"]["should_read"] == "sincere"
    assert g["after_good"]["ref"] == {"right": 13, "right_at_act": 13, "mean_chance": 0.1}
    assert g["after_bad"]["var"] == {"right": 8, "right_at_act": 8, "mean_chance": 0.9}
    assert len(out["cases"]) == len(PRAISE_PROBES)
    assert not any(t in str(out) for _, t, _, _ in PRAISE_PROBES)
    res = {
        "columns": {"with_previous": {"items": {}}, "without_previous": {"items": {}}},
        "probes": {"_right": {"ref": 0, "var": 0}, "_cases": 0},
        "praise": out,
    }
    table = markdown(res, "rc1", "cand")
    assert "| after_good | sincere | 13 | 13 (13), 0.1 | 0 (0), 0.9 |" in table
