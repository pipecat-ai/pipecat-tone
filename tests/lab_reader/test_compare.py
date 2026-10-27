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
