"""The reader's training lines: the class lists, the input format as the runtime renders it, and how the label sets and
the teachers' labels merge into one example per line. No torch, no volume."""

import numpy as np
import pytest

from lab.reader import data, schemas, sources
from lab.reader.data import previous_words, render


def test_class_lists_are_the_runtimes(rc1):
    assert list(schemas.EMOTIONS) == rc1["emotions"]
    assert list(schemas.MOVES) == list(rc1["turns"][0]["fp32"]["move"]["probabilities"])
    assert schemas.INTENSITIES == ("low", "medium", "high")


def test_render_and_names_cut():
    assert render("Yes.", None) == '"Yes."'
    assert (
        render("oh great", previous_words("Mara: Sorry, we're full."))
        == '"oh great" Before: "Sorry, we\'re full."'
    )
    assert previous_words("player: where's the key?") == "where's the key?"
    assert previous_words("no name here") == "no name here"
    assert previous_words("") is None and previous_words(None) is None


class _Tokenizer:
    def __init__(self):
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        return type("Encoding", (), {"ids": [1, 2], "attention_mask": [1, 1]})()


def test_the_runtime_reads_the_input_format():
    from pipecat_tone.reader.model import ToneReaderModel

    model = object.__new__(ToneReaderModel)
    model.config = {
        "outputs": ["emotion", "emotion_intensity", "backchannel", "move", "sarcasm"],
        "emotions": list(schemas.EMOTIONS),
        "intensities": list(schemas.INTENSITIES),
        "moves": list(schemas.MOVES),
    }
    model._tokenizer = _Tokenizer()
    sizes = (len(schemas.EMOTIONS), len(schemas.INTENSITIES), 2, len(schemas.MOVES), 2)
    model._session = type("Session", (), {"run": lambda self, _, inputs: [np.zeros((1, n)) for n in sizes]})()
    model._emotion_temperature = model._move_temperature = 1.0
    model._sarcasm_curve, model._sarcasm_threshold = None, 0.5
    for text, previous in (("  oh great ", " Sorry, we're full. "), ("Yes.", None), ("mm-hm", "  ")):
        model.tag(text, previous)
        assert model._tokenizer.seen[-1] == render(text, previous)


def line(lid, split="train", text="hello", previous="Mara: hi", twin_of=None, slices=(), source="mix"):
    return {
        "line_id": lid,
        "split": split,
        "twin_of": twin_of,
        "source": source,
        "slices": list(slices),
        "input": {
            "speaker": "player",
            "speaker_kind": "player",
            "to": "Mara",
            "targets": ["Mara"],
            "text": text,
            "previous": previous,
        },
    }


def fake_jsonl(store: dict):
    return lambda key: store.get(key, [])


def build_store():
    store = {}
    a, b = sources.ACT_EMOTION_SETS[0], sources.ACT_EMOTION_SETS[1]
    store[sources.labels_key(a)] = [
        {
            "line": line("c1"),
            "emotion": {"label": "angry", "intensity": "high"},
            "backchannel": {"label": "false"},
        },
        {
            "line": line("c2", "dev"),
            "emotion": {"label": "neutral", "intensity": "low"},
            "backchannel": {"label": "true"},
        },
        {
            "line": line("c3", "test", slices=("player_speech",)),
            "emotion": {"label": "happy", "intensity": "low"},
            "backchannel": {"label": "false"},
        },
        {
            "line": line("c4", "test"),
            "emotion": {"label": None, "intensity": None},
            "backchannel": {"label": None},
        },
    ]
    # the same line in a second set: the first set's labels stand
    store[sources.labels_key(b)] = [
        {"line": line("c1"), "emotion": {"label": "sad", "intensity": "low"}, "backchannel": {}}
    ]
    store[sources.labels_key(sources.EXCHANGE_SET)] = [
        {"line": line("x1"), "exchange": {"label": "asks", "agree": 2}},
        {"line": line("x2", "test"), "exchange": {"label": "answers", "agree": 3}},
    ]
    store[sources.sarcasm_key(sources.SARCASM_SETS["train"][0])] = [
        {"line": line("c1"), "sarcasm": {"label": True, "agree": 3}}
    ]
    store[sources.sarcasm_key(sources.NATURAL_SARCASM[1])] = [
        {"line": line("c3", "test", slices=("player_speech",)), "sarcasm": {"label": False, "agree": 2}}
    ]
    store[sources.sarcasm_key(sources.HARD_SARCASM_TEST[0])] = [
        {"line": line("s9", "test"), "sarcasm": {"label": True, "agree": 3}}
    ]
    soft = {"emo": [1 / 9] * 9, "emo_int": [1 / 3] * 3, "back": 0.1}
    store[sources.soft_key()] = [{"line_id": "p1", **soft}, {"line_id": "p2", **soft}]
    store[sources.pool_key()] = [line("p1"), line("p1t", twin_of="p1"), line("p2"), line("p3")]
    store[sources.exchange_soft_key("lines")] = [line("p1"), line("g1")]
    for k, t in enumerate(sources.EXCHANGE_TEACHERS):
        store[sources.exchange_soft_key(t)] = [
            {"line_id": "p1", "p": [1.0 if i == k else 0.0 for i in range(7)]},
            {"line_id": "g1", "p": [1 / 7] * 7},
        ]
    store[sources.sarcasm_soft_key("t0")] = [
        {"line_id": "p1", "p": 0.2},
        {"line_id": "c1", "p": 0.9},
        {"line_id": "x1", "p": 0.4},
    ]
    store[sources.sarcasm_soft_key("t1")] = [
        {"line_id": "p1", "p": 0.4},
        {"line_id": "c1", "p": 0.7},
        {"line_id": "x1", "p": 0.6},
    ]
    return store


def test_merge():
    lines = data.build(fake_jsonl(build_store()), sarcasm_teachers=("t0", "t1"))
    c1 = lines.by_id["c1"]
    assert (
        c1.emo == data.EMOTION_INDEX["angry"] and c1.emo_int == data.INTENSITY_INDEX["high"] and c1.back == 0
    )
    assert c1.sarc == 1 and c1.sarc_w == 1.0 and c1.s_sarc is None  # a hard label wins over the teachers
    assert c1.previous == "hi" and c1.text == "hello"
    assert lines.by_id["c2"].emo_int == -100 and lines.by_id["c2"].back == 1  # neutral: no intensity
    assert lines.by_id["x1"].move == data.MOVE_INDEX["asks"] and lines.by_id["x1"].move_w == pytest.approx(
        0.6667
    )
    assert lines.by_id["x1"].s_sarc == pytest.approx(0.5)
    p1, p1t = lines.by_id["p1"], lines.by_id["p1t"]
    assert p1.s_emo and p1t.s_emo == p1.s_emo  # the twin takes its clean line's labels
    assert p1.s_move == pytest.approx([1 / 3] * 3 + [0.0] * 4)
    assert p1t.s_move == p1.s_move and p1t.s_sarc == pytest.approx(0.3)
    assert "p3" not in lines.by_id  # no soft labels, no example
    assert lines.by_id["g1"].split == "train" and lines.by_id["g1"].s_move is not None
    s = lines.summary()
    assert s["split_clash"] == 0 and s["train_with"]["move"] == 4  # x1, p1, p1t, g1


def test_groups():
    lines = data.build(fake_jsonl(build_store()), sarcasm_teachers=("t0", "t1"))
    test = lines.split("test")
    g = data.eval_groups(test, "test")
    ids = {k: {test[i].line_id for i in v} for k, v in g.items()}
    # c4 has no emotion majority
    assert ids["emotion_main"] == set() and ids["emotion_player_speech"] == {"c3"}
    assert ids["move"] == {"x2"} and ids["sarcasm_hard"] == {"s9"} and ids["sarcasm_natural"] == {"c3"}
    dev = lines.split("dev")
    assert {dev[i].line_id for i in data.eval_groups(dev, "dev")["emotion"]} == {"c2"}


def test_split_clash_kept_out():
    store = build_store()
    store[sources.labels_key(sources.EXCHANGE_SET)].append(
        {"line": line("c2", "train"), "exchange": {"label": "closes", "agree": 3}}
    )
    lines = data.build(fake_jsonl(store))
    assert "c2" in lines.split_clash and all(
        e.line_id != "c2" for e in lines.split("dev") + lines.split("train")
    )


def test_r1_training_lines_each_clean_line_once():
    store = build_store()
    walk = list(data.r1_training_lines(fake_jsonl(store)))
    assert [(where, ln["line_id"]) for where, ln in walk] == [
        ("pool", "p1"),
        ("pool", "p1t"),
        ("pool", "p2"),
        ("pool", "p3"),
        ("exchange_pool", "p1"),
        ("exchange_pool", "g1"),
        ("act_emotion_train", "c1"),
        ("act_emotion_train", "c1"),
        ("exchange_train", "x1"),
        ("sarcasm_train", "c1"),
    ]
    assert list(data.r1_training_lines(fake_jsonl(store), sarcasm=False)) == walk[:-1]
    once = [(where, ln["line_id"]) for where, ln in data.clean_once(walk)]
    assert once == [
        ("pool", "p1"),
        ("pool", "p2"),
        ("pool", "p3"),
        ("exchange_pool", "g1"),
        ("act_emotion_train", "c1"),
        ("exchange_train", "x1"),
    ]
    # a line left out by its source does not count as seen
    again = [("a", line("y", source="goemotions")), ("b", line("y"))]
    assert [where for where, _ in data.clean_once(again, ("goemotions",))] == ["b"]


def test_mean_teachers_sums_in_file_order():
    files = [
        [{"line_id": "a", "p": 0.1, "v": [0.1, 1.0]}, {"line_id": "b", "p": 0.5, "v": [0.0, 0.0]}],
        [{"line_id": "a", "p": 0.2, "v": [0.2, 1.0]}],
        [{"line_id": "a", "p": 0.3, "v": [0.3, 1.0]}, {"line_id": "b", "p": 0.5, "v": [0.0, 0.0]}],
    ]
    means = data.mean_teachers(files, ("p", "v"))
    assert list(means) == ["a"]  # the lines every file has
    assert means["a"]["p"] == ((0.1 + 0.2) + 0.3) / 3 != (0.1 + (0.2 + 0.3)) / 3
    assert means["a"]["v"] == [((0.1 + 0.2) + 0.3) / 3, 1.0]
    assert data.mean_teachers([], ("p",)) == {}


# --- r2 ------------------------------------------------------------------------------------------------------------------


def r2_store():
    store = build_store()
    for k in list(store):  # r1's soft files are not read by r2's main path
        if "/soft/" in k:
            del store[k]
    store[sources.tonetext_labels_key(sources.R2_SETS["committee"])] = [
        {
            "line": line("v1", "dev", source="natcs"),
            "emotion": {"label": "angry", "intensity": "low"},
            "backchannel": {"label": "false"},
        },
        {
            "line": line("v2", "test", source="abcd"),
            "emotion": {"label": "neutral", "intensity": "low"},
            "backchannel": {"label": "false"},
        },
        {
            "line": line("v3", source="natcs"),
            "emotion": {"label": "happy", "intensity": "medium"},
            "backchannel": {"label": "false"},
        },
        # a Tone line relabelled by r2: r2's label stands
        {
            "line": line("c1"),
            "emotion": {"label": "surprised", "intensity": "high"},
            "backchannel": {"label": "false"},
        },
    ]
    store[sources.tonetext_labels_key(sources.R2_SETS["exchange"])] = [
        {"line": line("v1", "dev", source="natcs"), "exchange": {"label": "asks", "agree": 3}}
    ]
    store[sources.tonetext_labels_key(sources.R2_SETS["sarcasm"])] = [
        {"line": line("v2", "test", source="abcd"), "sarcasm": {"label": False, "agree": 3}}
    ]
    store[sources.open_key("natcs")] = [line(f"n{i}", "train", source="natcs") for i in range(5)] + [
        line("nd", "dev", source="natcs")
    ]
    store[sources.speech_key()] = [line("n0s", twin_of="n0", source="natcs")]
    store[sources.pool_key()] = store[sources.pool_key()] + [line("g1", source="goemotions")]
    soft = {"emo": [1 / 9] * 9, "emo_int": [1 / 3] * 3, "back": 0.2, "move": [1 / 7] * 7, "sarc": 0.1}
    ids = ["p1", "p2", "p3", "g1", "x1", "c1", "v3"] + [f"n{i}" for i in range(5)]
    store[sources.r2_soft_key("t0")] = [{"line_id": i, **soft} for i in ids]
    store[sources.r2_soft_key("t1")] = [{"line_id": i, **soft, "sarc": 0.3} for i in ids]
    return store


def test_r2_merge():
    lines = data.build_r2(fake_jsonl(r2_store()), teachers=("t0", "t1"), speech=True)
    c1 = lines.by_id["c1"]
    assert c1.emo == data.EMOTION_INDEX["surprised"] and data.R2_TAG in c1.sets  # r2's label wins over Tone's
    # hard labels kept, the missing heads filled
    assert c1.s_emo is None and c1.s_move is not None and c1.sarc == 1
    v3 = lines.by_id["v3"]
    assert v3.emo == data.EMOTION_INDEX["happy"] and v3.s_move is not None and v3.s_sarc == pytest.approx(0.2)
    assert "nd" not in lines.by_id  # an open dev conversation never reaches the pool
    n0s = lines.by_id["n0s"]
    assert n0s.twin_of == "n0" and n0s.s_emo is not None  # the speech twin takes its clean line's labels
    assert lines.by_id["p1"].s_move == pytest.approx([1 / 7] * 7)
    assert lines.by_id["v1"].split == "dev" and lines.by_id["v1"].move == data.MOVE_INDEX["asks"]
    assert "p3" in lines.by_id and "g1" in lines.by_id


def test_r2_drop_goemotions_and_groups():
    lines = data.build_r2(fake_jsonl(r2_store()), teachers=("t0", "t1"), exclude_sources=("goemotions",))
    assert "g1" not in lines.by_id
    dev = lines.split("dev")
    g = data.eval_groups_r2(dev, "dev")
    ids = {k: {dev[i].line_id for i in v} for k, v in g.items()}
    assert ids["voice_emotion"] == {"v1"} and ids["voice_move"] == {"v1"} and ids["emotion"] == {"c2"}
    test = lines.split("test")
    t = data.eval_groups_r2(test, "test")
    assert {test[i].line_id for i in t["voice_sarcasm"]} == {"v2"}


def test_r2_draw_is_fixed(monkeypatch):
    rows = [{"line_id": f"a{i}"} for i in range(50)]
    assert data.sha_order(rows, "s") == data.sha_order(list(reversed(rows)), "s")
    train = [line(f"n{i}", source="natcs") for i in range(12)]
    store = {sources.open_key("natcs"): train + [line("nd", "dev", source="natcs")]}
    monkeypatch.setitem(sources.OPEN_CAPS, "natcs", 5)
    assert data.r2_open_lines(fake_jsonl(store)) == data.sha_order(train, "r2-natcs")[:5]
    # at or under its cap, a source's lines as they come
    monkeypatch.setitem(sources.OPEN_CAPS, "natcs", 12)
    assert data.r2_open_lines(fake_jsonl(store)) == train


def test_r2_soft_id():
    assert data.r2_soft_id() == sources.R2_SOFT_ID and data.r2_soft_id("nogo") == "tonetext-r2-soft-nogo-0.1"
    store = r2_store()
    for t in ("t0", "t1"):
        store[sources.r2_soft_key(t, data.r2_soft_id("nogo"))] = store.pop(sources.r2_soft_key(t))
    lines = data.build_r2(fake_jsonl(store), teachers=("t0", "t1"), soft_id=data.r2_soft_id("nogo"))
    assert lines.by_id["v3"].s_sarc == pytest.approx(0.2)


def test_r2_lines_to_label():
    out = data.r2_lines_to_label(fake_jsonl(r2_store()), exclude_sources=("goemotions",))
    ids = {ln["line_id"] for ln in out}
    assert (
        {"p1", "p2", "p3", "v3", "c1", "x1"} <= ids
        and "g1" not in ids
        and "p1t" not in ids
        and "v1" not in ids
    )
    assert {f"n{i}" for i in range(5)} <= ids and "nd" not in ids
