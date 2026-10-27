import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pipecat.expressive.perception import PerceptionSignal, format_perception
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy

from lab import TONE_BENCH
from lab.corpus.breaths import (
    BREATH_VERSION,
    REPLY_RATES,
    breathing_replies,
    needs_breaths,
    variant_policy,
    with_breaths,
)
from lab.corpus.export import call_rows
from lab.corpus.rows import Row
from lab.corpus.scenario import _to_scenario, load
from lab.corpus.specs import DEV_DOMAINS, TEST_DOMAINS, V2_START, make_spec
from lab.corpus.speech_breaths import Turn, meeting_turns, sentence_rate, summary
from lab.corpus.teacher import Exchange, _find, _parse
from lab.train.calibrate import breathes, capped, fit
from lab.train.data import make_example
from lab.train.evaluate import breaths, predictor, score
from pipecat_tone.tagger import BREATHE, LabelSpace, ModelTagger, SentenceLabels, Span
from pipecat_tone.tagger.features import Layout
from pipecat_tone.tagger.model import sound_classes

# Loading a scenario needs tone-bench's checkout, which isn't public; without it, these tests are skipped.
needs_tone_bench = pytest.mark.skipif(
    not (TONE_BENCH / "bench").is_dir(), reason="tone-bench isn't checked out"
)


def row() -> Row:
    heard = [PerceptionSignal("tone", labels=[("sad", 0.8)])]
    return Row(
        call="c00001-abcdef",
        turn=1,
        n=1,
        split="train",
        domain="parcel courier",
        agent_model="phonellm-mini",
        space=LabelSpace.for_tts("cartesia"),
        guidance=[ExpressionGuidance.MATCH_STYLE],
        history=[
            {"role": "user", "text": "Hi.", "heard": None},
            {"role": "assistant", "text": "Hello."},
            {"role": "user", "text": "My parcel never came.", "heard": format_perception(heard)},
        ],
        earlier=["I'm so sorry."],
        sentence="Your ticket is 4TR8KW.",
        labels=SentenceLabels(spans=[Span(15, 21, "slow")]),
        trainable=True,
    )


def test_row_round_trip():
    r = row()
    assert Row.from_dict(r.to_dict()) == r


def test_row_turn_context_reads_the_call_as_the_processor_does():
    turn = row().turn_context()
    assert [m.role for m in turn.messages] == ["user", "assistant", "user"]
    assert turn.caller.text == "My parcel never came."
    assert turn.heard[0].labels == [("sad", 0.8)]
    assert turn.policy.guidance == [ExpressionGuidance.MATCH_STYLE]


def test_specs_are_seeded_and_split_by_domain():
    assert make_spec(7) == make_spec(7)
    for seed in range(50):
        spec = make_spec(seed)
        assert spec.split == (
            "dev" if spec.domain in DEV_DOMAINS else "test" if spec.domain in TEST_DOMAINS else "train"
        )


def test_teacher_quotes_become_offsets():
    sentence = "Your code is 4 8 2 6 3, okay?"
    assert _find(sentence, " 4 8 2 6 3, ") == (13, 22)
    assert _find(sentence, "YOUR CODE") == (0, 9)
    assert _find(sentence, "nowhere") is None
    data = {
        "replies": [
            {
                "turn": 1,
                "sentences": [
                    {
                        "sentence": 2,
                        "delivery": "calm",
                        "slow": ["4 8 2 6 3"],
                        "emphasis": [],
                        "pause_after": [{"word": "code", "ms": 300}],
                    },
                    {"sentence": 9, "delivery": "warm"},  # no such sentence
                ],
            },
            {"turn": 5, "sentences": []},  # no such turn
        ]
    }
    labels = _parse(data, [Exchange("Hi.", None, ["Sure.", sentence])])
    assert labels[0][0] == SentenceLabels()
    assert labels[0][1].delivery == "calm"
    assert labels[0][1].spans == [Span(13, 22, "slow")]
    assert labels[0][1].pauses[0].at == 9


# Every v0 and v1 spec, as their calls were built.
V0_V1_SPECS = "c7837114a19985b6"


def test_v0_and_v1_specs_are_what_their_calls_were_built_from():
    specs = [
        {k: v for k, v in make_spec(seed).to_dict().items() if k != "breathe"} for seed in range(V2_START)
    ]
    assert hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()[:16] == V0_V1_SPECS


@needs_tone_bench
def test_v2_asks_half_its_calls_to_breathe_and_writes_it_into_the_policy():
    specs = [make_spec(seed) for seed in range(V2_START, V2_START + 200)]
    assert 70 < sum(s.breathe for s in specs) < 130
    spec = next(s for s in specs if s.breathe)
    assert BREATHE in load(_to_scenario(spec, scenario_data())).policy.guidance


def scenario_data() -> dict:
    turn = {"tone": [{"label": "neutral", "confidence": 0.9}], "sarcasm": 0.0}
    return {
        "title": "A delay",
        "agent_prompt": "You help with deliveries.",
        "details": [],
        "turns": [{"caller": "Where is my sofa?", **turn}, {"caller": "That's too late.", **turn}],
    }


def test_v3_plants_a_moment_for_a_rare_delivery_in_every_call():
    from lab.corpus.specs import RARE_MOMENTS, V3_START

    assert all(make_spec(seed).events[0] in RARE_MOMENTS for seed in range(V3_START, V3_START + 50))
    assert not any(
        e in RARE_MOMENTS for seed in range(V2_START, V2_START + 50) for e in make_spec(seed).events
    )


def test_a_breath_goes_only_where_no_sound_starts_the_sentence():
    labels = [SentenceLabels(sound_before="chuckle"), SentenceLabels()]
    assert [lab.sound_before for lab in with_breaths(labels, [0, 1])] == ["chuckle", "breath"]


@needs_tone_bench
def test_export_writes_a_breath_variant_beside_the_call():
    spec = next(s for s in map(make_spec, range(200)) if s.tts == "elevenlabs" and s.sounds is None)
    scenario = _to_scenario(spec, scenario_data())
    policy = variant_policy(load(scenario).policy, "elevenlabs", spec.id)
    plain = {"same_words": True, "format_issues": [], "verdict": "pass", "note": ""}
    sentences = [["It's on the way.", "It arrives Friday."], ["I'm sorry."]]
    call = {
        "spec": spec.to_dict(),
        "scenario": scenario,
        "turns": [
            {
                "user": t["caller"],
                "heard": None,
                "sentences": s,
                "labels": [SentenceLabels().to_dict()] * len(s),
                **plain,
            }
            for t, s in zip(scenario_data()["turns"], sentences)
        ],
        "breaths": {
            "version": BREATH_VERSION,
            "policy": {"guidance": policy.guidance, "delivery": policy.delivery, "sounds": policy.sounds},
            "offered": True,
            "turns": [
                {"breaths": [1], **plain, "markup": ""},
                {"breaths": [], **plain, "verdict": "fail", "markup": ""},
            ],
        },
    }
    rows = list(call_rows(call))
    own, variant = rows[:3], rows[3:]
    assert all(BREATHE not in r.guidance and "breath" not in r.space.sounds for r in own)
    assert all(BREATHE in r.guidance and r.space.sounds["breath"] == "inhales deeply" for r in variant)
    assert [r.labels.sound_before for r in variant] == [None, "breath", None]
    assert [r.trainable for r in variant] == [True, True, False]
    assert {r.call for r in variant} == {f"{spec.id}-breathe"}
    assert variant[2].turn_context().previous == [SentenceLabels(), SentenceLabels(sound_before="breath")]
    assert own[2].turn_context().previous == [SentenceLabels(), SentenceLabels()]
    assert own[0].turn_context().previous is None


@needs_tone_bench
def test_a_call_that_asks_to_breathe_takes_the_passes_breaths_in_its_own_rows():
    spec = next(
        s for s in map(make_spec, range(V2_START, V2_START + 200)) if s.breathe and s.tts == "elevenlabs"
    )
    scenario = _to_scenario(spec, scenario_data())
    policy = load(scenario).policy
    plain = {"same_words": True, "format_issues": [], "verdict": "pass", "note": ""}
    teachers = [
        [SentenceLabels(sound_before="breath").to_dict(), SentenceLabels().to_dict()],
        [SentenceLabels().to_dict()],
    ]
    call = {
        "spec": spec.to_dict(),
        "scenario": scenario,
        "turns": [
            {
                "user": t["caller"],
                "heard": None,
                "sentences": ["It's on the way.", "It arrives Friday."][: len(lab)],
                "labels": lab,
                **plain,
            }
            for t, lab in zip(scenario_data()["turns"], teachers)
        ],
    }
    assert needs_breaths(call)
    call["breaths"] = {
        "version": BREATH_VERSION,
        "policy": {"guidance": policy.guidance, "delivery": policy.delivery, "sounds": policy.sounds},
        "offered": True,
        "turns": [
            {"breaths": [1], **plain, "markup": ""},
            {"breaths": [], **plain, "verdict": "fail", "markup": ""},
        ],
    }
    assert not needs_breaths(call)
    rows = list(call_rows(call))
    assert {r.call for r in rows} == {spec.id} and all(BREATHE in r.guidance for r in rows)
    assert [r.labels.sound_before for r in rows] == [None, "breath", None]
    assert [r.trainable for r in rows] == [True, True, False]


MEETING = """<Meeting><Transcript>
<Segment StartTime="0.0" EndTime="2.0" Participant="a">Hello <VocalSound Description="breath"/> there. How are you?</Segment>
<Segment StartTime="2.1" EndTime="2.5" Participant="b"><VocalSound Description="inbreath"/></Segment>
<Segment StartTime="2.5" EndTime="4.0" Participant="b">I'm fine, <VocalSound Description="breath"/> thanks for asking.</Segment>
<Segment StartTime="4.0" EndTime="4.3" Participant="a">Yeah.</Segment>
<Segment StartTime="4.3" EndTime="6.0" Participant="b">So the <Emphasis>report</Emphasis> is done. <VocalSound Description="inbreath"/> We sent it.</Segment>
<Segment StartTime="6.0" EndTime="7.0" Participant="a" DigitTask="true">One two three.</Segment>
<Segment StartTime="7.0" EndTime="8.0" Participant="c" CloseMic="false">Can you hear me?</Segment>
</Transcript></Meeting>"""


def test_icsi_turns_keep_the_floor_through_backchannels_and_take_breaths_from_just_before():
    assert meeting_turns("m", ET.fromstring(MEETING)) == [
        Turn("m/a", ((2, False), (3, False)), inside=(1,)),
        Turn("m/b", ((5, True), (5, False), (3, True)), inside=(2,)),
    ]


def test_speech_breath_statistics_and_a_sentences_rate():
    stats = summary(meeting_turns("m", ET.fromstring(MEETING)))
    assert (stats["turns"], stats["sentences"], stats["words"]) == (2, 5, 18)
    assert (stats["turns_with_breath"], stats["turns_with_two_or_more"]) == (1.0, 0.5)
    assert stats["breaths_inside_sentences"] == 0.5
    assert stats["words_between_breaths"]["p50"] == 5.0
    first = stats["sentences_by_length"]["first"]["1-5"]
    assert (first["breath_before"], first["breath_before_or_inside"]) == (0.5, 1.0)
    rate = sentence_rate(stats)
    assert (rate(2, True), rate(3, False)) == (1.0, 0.3333)


def test_calibration_counts_a_replys_breaths_up_to_its_cap():
    chosen = np.array([True, True, False, True, True, True])
    starts = np.array([True, False, False, True, False, False])
    assert capped(chosen, starts, 1).tolist() == [True, False, False, True, False, False]
    assert capped(chosen, starts, 2).tolist() == [True, True, False, True, True, False]
    assert capped(np.array([False, True, True]), np.array([True, False, False]), 1).tolist() == [
        False,
        True,
        False,
    ]
    assert capped(chosen, starts, None) is chosen


ICSI_SPELLING = """<Meeting><Transcript>
<Segment StartTime="0.0" EndTime="1.0" Participant="a">So the <Emphasis>report</Emphasis>. <VocalSound Description="inbreath"/> We sent it -</Segment>
<Segment StartTime="1.0" EndTime="1.2" Participant="b">O_K.</Segment>
<Segment StartTime="1.2" EndTime="2.0" Participant="a">and then <VocalSound Description="breath"/></Segment>
<Segment StartTime="2.0" EndTime="2.2" Participant="b"><VocalSound Description="inbreath"/></Segment>
<Segment StartTime="2.5" EndTime="3.0" Participant="b">Right, thanks.</Segment>
<Segment StartTime="3.1" EndTime="3.5" Participant="a">Sure thing.</Segment>
<Segment StartTime="3.6" EndTime="4.0" Participant="b">Good to hear.</Segment>
</Transcript></Meeting>"""


def test_icsi_turns_follow_its_spelling_and_place_each_breath_by_the_word_after_it():
    # "O_K." leaves a the floor; a split-off stop and a lone dash end sentences; a breath with no word after it is
    # dropped; and b's breath opens one turn, not two.
    assert meeting_turns("m", ET.fromstring(ICSI_SPELLING)) == [
        Turn("m/a", ((3, False), (3, True), (2, False))),
        Turn("m/b", ((2, True),)),
        Turn("m/a", ((2, False),)),
        Turn("m/b", ((3, False),)),
    ]


def test_breathing_turns_are_set_against_each_speakers_turns_shuffled():
    turns = [
        Turn(who, ((5, b),)) for who, b in (("s", True), ("s", True), ("s", False), ("t", False), ("t", True))
    ]
    stats = summary(turns)
    assert (stats["breath_after_breathing_turn"], stats["breath_after_plain_turn"]) == (0.5, 1.0)
    assert stats["breath_after_breathing_turn_by_chance"] == 0.3636


def test_each_band_gets_the_smallest_bias_that_meets_its_aim():
    margins = np.array([-1.0, -2.0, -3.0] * 2)
    bands, starts = np.array([0, 0, 0, 1, 1, 1]), np.ones(6, dtype=bool)
    assert fit(margins, bands, [1, 2], starts, None).tolist() == [1.01, 2.01]
    two_replies = np.array([True, False, False, True, False, False])
    assert breathes(margins, np.zeros(6, dtype=int), np.array([3.05]), two_replies, 1).sum() == 2


def test_breath_is_trained_in_its_own_head_and_never_as_a_sound():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    breathing = make_example(breathe_row("c1", 0, 0, True), tokenizer, Layout())
    sounds = sound_classes()
    assert (breathing.breath, breathing.breath_offered) == (1, 1)
    assert sounds[breathing.sound_before] == "none" and not breathing.sound_mask[sounds.index("breath")]
    assert make_example(row(), tokenizer, Layout()).breath_offered == 0


def breathe_row(call: str, turn: int, n: int, breath: bool) -> Row:
    policy = ExpressionPolicy(guidance=[BREATHE])
    return replace(
        row(),
        call=call,
        turn=turn,
        n=n,
        space=LabelSpace.for_tts("minimax", policy),
        guidance=policy.guidance,
        labels=SentenceLabels(sound_before="breath" if breath else None),
    )


class Breather:
    """A model that breathes before every sentence, capped at one a reply."""

    config = {"breaths_per_reply": 1}

    def predict(self, turn, earlier, sentence):
        return SentenceLabels(sound_before="breath")


def test_scoring_starts_a_turn_per_reply_so_a_cap_holds():
    rows = [breathe_row("c1", 0, 0, True), breathe_row("c1", 0, 1, False), breathe_row("c2", 0, 0, False)]
    predict = predictor(ModelTagger(Breather()))
    assert [predict(r).sound_before for r in rows] == ["breath", None, "breath"]
    assert score(rows, predictor(ModelTagger(Breather())))["breaths_predicted"] == 2


def test_sound_f1_leaves_drawn_breaths_to_their_own_score():
    gold = replace(
        breathe_row("c1", 0, 0, True), labels=SentenceLabels(sound_before="breath", sound_after="laugh")
    )
    found = score([gold], lambda r: SentenceLabels(sound_after="laugh"))
    assert (found["sound_f1"], found["breath_f1"]) == (1.0, 0.0)


def test_breaths_run_as_calls_and_score_against_the_teacher():
    rows = [breathe_row("c1", 0, 0, True), breathe_row("c1", 0, 1, False), breathe_row("c1", 1, 0, False)]
    found = breaths(rows, ModelTagger(Breather()))
    assert (found["breath_f1"], found["breaths_predicted"], found["breaths_gold"]) == (0.6667, 2, 1)
    assert (found["tagger"]["turns_with_breath"], found["teacher"]["turns_with_breath"]) == (1.0, 0.5)


def test_replies_breathe_at_peoples_rate_for_their_length():
    icsi = json.loads((Path(__file__).parents[1] / "results" / "breaths" / "icsi.json").read_text())
    assert all(
        abs(r - v["with_breath"]) < 0.001 for r, v in zip(REPLY_RATES, icsi["by_turn_length"].values())
    )
    short, long = {"sentences": ["Sure."]}, {"sentences": [" ".join(["word"] * 70) + "."]}
    calls = [breathing_replies(f"c{i}", [short, long]) for i in range(2000)]
    assert breathing_replies("c7", [short, long]) == calls[7]
    assert 0.03 < sum(0 in c for c in calls) / 2000 < 0.06 and 0.37 < sum(1 in c for c in calls) / 2000 < 0.45


def test_breath_kinds_come_from_the_breaths_people_take_to_speak(tmp_path, monkeypatch):
    import zipfile

    from lab.corpus import speech_breaths

    meeting = """<Meeting><Transcript>
<Segment StartTime="0.0" EndTime="0.8" Participant="a"><VocalSound Description="inbreath"/></Segment>
<Segment StartTime="0.9" EndTime="2.0" Participant="a">A long answer. <VocalSound Description="outbreath"/></Segment>
<Segment StartTime="2.0" EndTime="2.3" Participant="b"><VocalSound Description="breath"/></Segment>
<Segment StartTime="2.3" EndTime="3.0" Participant="b">Short.</Segment>
<Segment StartTime="3.0" EndTime="4.0" Participant="a"><VocalSound Description="breath"/></Segment>
</Transcript></Meeting>"""
    path = tmp_path / "icsi.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("transcripts/Btest.mrt", meeting)
    monkeypatch.setattr(speech_breaths, "fetch", lambda: zipfile.ZipFile(path))
    # Two breaths taken to speak, one deep; a's last breath leads to no words. One exhale in four marks.
    assert speech_breaths.breath_kinds() == {"light breath": 0.375, "deep breath": 0.375, "exhale": 0.25}


def test_a_soup_averages_its_models_weights_and_keeps_none_of_their_scores(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from lab.train.soup import soup

    folders = []
    for n, w in enumerate(([[1.0, 0.0], [0.0, 1.0]], [[3.0, 2.0], [2.0, 3.0]])):
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.array(w, dtype=np.float32), "w")],
        )
        folder = tmp_path / f"s{n}"
        folder.mkdir()
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(folder / "model.onnx")
        )
        (folder / "tokenizer.json").write_text("{}")
        (folder / "config.json").write_text(
            json.dumps({"layout": {}, "epoch": n, "sound_bias": {"breath": n}})
        )
        folders.append(folder)
    soup(tmp_path / "soup", folders)
    averaged = numpy_helper.to_array(onnx.load(str(tmp_path / "soup" / "model.onnx")).graph.initializer[0])
    assert averaged.tolist() == [[2.0, 1.0], [1.0, 2.0]]
    config = json.loads((tmp_path / "soup" / "config.json").read_text())
    assert config == {"layout": {}, "onnx": "model.int8.onnx", "soup": ["s0", "s1"]}
    assert (tmp_path / "soup" / "model.int8.onnx").exists()


def test_balanced_training_weighs_rare_deliveries_up():
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from lab.train.fit import _delivery_weights

    examples = (
        [SimpleNamespace(delivery=1)] * 100
        + [SimpleNamespace(delivery=2)] * 25
        + [SimpleNamespace(delivery=0)] * 50
    )
    plain, balanced = _delivery_weights(examples, False), _delivery_weights(examples, True)
    assert plain[:3] == [0.3, 1.0, 1.0] and set(plain[3:]) == {1.0}
    assert [round(w, 2) for w in balanced[:3]] == [0.3, 0.79, 1.58] and set(balanced[3:]) == {1.0}


def test_released_data_leaves_out_failed_calls_and_the_reviewers_model(tmp_path, monkeypatch):
    from lab import release

    (tmp_path / "a.json").write_text(
        json.dumps({"reviewer": {"model": "m", "version": 3}, "turns": []}, indent=1)
    )
    (tmp_path / "b.json").write_text(json.dumps({"error": "timed out"}, indent=1))
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "train.jsonl").write_text("{}\n")
    uploaded = {}

    class Api:
        def upload_folder(self, folder_path, path_in_repo, **kwargs):
            uploaded.update({p.name: p.read_text() for p in Path(folder_path).iterdir()})
            assert path_in_repo == "data/tagger/v9"

    monkeypatch.setattr(release, "HfApi", Api)
    release.release_data("tagger/v9", tmp_path)
    assert sorted(uploaded) == ["a.json", "stats.json", "train.jsonl"]
    assert json.loads(uploaded["a.json"]) == {"reviewer": {"version": 3}, "turns": []}
    assert uploaded["train.jsonl"] == "{}\n"
