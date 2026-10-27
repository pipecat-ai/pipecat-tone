import pytest
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.policy import ExpressionPolicy

from pipecat_tone.tagger import BREATHE
from pipecat_tone.tagger.labels import (
    TTS_ADAPTERS,
    LabelSpace,
    Pause,
    SentenceLabels,
    Span,
    label_map,
    tts_key,
)


@pytest.mark.parametrize("tts", ["cartesia", "minimax"])
def test_fixed_list_names_are_declared(tts):
    """Every name mapped for a fixed-list TTS is one it declares, or the label could never resolve."""
    caps = TTS_ADAPTERS[tts]().capabilities()
    declared = {t.name for t in caps.delivery}
    sounds = {t.name for t in caps.sounds}
    labels = label_map()
    for label, by_tts in labels.deliveries.items():
        assert set(by_tts.get(tts, [])) <= declared, label
    for label, by_tts in labels.sounds.items():
        assert set(by_tts.get(tts, [])) <= sounds, label


def test_every_label_lists_every_tts():
    labels = label_map()
    for kind in (labels.deliveries, labels.sounds):
        for label, by_tts in kind.items():
            assert set(by_tts) == set(TTS_ADAPTERS), label


def test_minimax_masks_what_it_cant_perform():
    s = LabelSpace.for_tts("minimax")
    assert "sympathetic" not in s.deliveries
    assert s.deliveries["warm"] == "happy"
    assert s.sounds["chuckle"] == "chuckle"
    assert not s.emphasis


def test_policy_narrows_to_its_names():
    policy = ExpressionPolicy(delivery=["calm", "happy", "slow"], sounds=["laughs"])
    s = LabelSpace.for_tts("minimax", policy)
    assert set(s.deliveries) == {"calm", "warm", "excited", "slow"}
    assert set(s.sounds) == {"laugh"}


def test_free_text_takes_the_first_offered_candidate():
    # airline_rebooking offers Fish "empathetic" and "compassionate" but not "sympathetic".
    policy = ExpressionPolicy(delivery=["calm", "empathetic", "compassionate", "slow"])
    s = LabelSpace.for_tts("fish", policy)
    assert s.deliveries["sympathetic"] == "empathetic"
    assert "apologetic" not in s.deliveries


def test_free_text_falls_back_to_the_canonical_name():
    s = LabelSpace.for_tts("elevenlabs")
    assert s.deliveries["calm"] == "calm"
    assert s.deliveries["apologetic"] == "apologetic"


def test_clamp_pause():
    s = LabelSpace.for_tts("fish")
    assert s.clamp_pause(50) == 100
    assert s.clamp_pause(9000) == 1500


def test_tts_key():
    assert tts_key("CartesiaTTSService#0") == "cartesia"
    assert tts_key("minimax") == "minimax"
    assert tts_key("OpenAITTSService#0") is None


def test_unknown_tts_raises():
    with pytest.raises(ValueError):
        LabelSpace.for_tts("openai")


def test_dict_round_trip():
    s = LabelSpace.for_tts("fish", ExpressionPolicy(delivery=["calm", "empathetic", "slow"]))
    assert LabelSpace.from_dict(s.to_dict()) == s
    # Corpus v0 rows list the canonical labels only.
    v0 = LabelSpace.from_dict({"tts": "fish", "deliveries": ["calm"], "sounds": [], "emphasis": False})
    assert v0.deliveries == {"calm": "calm"} and v0.pause_range_ms is None


def test_sentence_labels_dict_round_trip():
    labels = SentenceLabels(
        delivery="calm",
        spans=[Span(4, 9, "slow")],
        emphasis=[(0, 3)],
        pauses=[Pause(3, 300)],
        sound_after="laugh",
    )
    assert SentenceLabels.from_dict(labels.to_dict()) == labels


def test_vocabulary_names_come_first():
    s = LabelSpace.for_tts("elevenlabs", vocabulary={"warm": ["cheerful"], "chuckle": ["giggles"]})
    assert s.deliveries["warm"] == "cheerful" and s.sounds["chuckle"] == "giggles"
    # A fixed list still performs only what it declares.
    assert LabelSpace.for_tts("minimax", vocabulary={"warm": ["cheerful"]}).deliveries["warm"] == "happy"


def test_a_tts_labels_yaml_doesnt_list_is_matched_on_what_it_declares():
    caps = ExpressionCapabilities(
        delivery=[ExpressionTerm("empathetic"), ExpressionTerm("calm")], sounds=[ExpressionTerm("laughter")]
    )
    s = LabelSpace.resolve("other", caps)
    assert s.deliveries == {"calm": "calm", "sympathetic": "empathetic"}
    # As labels.yaml does for Cartesia, whose one sound is laughter.
    assert s.sounds == {"laugh": "laughter", "chuckle": "laughter"}


def test_breath_is_offered_only_with_breathe():
    asking = ExpressionPolicy(guidance=[BREATHE])
    assert "breath" not in LabelSpace.for_tts("elevenlabs").sounds
    assert LabelSpace.for_tts("elevenlabs", asking).sounds["breath"] == "inhales deeply"
    assert LabelSpace.for_tts("minimax", asking).sounds["breath"] == "breath"
    assert "breath" not in LabelSpace.for_tts("cartesia", asking).sounds
    assert (
        "breath"
        not in LabelSpace.for_tts("minimax", ExpressionPolicy(guidance=[BREATHE], sounds=["laughs"])).sounds
    )


def test_breath_kinds_resolve_only_where_breath_is_offered():
    breathe = ExpressionPolicy(guidance=[BREATHE])
    minimax = LabelSpace.for_tts("minimax", breathe)
    assert minimax.breaths == {"light breath": "breath", "deep breath": "inhale", "exhale": "exhale"}
    assert "deep breath" not in minimax.sounds and minimax.sound_name("deep breath") == "inhale"
    assert LabelSpace.for_tts("elevenlabs", breathe).breaths["light breath"] == "quick breath"
    assert (
        LabelSpace.for_tts("minimax").breaths == {} and LabelSpace.for_tts("cartesia", breathe).breaths == {}
    )
    assert LabelSpace.from_dict(minimax.to_dict()) == minimax
