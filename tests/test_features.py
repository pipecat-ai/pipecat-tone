from dataclasses import replace

from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from tokenizers import Tokenizer, models, pre_tokenizers

from lab.train.data import IGNORE, _bio
from pipecat_tone.tagger import (
    BREATHE,
    LabelSpace,
    Message,
    ModelTagger,
    SentenceLabels,
    TurnContext,
    llm_policy,
)
from pipecat_tone.tagger.features import Layout, encode, state_hints
from pipecat_tone.tagger.markup import render_reply
from pipecat_tone.tagger.model import BEGIN, INSIDE, OUTSIDE, _encode_pair, decode, decode_ranges


def test_encode_puts_the_sentence_last_and_points_at_it():
    turn = TurnContext(
        messages=[
            Message("user", "My parcel never came.", [PerceptionSignal("tone", labels=[("sad", 0.8)])])
        ],
        space=LabelSpace.for_tts("fish"),
        policy=ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE, "Keep it brief."]),
    )
    first, second, start = encode(
        turn, ["I'm sorry."], "Your ticket is 4TR8KW.", Layout(custom_guidance=True)
    )
    assert first == "caller (tone sad clear): My parcel never came."
    assert second[start:] == "Your ticket is 4TR8KW."
    assert "policy match style; Keep it brief." in second
    assert "state new detail" in second
    assert "caller sounded tone sad clear" in second
    assert "reply so far: I'm sorry." in second


def test_bio_and_decode_round_trip():
    sentence = "Your ticket is TR-58213, okay?"
    # Tokens as a subword tokenizer might cut them, with a non-sentence token first.
    offsets = [
        None,
        (0, 4),
        (4, 11),
        (11, 14),
        (14, 17),
        (17, 18),
        (18, 20),
        (20, 23),
        (23, 24),
        (24, 29),
        (29, 30),
    ]
    classes = _bio(offsets, [(15, 23)])
    assert classes[0] == IGNORE
    assert classes[4] == BEGIN and classes[5:8] == [INSIDE] * 3 and classes[8] == OUTSIDE
    assert decode_ranges(classes, offsets, sentence) == [(15, 23)]


def test_decode_widens_a_cut_word():
    sentence = "Say 4TR8KW please"
    offsets = [(0, 3), (3, 6), (6, 10), (10, 17)]
    assert decode_ranges([OUTSIDE, BEGIN, OUTSIDE, OUTSIDE], offsets, sentence) == [(4, 10)]


def test_state_hints():
    messages = [
        Message("user", "Hi, my order is late."),
        Message("assistant", "I'm sorry about that. Your ticket is 4TR8KW."),
        Message("user", "Can you slow down? Four T R eight K W?"),
    ]
    assert state_hints(messages, "Yes, 4TR8KW is right.") == [
        "apologized before",
        "caller asked to slow down or repeat",
        "caller just read this detail back",
    ]
    assert state_hints(messages[:2], "Your ticket is 4TR8KW.") == ["apologized before", "detail given before"]
    assert state_hints(messages[:1], "Anything else?") == []


def words_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate("a b c d e f g".split())}, unk_token="a")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_over_budget_the_call_is_cut_first():
    def tokens(layout, first, second):
        return _encode_pair(words_tokenizer(), first, second, layout).tokens

    assert tokens(Layout(max_tokens=4, cut_call_first=True), "a b", "c d e f") == ["c", "d", "e", "f"]
    # As v0 and v1 were trained: the longer text loses its start.
    assert tokens(Layout(max_tokens=4), "a b", "c d e f") == ["a", "b", "e", "f"]
    # A second text over budget on its own is cut too.
    assert len(tokens(Layout(max_tokens=4, cut_call_first=True), "a", "b c d e f g")) == 4


def test_breathe_is_a_preset_the_model_reads_and_the_llm_never_sees():
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE, BREATHE, "Be brief."])
    turn = TurnContext([Message("user", "Hi.")], LabelSpace.for_tts("fish", policy), policy)
    _, second, _ = encode(turn, [], "Hello.", Layout())
    assert "sounds laugh chuckle sigh breath" in second and "policy match style, breathe" in second
    assert llm_policy(policy).guidance == ["Be brief."]


def test_a_models_sound_bias_shifts_its_breaths_by_sentence_length():
    space = LabelSpace.for_tts("minimax", ExpressionPolicy(guidance=[BREATHE]))
    classes = (["none"], ["none", "laugh", "chuckle", "sigh", "breath"])
    quiet, tokens = [1.0, -5.0, -5.0, -5.0, -5.0], [[0.0, 0.0, 9.0]]
    heads = [[0.0], quiet, quiet, tokens, tokens, [0.5]]
    by_length = {"breath": [[5, -1.0], [None, 0.0]]}
    assert decode(heads, [(0, 2)], "Hi.", space, classes).sound_before == "breath"
    assert decode(heads, [(0, 2)], "Hi.", space, classes, by_length).sound_before is None
    assert decode(heads, [(0, 2)], "Hi " * 6, space, classes, by_length).sound_before == "breath"


def test_a_breath_head_breathes_only_where_no_other_sound_goes():
    space = LabelSpace.for_tts("minimax", ExpressionPolicy(guidance=[BREATHE]))
    classes = (["none"], ["none", "laugh", "chuckle", "sigh", "breath"])
    quiet, after = [1.0, -5.0, -5.0, -5.0, 9.0], [5.0, -5.0, -5.0, -5.0, -5.0]
    tokens = [[0.0, 0.0, 9.0]]

    def before(sounds, breath, bias=None):
        heads = [[0.0], sounds, after, tokens, tokens, [breath]]
        return decode(heads, [(0, 2)], "Hi.", space, classes, bias).sound_before

    assert before(quiet, 0.5) == "breath"  # the breath class of the sound head never wins
    assert before(quiet, 0.5, {"breath": [[None, -1.0]]}) is None
    assert before([1.0, 3.0, -5.0, -5.0, -5.0], 9.0) == "laugh"
    assert before(quiet, -0.5) is None


def test_the_last_reply_is_read_only_with_previous_tags():
    turn = TurnContext([Message("user", "Hi.")], LabelSpace.for_tts("minimax"), ExpressionPolicy())
    asked = replace(
        turn, previous=[SentenceLabels(delivery="sympathetic"), SentenceLabels(sound_before="breath")]
    )
    assert "last reply" not in encode(asked, [], "Hello.", Layout())[1]
    assert "last reply none" in encode(turn, [], "Hello.", Layout(previous_tags=True))[1]
    assert "last reply sympathetic breath" in encode(asked, [], "Hello.", Layout(previous_tags=True))[1]
    assert (
        "last reply plain"
        in encode(replace(turn, previous=[SentenceLabels()]), [], "Hi.", Layout(previous_tags=True))[1]
    )


async def test_the_model_tagger_gives_a_model_its_own_last_reply():
    seen = []

    class Model:
        config = {}

        def predict(self, turn, earlier, sentence):
            seen.append(turn.previous)
            return SentenceLabels(delivery="calm")

    tagger = ModelTagger(Model())
    turn = TurnContext([Message("user", "Hi.")], LabelSpace.for_tts("minimax"))
    for _ in range(2):
        await tagger.start_turn(turn)
        await tagger.tag_sentence("Hello.", [])
    tagger.start_call()
    await tagger.start_turn(turn)
    await tagger.tag_sentence("Hello.", [])
    teachers = [SentenceLabels(sound_before="breath")]
    await tagger.start_turn(replace(turn, previous=teachers))
    await tagger.tag_sentence("Hello.", [])
    assert seen == [None, [SentenceLabels(delivery="calm")], None, teachers]


async def test_a_models_config_caps_a_replys_breaths():
    class Model:
        config = {"breaths_per_reply": 1}

        def predict(self, turn, earlier, sentence):
            return SentenceLabels(sound_before="breath")

    tagger = ModelTagger(Model())
    turn = TurnContext([Message("user", "Hi.")], LabelSpace.for_tts("minimax"))
    for _ in range(2):
        await tagger.start_turn(turn)
        assert [(await tagger.tag_sentence(s, [])).sound_before for s in ("One.", "Two.")] == ["breath", None]


async def test_a_breath_is_drawn_as_a_kind_the_tts_has_and_read_back_as_a_breath():
    class Breather:
        config = {"breath_kinds": {"light breath": 0.6, "deep breath": 0.35, "exhale": 0.05}}

        def predict(self, turn, earlier, sentence):
            return SentenceLabels(sound_before="breath")

    policy = ExpressionPolicy(guidance=[BREATHE])
    space = LabelSpace.for_tts("minimax", policy)
    tagger = ModelTagger(Breather())

    async def kind(caller: str) -> str:
        await tagger.start_turn(TurnContext([Message("user", caller)], space, policy))
        return (await tagger.tag_sentence("Hello.", [])).sound_before

    kinds = [await kind(f"Caller {n}.") for n in range(400)]
    assert await kind("Caller 7.") == kinds[7]
    assert 0.5 < kinds.count("light breath") / 400 < 0.7 and 0.25 < kinds.count("deep breath") / 400 < 0.45
    assert set(kinds) == {"light breath", "deep breath", "exhale"}
    await tagger.start_turn(TurnContext([Message("user", "Again.")], space, policy))
    assert tagger._turn.previous == [SentenceLabels(sound_before="breath")]
    assert render_reply(space, ["Hi."], [SentenceLabels(sound_before="deep breath")]) == "[[inhale]] Hi."
    assert "inhale" not in encode(tagger._turn, [], "Hi.", Layout())[1]
