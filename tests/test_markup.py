import pytest
from pipecat.expressive.expression import EMPHASIS_DELIVERY
from pipecat.expressive.formatter import ExpressionFormatter, Sound, Text

from pipecat_tone.tagger.labels import TTS_ADAPTERS, LabelSpace, Pause, SentenceLabels, Span
from pipecat_tone.tagger.markup import (
    ReplyRenderer,
    render_reply,
    same_words,
    without_markup,
    without_tts_markup,
)


def renderer(tts: str) -> ReplyRenderer:
    return ReplyRenderer(LabelSpace.for_tts(tts))


def test_delivery_is_written_only_when_it_changes():
    r = renderer("cartesia")
    assert r.render("I'm sorry about that.", SentenceLabels(delivery="sympathetic")) == (
        "<<sympathetic>> I'm sorry about that."
    )
    assert r.render("It happens.", SentenceLabels(delivery="sympathetic")) == "It happens."
    assert r.render("Anything else?", SentenceLabels()) == "<<neutral>> Anything else?"


def test_no_labels_is_the_sentence_unchanged():
    assert renderer("fish").render("Sure, one moment.", SentenceLabels()) == "Sure, one moment."


def test_slow_span():
    sentence = "Your reference is 7 Q 2 K 9 X."
    start = sentence.index("7")
    labels = SentenceLabels(spans=[Span(start, start + len("7 Q 2 K 9 X"), "slow")])
    assert renderer("minimax").render(sentence, labels) == "Your reference is <<slow: 7 Q 2 K 9 X>>."


def test_emphasis_only_where_the_tts_stresses():
    sentence = "It's not your fault."
    labels = SentenceLabels(emphasis=[(5, 8)])
    assert renderer("elevenlabs").render(sentence, labels) == "It's ^^not^^ your fault."
    assert renderer("cartesia").render(sentence, labels) == sentence


def test_sounds_and_pause():
    r = renderer("elevenlabs")
    labels = SentenceLabels(sound_before="laugh", pauses=[Pause(4, 400)])
    assert r.render("Fair point, really.", labels) == "[[laughs]] Fair [[pause 400]] point, really."


def test_pause_clamped_to_the_tts_range():
    labels = SentenceLabels(pauses=[Pause(12, 4000)])
    assert renderer("fish").render("Let me check that.", labels) == "Let me check [[pause 1500]] that."


def test_masked_and_overlapping_labels_are_dropped():
    r = renderer("minimax")
    sentence = "I am so sorry about this."
    labels = SentenceLabels(
        delivery="sympathetic",  # minimax can't
        spans=[Span(0, 4, "slow"), Span(2, 7, "slow")],  # the second overlaps the first
        pauses=[Pause(1, 200)],  # inside the first span
        sound_after="sigh",
    )
    assert r.render(sentence, labels) == "<<slow: I am>> so sorry about this. [[sighs]]"


def test_a_range_that_splits_a_word_or_runs_past_the_end_is_dropped():
    r = renderer("elevenlabs")
    labels = SentenceLabels(emphasis=[(0, 50), (6, 7), (5, 8), (6, 11)])
    assert r.render("Hello there.", labels) == "Hello ^^there^^."


CASES = [
    ("cartesia", "Your code is 4 8 2 6 3.", SentenceLabels(delivery="calm", spans=[Span(13, 22, "slow")])),
    ("minimax", "Oh no, that's awful.", SentenceLabels(delivery="sad", sound_before="sigh")),
    (
        "elevenlabs",
        "Great news, you're all set!",
        SentenceLabels(delivery="excited", emphasis=[(0, 5)], pauses=[Pause(11, 300)]),
    ),
    ("fish", "Ha, that's a good one.", SentenceLabels(delivery="warm", sound_before="laugh")),
]


@pytest.mark.parametrize("tts,sentence,labels", CASES)
def test_markup_speaks_exactly_the_plain_words(tts, sentence, labels):
    markup = renderer(tts).render(sentence, labels)
    assert markup != sentence
    assert same_words(sentence, markup, TTS_ADAPTERS[tts]())


@pytest.mark.parametrize("tts,sentence,labels", CASES)
def test_pipecat_reads_what_the_renderer_writes(tts, sentence, labels):
    space = LabelSpace.for_tts(tts)
    segments = ExpressionFormatter(TTS_ADAPTERS[tts], owner="test").parse(
        renderer(tts).render(sentence, labels), whole=True
    )
    words = [s for s in segments if isinstance(s, Text) and s.text.strip()]
    assert all(space.deliveries[labels.delivery] in w.delivery for w in words)
    for span in labels.spans:
        assert any(
            w.text == sentence[span.start : span.end] and space.deliveries[span.delivery] in w.span
            for w in words
        )
    for start, end in labels.emphasis:
        assert any(w.text == sentence[start:end] and EMPHASIS_DELIVERY in w.delivery for w in words)
    sounds = [s.name for s in segments if isinstance(s, Sound)]
    assert sounds == [space.sounds[s] for s in (labels.sound_before, labels.sound_after) if s]


def test_without_markup_keeps_only_the_original_whitespace():
    assert without_markup("<<sympathetic>> I'm") == "I'm"
    assert without_markup(" <<neutral>> Bye.") == " Bye."
    assert without_markup("<<slow: NW-48213>>.") == "NW-48213."
    assert without_markup("plain  text ,") == "plain  text ,"
    assert without_markup("It's ^^not^^ you [[laughs]], really.") == "It's not you, really."


def test_without_tts_markup():
    assert without_tts_markup('<emotion value="sympathetic"/> I\'m sorry.') == "I'm sorry."
    assert without_tts_markup("[sympathetic] Sorry, ") == "Sorry, "


def test_render_reply_joins_sentences_as_a_stream_would():
    labels = [SentenceLabels(delivery="sympathetic"), SentenceLabels()]
    markup = render_reply(LabelSpace.for_tts("cartesia"), ["I'm sorry.", "Anything else?"], labels)
    assert markup == "<<sympathetic>> I'm sorry. <<neutral>> Anything else?"
