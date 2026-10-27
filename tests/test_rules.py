from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.policy import ExpressionPolicy

from pipecat_tone.tagger import LabelSpace, Message, RulesTagger, TurnContext
from pipecat_tone.tagger.markup import render_reply
from pipecat_tone.tagger.patterns import DETAILS


def turn(text: str, tts: str = "cartesia", tone=None, sarcasm: float = 0.0, policy=None) -> TurnContext:
    heard = []
    if tone:
        heard.append(PerceptionSignal("tone", labels=sorted(tone.items(), key=lambda p: -p[1])))
    heard.append(PerceptionSignal("sarcasm", value=sarcasm))
    return TurnContext(messages=[Message("user", text, heard)], space=LabelSpace.for_tts(tts, policy))


async def tag(t: TurnContext, *sentences: str) -> str:
    tagger = RulesTagger()
    await tagger.start_turn(t)
    labels = [await tagger.tag_sentence(s, list(sentences[:n])) for n, s in enumerate(sentences)]
    return render_reply(t.space, list(sentences), labels)


def test_details():
    found = [m[0] for m in DETAILS.finditer("Ref 7Q2K9X, call 617-555-0193 or 48263, a@b.com, A B 7 4 9.")]
    assert found == ["7Q2K9X", "617-555-0193", "48263", "a@b.com", "A B 7 4 9"]
    assert not list(DETAILS.finditer("It costs $842.19 and posts in 2026, in 3 to 5 days."))


async def test_frustrated_caller_gets_calm_and_codes_are_slowed():
    t = turn("This is ridiculous.", tone={"frustrated": 0.88, "neutral": 0.1})
    assert await tag(t, "I can fix that.", "Your reference is 7Q2K9X.") == (
        "<<calm>> I can fix that. Your reference is <<slow: 7Q2K9X>>."
    )


async def test_a_guess_is_ignored():
    t = turn("Okay.", tone={"frustrated": 0.45, "neutral": 0.4})
    assert await tag(t, "Sure thing.") == "Sure thing."


async def test_sarcasm_isnt_read_as_happy():
    t = turn("Great. Just great.", tone={"positive": 0.6, "frustrated": 0.3}, sarcasm=0.86)
    assert await tag(t, "I hear you.") == "<<calm>> I hear you."


async def test_apology_falls_back_when_masked():
    # MiniMax has neither apologetic nor sympathetic, so an apology is calm.
    t = turn("Hi.", tts="minimax", tone={"neutral": 0.8})
    assert await tag(t, "I apologize for the wait.") == "<<calm>> I apologize for the wait."


async def test_joke_gets_a_chuckle_where_the_tts_has_one():
    t = turn("Haha, my cat ordered it.", tts="elevenlabs", tone={"positive": 0.8})
    assert (
        await tag(t, "That's a first!", "Let me cancel it.")
        == "<<warmly>> [[chuckles]] That's a first! Let me cancel it."
    )
    policy = ExpressionPolicy(sounds=[])
    assert "[[" not in await tag(turn("Haha.", tts="elevenlabs", policy=policy), "Ha.")
