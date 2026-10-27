"""Writes a call's scenario from its spec: the agent's prompt and the caller's lines, with readings.

It's put in tone-bench's scenario format only so its loader builds the policy and readings.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic
import yaml

from lab import llm
from lab.corpus.specs import Spec
from pipecat_tone.tagger.presets import BREATHE
from pipecat_tone.tagger.readings import TONE_LABELS

if TYPE_CHECKING:
    from bench.scenario import Scenario

WRITER_EFFORT = "medium"
AGENT_SUFFIX = (
    "Your replies are spoken aloud on a phone call, so never use emojis, lists or any formatting that "
    "can't be spoken. Keep replies to one to three short sentences."
)

SYSTEM = """\
You write realistic customer service phone calls for training a voice agent. You write the business's \
instructions for its agent, and the caller's side of the call, one line per caller turn. The agent's \
replies are not written: a real agent will answer each caller line, so each line must make sense after \
any competent agent reply to the one before.

For each caller turn, also write how the caller sounded, as a speech-emotion model would read it from \
their voice: their two likeliest tones among positive, neutral, sad and frustrated, each with a \
confidence from 0 to 1 (the two summing to at most 1), and a sarcasm score from 0 to 1. Readings are \
noisy, like a real model's: mix clear readings (top tone 0.7 or above), middling ones (0.5 to 0.7), weak \
guesses (below 0.5), and now and then one that disagrees with the words. Sarcasm is low except where \
the caller is actually sarcastic.

The agent's instructions: who the agent is, the business, what it can and can't do, and the specific \
facts it needs, written as a short system prompt. Include specific details the agent will have to give \
the caller, such as a confirmation code, an order or ticket number, a phone number, an address or an \
email, written as they'd be read (for example a code like 4TR8KW or a number like 3051-7724). Keep it to \
one business and one problem, with a believable resolution.

The caller speaks naturally, as people do on the phone: short turns, sometimes rambling, never \
narrating stage directions. Follow the persona, mood arc and events you are given, spread over the \
call, and end the call naturally."""

SCHEMA = llm.obj(
    title=llm.STRING,
    agent_prompt=llm.STRING,
    details=llm.array(llm.STRING),
    turns=llm.array(
        llm.obj(
            caller=llm.STRING,
            tone=llm.array(
                llm.obj(label={"type": "string", "enum": list(TONE_LABELS)}, confidence=llm.NUMBER)
            ),
            sarcasm=llm.NUMBER,
        )
    ),
)


async def write_scenario(spec: Spec, client: anthropic.AsyncAnthropic) -> tuple[dict, dict]:
    """The scenario in tone-bench's format, and what writing it cost."""
    prompt = "\n".join(
        [
            f"Business: {spec.domain}",
            f"Caller: {spec.persona}",
            f"Mood over the call: {spec.mood_arc}",
            "Events, in any order:",
            *[f"- {e}" for e in spec.events],
            f"Caller turns: {spec.turns}",
        ]
    )
    reply = await llm.claude_json(client, SYSTEM, prompt, SCHEMA, WRITER_EFFORT, max_tokens=32000)
    if reply.error:
        raise RuntimeError(f"writer {reply.error}")
    return _to_scenario(spec, reply.data), {"usage": reply.usage, "cost_usd": reply.cost_usd}


def load(raw: dict) -> "Scenario":
    """A tone-bench Scenario from a scenario dict, through tone-bench's loader."""
    from bench.scenario import load_scenario

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f, sort_keys=False)
        path = Path(f.name)
    try:
        return load_scenario(path)
    finally:
        path.unlink()


def _to_scenario(spec: Spec, data: dict) -> dict:
    # tone-bench's loader names Pipecat's presets only; the tagger's BREATHE goes in as its words.
    presets = ["MATCH_STYLE", "SLOW_FOR_DETAILS", *([BREATHE] if spec.breathe else [])]
    policy: dict = {"guidance": [*presets, *spec.guidance]}
    if spec.delivery is not None:
        policy["delivery"] = spec.delivery
    if spec.sounds is not None:
        policy["sounds"] = spec.sounds
    turns = []
    for t in data["turns"]:
        tones = sorted(t["tone"], key=lambda x: -x["confidence"])[:2]
        heard = {
            "tone": {x["label"]: round(float(x["confidence"]), 2) for x in tones},
            "sarcasm": round(float(t["sarcasm"]), 2),
        }
        turns.append({"user": t["caller"], "heard": heard})
    return {
        "name": spec.id,
        "title": data["title"],
        "tts": spec.tts,
        "system_prompt": f"{data['agent_prompt'].strip()}\n\n{AGENT_SUFFIX}",
        "policy": policy,
        "slow": [str(d) for d in data["details"]],
        "turns": turns,
    }
