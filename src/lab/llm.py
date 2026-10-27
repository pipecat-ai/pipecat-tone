"""Claude and OpenAI calls that return JSON from a strict schema, with usage and errors (and Claude's cost) in one shape."""

import json
import os
from dataclasses import dataclass, field

import anthropic
import openai
from openai import AsyncOpenAI

# Model ids are read by name from the environment, never written down: in a job, from the tone-tagger-keys secret.
CLAUDE = "LABELLER_B_MODEL"  # the Claude model that writes scenarios and teaches; the reader's labeller B
# $ per million tokens at Standard for that model; batches halve every rate. Cache reads are a twentieth of input.
CLAUDE_PRICES = {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00}
# The Messages API's usage field for each price.
CLAUDE_USAGE = {
    "input": "input_tokens",
    "cache_write": "cache_creation_input_tokens",
    "cache_read": "cache_read_input_tokens",
    "output": "output_tokens",
}
STOPS = ("refusal", "max_tokens")


def model_id(env: str) -> str:
    """A model's id from the environment variable that names it."""
    model = os.environ.get(env)
    if not model:
        raise RuntimeError(f"set {env} in .env and the tone-tagger-keys secret: a model id")
    return model


@dataclass
class Reply:
    data: dict | None
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None


async def claude_json(
    client: anthropic.AsyncAnthropic,
    system: str,
    prompt: str,
    schema: dict,
    effort: str,
    max_tokens: int = 64000,
) -> Reply:
    try:
        async with client.beta.messages.stream(
            model=model_id(CLAUDE),
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = await stream.get_final_message()
    except anthropic.APIError as e:
        return Reply(None, error=f"{type(e).__name__}: {e}")
    message = response.to_dict()
    usage = {k: message["usage"].get(name) or 0 for k, name in CLAUDE_USAGE.items()}
    cost = round(sum(usage[k] * CLAUDE_PRICES[k] for k in usage) / 1e6, 5)
    try:
        data, stop = read_message(message)
    except ValueError:
        return Reply(None, usage, cost, "invalid JSON")
    if stop:
        return Reply(None, usage, cost, f"stopped: {stop}")
    return Reply(data, usage, cost)


def read_message(message: dict) -> tuple:
    """A Messages API response's JSON answer from its first text block, or the stop reason that left it without one:
    (answer, None) or (None, refusal or max_tokens). Text that isn't JSON raises ValueError."""
    if message.get("stop_reason") in STOPS:
        return None, message["stop_reason"]
    text = next((b.get("text", "") for b in message.get("content") or [] if b.get("type") == "text"), "")
    return json.loads(text), None


async def openai_json(
    client: AsyncOpenAI, model: str, effort: str, system: str, prompt: str, schema: dict
) -> Reply:
    try:
        response = await client.chat.completions.create(
            model=model,
            reasoning_effort=effort,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema, "strict": True},
            },
        )
    except openai.APIError as e:
        return Reply(None, error=f"{type(e).__name__}: {e}")
    usage = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
    try:
        return Reply(json.loads(response.choices[0].message.content or ""), usage)
    except json.JSONDecodeError:
        return Reply(None, usage, error="invalid JSON")


def obj(**properties) -> dict:
    """A strict object schema: every property required, nothing else allowed."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def array(items: dict) -> dict:
    return {"type": "array", "items": items}


def enum_or_null(names: list[str]) -> dict:
    return {"anyOf": [{"type": "string", "enum": names}, {"type": "null"}]}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}


def call_lines(guidance: list[str], turns: list[tuple[str, str | list[str]]]) -> list[str]:
    """The developer's guidance and the call as prompt lines: each turn's caller line, then the agent's
    reply, numbered when it's a list of sentences."""
    lines = ["Developer's guidance for this call:", *[f"- {line}" for line in guidance], "", "<call>"]
    for t, (caller, agent) in enumerate(turns, 1):
        lines += [f"Turn {t}", f"Caller: {caller}"]
        if isinstance(agent, str):
            lines.append(f"Agent: {agent}")
        else:
            lines += ["Agent:", *[f"  ({n}) {s}" for n, s in enumerate(agent, 1)]]
        lines.append("")
    return [*lines, "</call>"]
