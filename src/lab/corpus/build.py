"""Builds one corpus call: the scenario, the call held plain, the teacher's labels, checks and review.

The call is held through ``lab.bench``'s pipeline with the no-op tagger, so the deployment LLM gets
the instruction it gets in production and its replies come back plain, split as the tagger splits them.
"""

import traceback

import anthropic
from openai import AsyncOpenAI

from lab import llm
from lab.corpus import breaths, review, teacher
from lab.corpus.review import REVIEWER, REVIEWER_VERSION, Reviewer
from lab.corpus.scenario import load, write_scenario
from lab.corpus.specs import Spec
from lab.corpus.teacher import TEACHER_VERSION, Exchange, Teacher
from pipecat_tone.tagger import LabelSpace, NoOpTagger
from pipecat_tone.tagger.markup import keep, render_reply, same_words

# What a call keeps of each turn when it's labeled again.
HELD = ("user", "heard", "user_message", "reply", "sentences")
# Each labeler's current version, and the version before breaths.
VERSIONS = {
    "teacher": (TEACHER_VERSION, teacher.BREATHLESS_VERSION),
    "reviewer": (REVIEWER_VERSION, review.BREATHLESS_VERSION),
}


async def build_call(spec: Spec) -> dict:
    """The call's record, or its spec and the error that stopped it."""
    return await _recorded({"spec": spec.to_dict()}, "build", _build(spec))


async def reteach_call(call: dict) -> dict:
    """The call labeled and reviewed again by the current teacher; its scenario and replies are kept, and its breath
    variant is made again from the new labels."""
    call = {k: v for k, v in call.items() if k != "breaths"}
    return await _recorded(call, "reteach", _labeled(call, call["turns"]))


async def rereview_call(call: dict) -> dict:
    """The call's tagging reviewed again with the current rubric; labels and markup are kept."""
    return await _recorded(call, "rereview", _labeled(call, call["turns"], teach=False))


async def breathe_call(call: dict) -> dict:
    """The call with its breath variant; a failure is kept apart, so the call itself still trains."""
    try:
        done = await breaths.breathe_call(call)
    except Exception as e:
        return {**call, "breaths_error": f"{type(e).__name__}: {e}"}
    return {k: v for k, v in done.items() if k != "breaths_error"}


def is_call(name: str) -> bool:
    """Whether a batch's file is a call, not its stats."""
    return name.endswith(".json") and not name.endswith("stats.json")


def call_error(call: dict) -> str | None:
    """What stopped the call or its teacher, if anything."""
    return call.get("error") or (call.get("teacher") or {}).get("error")


def call_cost(call: dict) -> float:
    breath_teacher = (call.get("breaths") or {}).get("teacher") or {}
    return sum((call.get(k) or {}).get("cost_usd", 0) for k in ("teacher", "writer")) + breath_teacher.get(
        "cost_usd", 0
    )


def is_stale(call: dict, role: str) -> bool:
    """Whether a built call was labeled by an older ``teacher`` or ``reviewer`` than it needs (1 from before
    versions). A call whose policy offers no breath needs only the version before breaths."""
    if "turns" not in call:
        return False
    current, breathless = VERSIONS[role]
    return (call.get(role) or {}).get("version", 1) < (current if offers_breath(call) else breathless)


def offers_breath(call: dict) -> bool:
    scenario = load(call["scenario"])
    return "breath" in LabelSpace.for_tts(scenario.tts, scenario.policy).sounds


async def _recorded(base: dict, stage: str, work) -> dict:
    """The work's result, or ``base`` with the error that stopped it: one failed call never stops a batch."""
    try:
        return await work
    except Exception as e:
        return {**base, "error": f"{stage}: {type(e).__name__}: {e}", "trace": traceback.format_exc()}


async def _build(spec: Spec) -> dict:
    from lab.bench import model_config
    from lab.bench.agent import run_call

    raw, writer = await write_scenario(spec, anthropic.AsyncAnthropic())
    instruction, records = await run_call(load(raw), model_config(spec.agent_model), NoOpTagger())
    turns = [
        {
            "user": r.user,
            "heard": r.heard,
            "user_message": r.user_message,
            "reply": r.reply,
            "sentences": r.tagger["sentences"],
        }
        for r in records
    ]
    call = {"spec": spec.to_dict(), "scenario": raw, "instruction": instruction, "writer": writer}
    return await _labeled(call, turns)


async def _labeled(call: dict, turns: list[dict], teach: bool = True) -> dict:
    """The call with its turns labeled by the teacher (unless ``teach`` is off), rendered, checked and reviewed."""
    from bench.format_check import check_reply
    from bench.profiles import tts_adapter

    scenario = load(call["scenario"])
    adapter = tts_adapter(scenario.tts)
    labeled = dict(call)
    if teach:
        turns = [{k: t[k] for k in HELD} for t in turns]
        space = LabelSpace.for_tts(scenario.tts, scenario.policy)
        exchanges = [Exchange(st.user, st.heard, t["sentences"]) for st, t in zip(scenario.turns, turns)]
        teaching = await Teacher(anthropic.AsyncAnthropic()).label(exchanges, space, scenario.policy)
        for turn, labels in zip(turns, teaching.labels):
            kept = [keep(s, label, space) for s, label in zip(turn["sentences"], labels)]
            turn["labels"] = [k.to_dict() for k in kept]
            turn["markup"] = render_reply(space, turn["sentences"], kept)
            turn["format_issues"] = check_reply(
                turn["markup"], adapter, scenario.policy, slow=scenario.slow
            ).issues
            turn["same_words"] = same_words(" ".join(turn["sentences"]), turn["markup"], adapter)
        labeled["teacher"] = {
            "version": TEACHER_VERSION,
            "usage": teaching.usage,
            "cost_usd": teaching.cost_usd,
            "error": teaching.error,
        }

    reviewed = [(t["user_message"], t["markup"]) for t in turns]
    reviewer = Reviewer(AsyncOpenAI())
    verdicts, usage = await reviewer.review(scenario.policy.guidance, reviewed) if turns else ([], {})
    labeled["turns"] = [{**t, "verdict": v["verdict"], "note": v["note"]} for t, v in zip(turns, verdicts)]
    labeled["reviewer"] = {"model": llm.model_id(REVIEWER), "version": REVIEWER_VERSION, "usage": usage}
    return labeled
