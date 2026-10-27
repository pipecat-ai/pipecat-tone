"""The move question (line tags 0.2): the move a line makes in the conversation, read with the line before it, and the
measures the move is scored by."""

import json
from collections import Counter
from itertools import combinations

from lab.reader import scoring
from lab.reader.committee import (
    MEMBERS,
    clients,
    guide_section,
    majority,
    user_message,
    view,
    worked_examples,
)
from lab.reader.schemas import EXCHANGES, LineInput

GUIDE = "line-tags-guide-0.3.md"

# Worked examples A and B see: one or more per class, the main hard edges.
FEW_SHOT_IDS = (
    "ex-01-room-question",
    "ex-02-knows",
    "ex-03-rumour",
    "ex-05-doesnt-know",
    "ex-06-dodging-question",
    "ex-07-request-for-action",
    "ex-09-couldnt-say",
    "ex-12-npc-answers-npc",
    "ex-13-npc-closes",
    "ex-16-reaction",
    "ex-20-no-thanks-closes",
    "ex-25-points-elsewhere",
    "ex-26-declines-request",
    "ex-34-answers-then-asks",
    "ex-36-no-contact",
    "ex-37-parting-advice",
)


# --- what the labellers read -----------------------------------------------------------------------------------------


def system_prompt() -> str:
    examples = worked_examples("0.2")
    shots = "\n\n".join(
        f"{user_message(view(examples[i].input))}\n"
        f"Answer:\n{json.dumps({'exchange': examples[i].tags.exchange.label})}"
        for i in FEW_SHOT_IDS
    )
    return (
        "You label lines of dialogue from games for a classifier that helps a game decide who speaks next in a "
        "conversation. Follow this guide exactly.\n\n"
        + guide_section(GUIDE, "## Exchange move")
        + "\n\n## Worked examples\n\n"
        + shots
        + "\n\nAnswer with the JSON object only: the exchange move of `line`, read after `previous_line` when there is "
        "one. `speaker`, `said_to` and `people_in_scene` say who is talking to whom; `the player` is the person playing the game."
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["exchange"],
    "properties": {"exchange": {"type": "string", "enum": list(EXCHANGES)}},
}


def label_from_answer(answer: dict) -> str:
    lab = str(answer["exchange"]).strip().lower()
    if lab not in EXCHANGES:
        raise ValueError(f"not an exchange class: {lab[:40]}")
    return lab


FORM = clients.Form(view, OUTPUT_SCHEMA, "exchange", 2_000, lambda answer, _: label_from_answer(answer))


# --- D: a predicate per move -----------------------------------------------------------------------------------------
# Each predicate states its own move with the guide's main edges, and excludes the moves the guide's order puts before
# or beside it.

PREDICATES = {
    "asks": "Does `line` want a reply or an action from the person it is said to (`said_to`), or from the room: a question, "
    "a request, an order, or an invitation to speak? Not a line whose main job is to answer, refuse or close, even "
    "if it ends with a question; not a question that dodges what `previous_line` asked; not a rhetorical question; "
    "not a parting word inside a farewell.",
    "answers": "Does `line` give what `previous_line` asked for, or offer information, stated as fact rather than as a "
    "rumour, a guess or something the speaker is unsure of? A plain yes or no counts, a denial included ('No, I "
    "don't owe anyone.'), and so does agreeing to do what was asked or giving an opinion that was asked for. A no "
    "to 'have you seen it' or 'do you know' does not count.",
    "hedges": "Does `line` offer something only second-hand or unsure: a rumour, hearsay, a guess or a half memory (I think, "
    "maybe, they say, last I heard)?",
    "doesnt_know": "Does `line` say the speaker does not know, has not seen it or cannot tell (including a no to 'have you "
    "seen it' or 'do you know'), or only point to someone else who might know?",
    "withholds": "Does `line` decline, deflect or dodge what was asked: refuse to say, refuse to do it, answer a question with "
    "a question that dodges it, or change the subject? A plain denial or a reluctant answer is not withholding.",
    "closes": "Does `line` wind the exchange down and want no reply: thanks, agreement that settles it, a farewell (with or "
    "without a parting piece of advice), a last word, or a short acknowledgement after an answer?",
    "other": "Is `line` only a remark, an opinion nobody asked for, banter, a greeting, a reaction, a listening sound, or an "
    "insult or threat that asks nothing? Not if it answers, asks, hedges, says it does not know, refuses or closes; "
    "an opinion that `previous_line` asked for is not other.",
}
D_QIDS = {f"q{i + 1}": m for i, m in enumerate(EXCHANGES)}  # neutral question ids
D_ORDER = ("answers", "hedges", "doesnt_know", "withholds", "asks", "closes", "other")
D_PREFACE = (
    "One line of dialogue from a video game, to be labelled for the move it makes in the conversation. The state "
    "below is JSON; the questions refer to its fields by name in backticks.\n"
)


def d_state(inp: LineInput) -> dict:
    about = "a line of dialogue in a video game and the line before it; `the player` is the person playing the game"
    return {"about": about, **view(inp)}


def d_request(inp: LineInput, model: str) -> dict:
    return {
        "model": model,
        "input": D_PREFACE + json.dumps(d_state(inp), indent=1, sort_keys=True, ensure_ascii=False),
        "questions": [
            {"type": "predicate", "name": q, "instructions": PREDICATES[m]} for q, m in D_QIDS.items()
        ],
    }


def d_label(probs: dict[str, float]) -> str:
    """The first move in the guide's order whose probability reaches one half, else the most probable."""
    for m in D_ORDER:
        if probs.get(m, 0.0) >= 0.5:
            return m
    return max(EXCHANGES, key=lambda m: probs.get(m, -1.0))


def parse_d(data: dict, inp: LineInput) -> tuple[str | None, str | None, dict, dict]:
    """(label, error, usage, the seven probabilities). A refused or missing move fails the line: no D vote."""
    answers, refused = {}, False
    for a in data.get("answers") or []:
        if a.get("type") == "predicate" and isinstance(a.get("probability"), (int, float)):
            answers[a["name"]] = float(a["probability"])
        else:
            refused = True
    if set(D_QIDS) - set(answers):
        raise clients.DecisionsError(clients.REFUSAL if refused else clients.INVALID)
    probs = {m: answers[q] for q, m in D_QIDS.items()}
    return d_label(probs), None, data.get("usage") or {}, probs


# --- the majority ----------------------------------------------------------------------------------------------------


def vote(labels: dict[str, str | None]) -> dict:
    """{"label": the majority or None, "agree": how many gave the commonest label, "votes": {member: label}} over the
    members that answered."""
    votes = {m: v for m, v in labels.items() if v is not None}
    label, agree = majority(votes)
    return {"label": label, "agree": agree, "votes": votes}


# --- the committee's agreement: per-class F1 rounds to 4 places, as the move's scorecard does --------------------------


def _macro_f1(gold, pred) -> float:
    return scoring.macro_f1(gold, pred, EXCHANGES, 4)


def _both_ways(pairs: list[tuple[str, str]]) -> float:
    """Macro-F1 between two labellers' labels, both directions averaged."""
    x, y = zip(*pairs)
    return (_macro_f1(x, y) + _macro_f1(y, x)) / 2


def _both(votes: list[dict], a: str, b: str) -> tuple[list[str], list[str]]:
    """Members a and b's labels on the lines both answered, from each line's votes."""
    both = [v for v in votes if v.get(a) and v.get(b)]
    return [v[a] for v in both], [v[b] for v in both]


def _pair(x: list[str], y: list[str]) -> dict:
    """Two members on the same lines: accuracy, kappa, and macro-F1 both directions averaged."""
    return {
        "lines": len(x),
        "accuracy": round(sum(a == b for a, b in zip(x, y)) / max(len(x), 1), 4),
        "kappa": round(scoring.kappa(x, y), 4),
        "macro_f1": round((_macro_f1(x, y) + _macro_f1(y, x)) / 2, 4),
    }


def agreement(votes: list[dict]) -> dict:
    """The committee against itself, from the votes rows: each pair of members on the lines both answered, each
    member's macro-F1 against the majority, the unanimous lines and the commonest disagreements."""
    major = [v for v in votes if v["label"]]
    kinds = Counter(
        " / ".join(sorted(set(v["votes"].values()))) for v in votes if len(set(v["votes"].values())) > 1
    )
    return {
        "pairs": {
            f"{a}-{b}": _pair(*_both([v["votes"] for v in votes], a, b)) for a, b in combinations(MEMBERS, 2)
        },
        "member_vs_majority_macro_f1": {
            m: round(
                _macro_f1(
                    [v["label"] for v in major if v["votes"].get(m)],
                    [v["votes"][m] for v in major if v["votes"].get(m)],
                ),
                4,
            )
            for m in MEMBERS
        },
        "unanimous": sum(1 for v in votes if v["label"] and v["agree"] == 3),
        "disagreement_kinds": dict(kinds.most_common(12)),
    }


def committee_level(rows: list[dict], split: str = "test") -> dict:
    """Labellers A and B on the clean lines of a split, unseen settings left out (rows {"line", "exchange": {"label",
    "votes"}}): on the lines with a majority, the lines a model is scored on; `all_lines` adds those without one."""
    got = [
        r
        for r in rows
        if r["line"]["split"] == split
        and not r["line"].get("twin_of")
        and "unseen_genre" not in (r["line"].get("slices") or [])
    ]

    def level(rs):
        x, y = _both([r["exchange"]["votes"] for r in rs], "a", "b")
        pair = _pair(x, y)
        return {
            "lines": pair["lines"],
            "macro_f1": pair["macro_f1"],
            "macro_f1_ci": scoring.bootstrap(list(zip(x, y)), _both_ways),
            "accuracy": pair["accuracy"],
            "kappa": pair["kappa"],
            "answered_vs_open": round(
                (scoring.answered_vs_open(x, y)["rate"] + scoring.answered_vs_open(y, x)["rate"]) / 2, 4
            ),
            "asks_vs_rest": scoring.asks_vs_rest(x, y)["rate"],
        }

    return {
        **level([r for r in got if r["exchange"]["label"]]),
        "all_lines": level(got),
        "no_majority": sum(1 for r in got if not r["exchange"]["label"]),
        "of": len(got),
    }
