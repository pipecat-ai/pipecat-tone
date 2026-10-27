"""Writes a batch's training rows beside its calls: train, dev and test (split by domain), and stats.json.

    uv run python -m lab.corpus.export data/corpus/v0

A row is trained on only when its reply passed review, its markup speaks exactly the plain words, and
it has no format issue; the rest stay as context. Every call is checked against tone-bench's scenarios.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pipecat.expressive.policy import ExpressionPolicy

from lab.corpus.breaths import BREATH_VERSION, with_breaths, without_breaths
from lab.corpus.build import call_cost, call_error, is_call, is_stale
from lab.corpus.contamination import closest
from lab.corpus.rows import Row
from lab.corpus.scenario import load
from lab.corpus.teacher import TEACHER_VERSION
from pipecat_tone.tagger import BREATHE, LabelSpace, SentenceLabels

SPLITS = ("train", "dev", "test")


def call_rows(call: dict) -> Iterator[Row]:
    """The call's rows, then its breath variant's, once it has a current breath pass. A call that asks to breathe
    has no variant: its own rows take the pass's breaths in place of its teacher's."""
    scenario = load(call["scenario"])
    turns = call["turns"]
    plain = [[SentenceLabels.from_dict(d) for d in t["labels"]] for t in turns]
    own = [_trainable(t) for t in turns]
    breaths = call.get("breaths") or {}
    current = breaths.get("version") == BREATH_VERSION
    asks = BREATHE in scenario.policy.guidance
    if not (asks and current):
        yield from _rows(call, call["spec"]["id"], scenario.tts, scenario.policy, plain, own)
    if not current:
        return
    if breaths["offered"]:
        labels = [with_breaths(without_breaths(lab), v["breaths"]) for lab, v in zip(plain, breaths["turns"])]
        trainable = [_trainable(v) for v in breaths["turns"]]
    else:
        labels, trainable = plain, own
    name = call["spec"]["id"] if asks else f"{call['spec']['id']}-breathe"
    yield from _rows(call, name, scenario.tts, ExpressionPolicy(**breaths["policy"]), labels, trainable)


def _trainable(turn: dict) -> bool:
    return turn.get("verdict") == "pass" and turn["same_words"] and not turn["format_issues"]


def _rows(
    call: dict,
    call_id: str,
    tts: str,
    policy: ExpressionPolicy,
    labels: list[list[SentenceLabels]],
    trainable: list[bool],
) -> Iterator[Row]:
    spec = call["spec"]
    space = LabelSpace.for_tts(tts, policy)
    history: list[dict] = []
    for t, turn in enumerate(call["turns"]):
        history.append({"role": "user", "text": turn["user"], "heard": turn["heard"]})
        for n, sentence in enumerate(turn["sentences"]):
            yield Row(
                call=call_id,
                turn=t,
                n=n,
                split=spec["split"],
                domain=spec["domain"],
                agent_model=spec["agent_model"],
                space=space,
                guidance=policy.guidance,
                history=list(history),
                earlier=turn["sentences"][:n],
                sentence=sentence,
                labels=labels[t][n],
                trainable=trainable[t],
            )
        history.append(
            {
                "role": "assistant",
                "text": " ".join(turn["sentences"]),
                "labels": [lab.to_dict() for lab in labels[t]],
            }
        )


def export(folder: Path) -> dict:
    calls = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if is_call(p.name)]
    # Labels from an older teacher wait until the call is retaught.
    errors = [c for c in calls if call_error(c)]
    stale = [c for c in calls if c not in errors and is_stale(c, "teacher")]
    ok = [c for c in calls if c not in errors and c not in stale]
    stats: Counter = Counter()
    deliveries: Counter = Counter()
    out = {split: (folder / f"{split}.jsonl").open("w") for split in SPLITS}
    for call in ok:
        for row in call_rows(call):
            out[row.split].write(json.dumps(row.to_dict()) + "\n")
            stats[f"{row.split}_rows"] += 1
            stats[f"{row.split}_trainable"] += row.trainable
            if row.trainable:
                deliveries[row.labels.delivery or "none"] += 1
                stats["spans"] += len(row.labels.spans)
                stats["emphasis"] += len(row.labels.emphasis)
                stats["sounds"] += bool(row.labels.sound_before) + bool(row.labels.sound_after)
                stats["breaths"] += row.labels.sound_before == "breath"
            if row.call.endswith("-breathe"):
                stats[f"{row.split}_breathe_rows"] += 1
    for f in out.values():
        f.close()
    closest_calls = sorted(((closest(c["scenario"]), c["spec"]["id"]) for c in ok), reverse=True)[:5]
    summary = {
        "calls": len(calls),
        "teacher_version": TEACHER_VERSION,
        "stale": len(stale),
        "errors": len(errors),
        **stats,
        "deliveries": dict(deliveries.most_common()),
        "cost_usd": round(sum(call_cost(c) for c in calls), 2),
        "closest_to_tone_bench": [
            {"call": cid, "scenario": name, "overlap": round(score, 3)}
            for (score, name), cid in closest_calls
        ],
    }
    (folder / "stats.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    print(json.dumps(export(parser.parse_args().folder), indent=2))
