"""The labelling committee: three models from three families, A, B and D, answer each question about a line on their
own, and two matching votes make a label."""

import json
from collections import Counter
from pathlib import Path

import yaml

from lab.reader.schemas import LineInput, LineTagRecord, LineTagRecord02

MEMBERS = ("a", "b", "d")
# The labellers' model ids: read by name from the environment, never written down.
MODEL_ENV = {m: f"LABELLER_{m.upper()}_MODEL" for m in MEMBERS}
QUORUM = 2
GUIDES = Path(__file__).with_name("guides")


def guide(name: str) -> str:
    return (GUIDES / name).read_text(encoding="utf-8")


def guide_section(name: str, heading: str) -> str:
    """A guide's section, from its heading to the next top-level heading."""
    g = guide(name)
    start = g.index(heading)
    end = g.index("\n## ", start + 1)
    return g[start:end].strip()


def worked_examples(version: str) -> dict[str, LineTagRecord]:
    """A label guide's worked examples by id: line tags 0.1 or 0.2."""
    record = LineTagRecord02 if version == "0.2" else LineTagRecord
    rows = yaml.safe_load(guide(f"line-tags-{version}.examples.yaml"))
    return {e.line_id: e for e in (record.model_validate(r) for r in rows)}


def view(inp: LineInput, previous_first: bool = True) -> dict:
    """The line as the committee sees it, as unsorted JSON: who speaks to whom, then the line before it and the line,
    or the line and then the line before it."""
    out = {
        "speaker": inp.speaker,
        "speaker_is": "the player" if inp.speaker_kind == "player" else "a character",
        "said_to": inp.to or "the room",
        "people_in_scene": list(inp.targets),
    }
    before = {"previous_line": inp.previous} if inp.previous else {}
    if previous_first:
        return {**out, **before, "line": inp.text}
    return {**out, "line": inp.text, **before}


def user_message(shown: dict) -> str:
    return "Input:\n" + json.dumps(shown, ensure_ascii=False)


def majority(votes: dict) -> tuple:
    """(the label QUORUM or more members gave and no other label matched, or None; how many gave the commonest)."""
    counts = Counter(votes.values()).most_common()
    if not counts:
        return None, 0
    top, n = counts[0]
    if n < QUORUM or (len(counts) > 1 and counts[1][1] == n):
        return None, n
    return top, n
