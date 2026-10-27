"""Training rows: one per agent sentence of a built call, with the call so far as the tagger saw it."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

from pipecat.expressive.policy import ExpressionPolicy

from pipecat_tone.tagger import LabelSpace, Message, SentenceLabels, TurnContext, readings


@dataclass
class Row:
    """A sentence, its call and reply so far, and the teacher's labels, which are trained on if ``trainable``."""

    call: str
    turn: int
    n: int
    split: str
    domain: str
    agent_model: str
    space: LabelSpace
    guidance: list[str]
    history: list[
        dict
    ]  # {role, text, heard} and the agent's {role, text, labels}, ``heard`` as Pipecat formats it
    earlier: list[str]
    sentence: str
    labels: SentenceLabels
    trainable: bool

    def turn_context(self) -> TurnContext:
        messages = [
            Message(h["role"], h["text"], readings.parse(h["heard"])[0] if h.get("heard") else None)
            for h in self.history
        ]
        # The teacher's labels on the agent's previous reply, as the runtime gives a model its own.
        replies = [h for h in self.history if h["role"] == "assistant" and "labels" in h]
        previous = [SentenceLabels.from_dict(d) for d in replies[-1]["labels"]] if replies else None
        return TurnContext(messages, self.space, ExpressionPolicy(guidance=self.guidance), previous)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "space"}
        return {**d, **self.space.to_dict(), "labels": self.labels.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        plain = {f.name for f in fields(cls)} - {"space", "labels"}
        return cls(
            **{k: d[k] for k in plain},
            space=LabelSpace.from_dict(d),
            labels=SentenceLabels.from_dict(d["labels"]),
        )


def load_rows(path: Path) -> list[Row]:
    """A split's trainable rows."""
    rows = (Row.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip())
    return [r for r in rows if r.trainable]
