"""The interface every tagger implements: a turn pass per reply, then a fast sentence pass per sentence."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.policy import ExpressionPolicy

from .labels import LabelSpace, SentenceLabels


@dataclass
class Message:
    """A message of the call: role (``user`` or ``assistant``), plain words, and a caller's reading."""

    role: str
    text: str
    heard: list[PerceptionSignal] | None = None


@dataclass
class TurnContext:
    """What a tagger knows when a reply starts: the call so far, its label space, the full policy, and the labels it
    gave the agent's previous reply, if it tagged one."""

    messages: list[Message]
    space: LabelSpace
    policy: ExpressionPolicy = field(default_factory=ExpressionPolicy)
    previous: list[SentenceLabels] | None = None

    @property
    def caller(self) -> Message | None:
        """The caller's message this reply answers."""
        return next((m for m in reversed(self.messages) if m.role == "user"), None)

    @property
    def heard(self) -> list[PerceptionSignal] | None:
        return self.caller.heard if self.caller else None


class Tagger(ABC):
    """Chooses expression labels for the agent's replies."""

    name: str = "tagger"

    def start_call(self) -> None:
        """Forget the last call."""

    async def start_turn(self, turn: TurnContext) -> None:
        """Read the call before the reply's first sentence."""

    @abstractmethod
    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        """Label one sentence, given the reply's earlier sentences, all without markup."""


class NoOpTagger(Tagger):
    """Never tags: plain replies."""

    name = "noop"

    async def tag_sentence(self, sentence: str, earlier: list[str]) -> SentenceLabels:
        return SentenceLabels()
