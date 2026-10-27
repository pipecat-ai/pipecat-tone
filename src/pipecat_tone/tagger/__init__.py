"""The tone tagger: labels each sentence of an LLM's plain reply and renders the markup its TTS performs."""

from .labels import LabelSpace, Pause, SentenceLabels, Span
from .markup import ReplyRenderer
from .model import ModelTagger, ToneTaggerModel
from .presets import BREATHE
from .processor import PlainContext, ToneTaggerParams, ToneTaggerProcessor, llm_policy
from .rules import RulesTagger
from .tagger import Message, NoOpTagger, Tagger, TurnContext

__all__ = [
    "BREATHE",
    "LabelSpace",
    "Message",
    "ModelTagger",
    "NoOpTagger",
    "Pause",
    "PlainContext",
    "ReplyRenderer",
    "RulesTagger",
    "SentenceLabels",
    "Span",
    "Tagger",
    "ToneTaggerModel",
    "ToneTaggerParams",
    "ToneTaggerProcessor",
    "TurnContext",
    "llm_policy",
]
