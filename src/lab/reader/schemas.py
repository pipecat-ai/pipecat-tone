"""The lines and labels the reader's pipeline reads: the line classifier's line tags, in which every committee label set is
written (emotion, an act toward each person present and backchannel in 0.1, the move in 0.2, sarcasm in 0.3), and the
reader's contract, ``tone-text-0.1``: one line and the line before it, no names, in Tone's classes."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT = "tone-text-0.1"
SCHEMA_VERSION = "line-tags-0.1"
SCHEMA_VERSION_02 = "line-tags-0.2"
PLAYER = "player"
# Characters in a line, and in the line before it as "Name: text".
MAX_TEXT, MAX_PREVIOUS = 400, 480

Emotion = Literal[
    "neutral", "happy", "amused", "angry", "sad", "afraid", "surprised", "disgusted", "contemptuous"
]
EMOTIONS: tuple[str, ...] = Emotion.__args__

Act = Literal[
    "insult",
    "threat",
    "accusation",
    "tease",
    "demand",
    "dismiss",
    "refusal",
    "request",
    "thanks",
    "praise",
    "apology",
    "comfort",
    "warning",
    "none",
]
ACTS: tuple[str, ...] = Act.__args__

Intensity = Literal["low", "medium", "high"]
INTENSITIES: tuple[str, ...] = Intensity.__args__

Exchange = Literal["asks", "answers", "hedges", "doesnt_know", "withholds", "closes", "other"]
EXCHANGES: tuple[str, ...] = Exchange.__args__
MOVES = EXCHANGES
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}
# After either of these, a question is still open: someone else may answer it.
STILL_OPEN = ("doesnt_know", "withholds")

# The hostile acts; `demand` counts only at high intensity.
HOSTILE_ACTS = ("insult", "threat", "accusation")

Name = Annotated[str, StringConstraints(min_length=1, max_length=40, strip_whitespace=True)]
LineText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT)]
LineId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.:-]*$", max_length=120)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
MAX_TARGETS = 12


def is_hostile(act: str, intensity: str) -> bool:
    return act in HOSTILE_ACTS or (act == "demand" and intensity == "high")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LineInput(Strict):
    """One line and the people in the scene."""

    speaker: Name
    speaker_kind: Literal["player", "npc"]
    to: Name | None = None  # who it was said to; None: to the room
    targets: list[Name] = Field(min_length=1, max_length=MAX_TARGETS)
    text: LineText
    previous: str | None = Field(default=None, max_length=MAX_PREVIOUS)  # one earlier line, "Name: text"

    @model_validator(mode="after")
    def _cast(self) -> "LineInput":
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be distinct")
        if self.speaker in self.targets:
            raise ValueError("the speaker is not a target of their own line")
        if (self.speaker == PLAYER) != (self.speaker_kind == "player"):
            raise ValueError("the player speaks as `player`, and only the player does")
        if self.to is not None and self.to not in self.targets:
            raise ValueError(f"`to` ({self.to}) must be one of the targets")
        return self


class ActTag(Strict):
    act: Act
    intensity: Intensity = "low"
    confidence: Probability | None = None

    @model_validator(mode="after")
    def _none_is_low(self) -> "ActTag":
        if self.act == "none" and self.intensity != "low":
            raise ValueError("an act of none has intensity low")
        return self


class LineTags(Strict):
    emotion: Emotion
    emotion_intensity: Intensity = "low"
    emotion_confidence: Probability | None = None
    acts: dict[Name, ActTag]
    backchannel: bool = False

    @model_validator(mode="after")
    def _neutral_and_backchannel(self) -> "LineTags":
        if self.emotion == "neutral" and self.emotion_intensity != "low":
            raise ValueError("a neutral emotion has intensity low")
        if self.backchannel and any(a.act != "none" for a in self.acts.values()):
            raise ValueError("a backchannel carries no act toward anyone")
        return self


class LineTagRecord(Strict):
    """A labelled line: the label guide's worked examples."""

    schema_version: Literal["line-tags-0.1"] = SCHEMA_VERSION
    line_id: LineId
    input: LineInput
    tags: LineTags
    label_source: Literal["guide", "committee", "person", "model"] = "guide"
    slices: list[Annotated[str, StringConstraints(pattern=r"^[a-z_]+$")]] = []
    notes: str | None = None

    @model_validator(mode="after")
    def _one_act_per_target(self) -> "LineTagRecord":
        if set(self.tags.acts) != set(self.input.targets):
            missing = sorted(set(self.input.targets) - set(self.tags.acts))
            extra = sorted(set(self.tags.acts) - set(self.input.targets))
            raise ValueError(f"acts must cover exactly the targets (missing {missing}, extra {extra})")
        return self


class ExchangeTag(Strict):
    """The move a line makes, read with the line before it."""

    label: Exchange
    confidence: Probability | None = None


class LineTags02(LineTags):
    exchange: ExchangeTag | None = None


class LineTagRecord02(LineTagRecord):
    schema_version: Literal["line-tags-0.1", "line-tags-0.2"] = SCHEMA_VERSION_02
    tags: LineTags02

    @model_validator(mode="after")
    def _exchange_needs_02(self) -> "LineTagRecord02":
        if self.tags.exchange is not None and self.schema_version != SCHEMA_VERSION_02:
            raise ValueError(
                "`exchange` is a line-tags-0.2 field: the record must say schema_version line-tags-0.2"
            )
        return self


class SarcasmTag(Strict):
    """Whether a line means something other than, usually the opposite of, what its words say."""

    sarcastic: bool
    probability: Probability | None = None


class LineTags03(LineTags02):
    sarcasm: SarcasmTag | None = None
