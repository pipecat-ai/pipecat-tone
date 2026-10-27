"""Audible breaths in real conversation: where the transcribers of the ICSI Meeting Corpus (CC BY 4.0) heard a speaker
breathe on a close-talking microphone, as statistics to set the tagger's breaths against. Its words are never
trained on.

    uv run python -m lab.corpus.speech_breaths
"""

import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from statistics import quantiles

from lab import ROOT
from pipecat_tone.tagger.model import is_word

URL = "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip"
SHA256 = "9db98ebdd61a085b4f45f24f46e9c9f6c6f26c3c9cd41643a3b9e74362c85de7"
CACHE = ROOT / "data" / "icsi" / "ICSI_original_transcripts.zip"
OUT = ROOT / "results" / "breaths" / "icsi.json"
# Inhalations: "outbreath" is an exhale, and "breath-laugh" mostly laughter.
BREATHS = {"breath", "inbreath", "long breath"}
# Replies that leave the floor with the speaker, as ICSI spells them.
BACKCHANNELS = set("yeah uh-huh mm-hmm mm mmm hmm huh right okay o_k oh yep uh um wow".split())
LEAD_S = 1.5  # how long before a turn's first word a breath still opens it
BREATH = "<breath>"  # a breath among a turn's tokens
# A sentence ends at its punctuation, or at a dash on its own, which ends a phrase given up or restarted.
END = re.compile(r"^-$|[.?!][\"')]*$")
DEEP_S = 0.6  # a breath marked on its own that lasts longer than this is a deep one
TURN_LENGTHS = ((1, 5), (6, 15), (16, 30), (31, 60), (61, None))
SENTENCE_LENGTHS = ((1, 5), (6, 10), (11, 20), (21, None))


@dataclass(frozen=True)
class Turn:
    """One speaker's turn: each sentence's word count and whether a breath comes before it, and where breaths fall
    inside its sentences, as the number of the turn's words before them. A speaker's turns share ``speaker``."""

    speaker: str
    sentences: tuple[tuple[int, bool], ...]
    inside: tuple[int, ...] = ()

    @property
    def words(self) -> int:
        return sum(n for n, _ in self.sentences)

    @property
    def breaths(self) -> int:
        return sum(b for _, b in self.sentences) + len(self.inside)


def fetch() -> zipfile.ZipFile:
    """The ICSI transcripts, downloaded once and checked."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "Wget/1.21.4"})
        CACHE.write_bytes(urllib.request.urlopen(request).read())
    if hashlib.sha256(CACHE.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"{CACHE} isn't the ICSI transcripts this study was made with")
    return zipfile.ZipFile(CACHE)


def meeting_turns(name: str, meeting: ET.Element) -> list[Turn]:
    """A meeting's turns, in order. A turn starts when someone says more than a backchannel while another holds the
    floor; their last breath, if it ended up to ``LEAD_S`` before, opens it. Listeners' breaths and backchannels are
    otherwise left out, as are digit reading and speech off a close microphone."""
    floor, heard, held = None, {}, []
    for seg in meeting.iter("Segment"):
        who = seg.get("Participant")
        if not who or seg.get("DigitTask") == "true" or seg.get("CloseMic") == "false":
            continue
        tokens = list(_tokens(seg))
        words = [t for t in tokens if t != BREATH and is_word(t)]
        start, end = float(seg.get("StartTime")), float(seg.get("EndTime"))
        if who != floor and any(w.lower().strip(".,?!-\"'") not in BACKCHANNELS for w in words):
            floor, lead = who, heard.pop(who, None)
            held.append((who, [BREATH] if lead is not None and start - lead <= LEAD_S else []))
        if who == floor:
            held[-1][1].extend(tokens)
        elif not words and BREATH in tokens:
            heard[who] = end
    return [turn for who, tokens in held if (turn := _turn(f"{name}/{who}", tokens)).sentences]


def icsi_turns() -> list[Turn]:
    with fetch() as z:
        names = sorted(n for n in z.namelist() if re.fullmatch(r"transcripts/B\w+\.mrt", n))
        return [t for n in names for t in meeting_turns(PurePosixPath(n).stem, ET.fromstring(z.read(n)))]


def breath_kinds() -> dict[str, float]:
    """How people's breaths divide into light and deep inhales and exhales: exhales from the transcribers' marks, and
    deep inhales as the inhales marked on their own, before the speaker's words within ``LEAD_S``, that last longer
    than ``DEEP_S``."""
    marks: Counter = Counter()
    lengths = []
    with fetch() as z:
        for name in sorted(n for n in z.namelist() if re.fullmatch(r"transcripts/B\w+\.mrt", n)):
            waiting: dict[str, tuple[float, float]] = {}
            for seg in ET.fromstring(z.read(name)).iter("Segment"):
                who = seg.get("Participant")
                if not who or seg.get("DigitTask") == "true" or seg.get("CloseMic") == "false":
                    continue
                marks.update(v.get("Description") for v in seg.iter("VocalSound"))
                tokens = list(_tokens(seg))
                start, end = float(seg.get("StartTime")), float(seg.get("EndTime"))
                if tokens and all(t == BREATH for t in tokens):
                    waiting[who] = (end, end - start)
                elif any(t != BREATH and is_word(t) for t in tokens) and who in waiting:
                    breath_end, length = waiting.pop(who)
                    if start - breath_end <= LEAD_S and 0.05 <= length <= 3.0:
                        lengths.append(length)
    exhale = marks["outbreath"] / (sum(marks[b] for b in BREATHS) + marks["outbreath"])
    deep = (1 - exhale) * sum(x > DEEP_S for x in lengths) / len(lengths)
    return {
        "light breath": round(1 - exhale - deep, 3),
        "deep breath": round(deep, 3),
        "exhale": round(exhale, 3),
    }


def summary(turns: list[Turn]) -> dict:
    """How often turns open with or hold a breath, by length; how often a breath comes before a sentence, or before or
    inside it, by its place and length; how far apart breaths fall; and whether a speaker's breathing turns follow one
    another."""
    sentences = [s for t in turns for s in _sentences(t)]
    before = sum(b for _, _, b, _ in sentences)
    inside = sum(len(t.inside) for t in turns)
    words = sum(t.words for t in turns)
    gaps = [g for t in turns for g in _spans(t)]
    speakers: dict[str, list[bool]] = {}
    for t in turns:
        speakers.setdefault(t.speaker, []).append(t.breaths > 0)
    pairs = [(a, b) for flags in speakers.values() for a, b in zip(flags, flags[1:])]
    # Were each speaker's turns shuffled, how often a breathing turn would follow a breathing one.
    chance = sum(sum(f) * (sum(f) - 1) / len(f) for f in speakers.values()) / max(
        sum(sum(f) * (len(f) - 1) / len(f) for f in speakers.values()), 1e-9
    )
    return {
        "turns": len(turns),
        "sentences": len(sentences),
        "words": words,
        "turns_with_breath": _share(t.breaths > 0 for t in turns),
        "turns_with_breath_before_a_sentence": _share(any(b for _, b in t.sentences) for t in turns),
        "turns_opening_with_breath": _share(t.sentences[0][1] for t in turns),
        "turns_with_two_or_more": _share(t.breaths > 1 for t in turns),
        "breaths_per_100_words": round(100 * (before + inside) / max(words, 1), 2),
        "breaths_inside_sentences": round(inside / max(before + inside, 1), 4),
        "by_turn_length": {
            band_name(*length): {
                "turns": len(group),
                "with_breath": _share(t.breaths > 0 for t in group),
                "before_a_sentence": _share(any(b for _, b in t.sentences) for t in group),
                "opening": _share(t.sentences[0][1] for t in group),
                "two_or_more": _share(t.breaths > 1 for t in group),
            }
            for k, length in enumerate(TURN_LENGTHS)
            if (group := [t for t in turns if band(t.words, TURN_LENGTHS) == k])
        },
        "sentences_by_length": {
            place: {
                band_name(*length): {
                    "sentences": len(group),
                    "breath_before": _share(b for b, _ in group),
                    "breath_before_or_inside": _share(e for _, e in group),
                }
                for k, length in enumerate(SENTENCE_LENGTHS)
                if (group := [(b, e) for i, n, b, e in sentences if (i == 0) == first and band(n) == k])
            }
            for place, first in (("first", True), ("later", False))
        },
        "words_between_breaths": _percentiles(gaps) if len(gaps) > 1 else None,
        "breath_after_breathing_turn": _share(b for a, b in pairs if a),
        "breath_after_breathing_turn_by_chance": round(chance, 4),
        "breath_after_plain_turn": _share(b for a, b in pairs if not a),
    }


def sentence_rate(stats: dict) -> Callable[[int, bool], float]:
    """A sentence's chance of a breath before or inside it, from a summary, by its words and whether it opens the
    turn. The tagger breathes only before a sentence, so a breath a speaker takes inside one goes before it."""
    table = stats["sentences_by_length"]

    def rate(words: int, first: bool) -> float:
        name = band_name(*SENTENCE_LENGTHS[band(words)])
        return table["first" if first else "later"][name]["breath_before_or_inside"]

    return rate


def band(words: int, lengths: tuple = SENTENCE_LENGTHS) -> int:
    """The length band ``words`` falls in; the first for none."""
    return next(k for k, (_, high) in enumerate(lengths) if high is None or words <= high)


def band_name(low: int, high: int | None) -> str:
    """A length band's name: "6-10", or "21+" when it has no top."""
    return f"{low}-{high}" if high else f"{low}+"


def _sentences(turn: Turn) -> Iterator[tuple[int, int, bool, bool]]:
    """Each sentence's place in the turn, its words, and whether a breath comes before it, or before or inside it."""
    at = 0
    for i, (n, b) in enumerate(turn.sentences):
        yield i, n, b, b or any(at < p < at + n for p in turn.inside)
        at += n


def _tokens(el: ET.Element) -> Iterator[str]:
    if el.text:
        yield from el.text.split()
    for child in el:
        if child.tag == "VocalSound":
            if child.get("Description") in BREATHS:
                yield BREATH
        elif child.tag not in ("NonVocalSound", "Comment", "Pause"):
            yield from _tokens(child)
        if child.tail:
            yield from child.tail.split()


def _turn(speaker: str, tokens: list[str]) -> Turn:
    """A turn from its tokens. A breath is placed by the word after it: inside the sentence if the word is in it,
    before it if the word starts one, and nowhere if no word follows."""
    sentences, inside, at, words, before, pending = [], [], 0, 0, False, False
    for t in tokens:
        if t == BREATH:
            pending = True
            continue
        if is_word(t):
            if pending and words:
                inside.append(at)
            before, pending = before or (pending and not words), False
            at, words = at + 1, words + 1
        if words and END.search(t):
            sentences.append((words, before))
            words, before = 0, False
    if words:
        sentences.append((words, before))
    return Turn(speaker, tuple(sentences), tuple(inside))


def _spans(turn: Turn) -> list[int]:
    """Words between one breath and the next within a turn."""
    starts = [sum(n for n, _ in turn.sentences[:i]) for i, (_, b) in enumerate(turn.sentences) if b]
    at = sorted([*starts, *turn.inside])
    return [b - a for a, b in zip(at, at[1:])]


def _percentiles(values: list[int]) -> dict[str, float]:
    cuts = quantiles(values, n=20, method="inclusive")
    return {"p10": cuts[1], "p25": cuts[4], "p50": cuts[9], "p75": cuts[14]}


def _share(flags: Iterable[bool]) -> float:
    flags = list(flags)
    return round(sum(flags) / max(len(flags), 1), 4)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "source": {"url": URL, "sha256": SHA256, "licence": "CC-BY-4.0"},
        **summary(icsi_turns()),
        "breath_kinds": breath_kinds(),
    }
    OUT.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
