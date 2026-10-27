"""Openly licensed conversation sets as corpus lines for r2's pool and its voice-agent dev and test sets, one per turn in
Tone's corpus-line shape plus `native`, the source's own label. Pure functions: ``lab.reader.pipeline.open_ingest``
downloads each source and feeds its files to the parsers here."""

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from lab.reader.schemas import MAX_PREVIOUS, MAX_TEXT, PLAYER, LineId, LineInput

CORPUS_VERSION = "tonetext-open-0.1"
AGENT = "Agent"
# Neutral names for meeting participants, in the order of their ids.
NAMES = (
    "Alex",
    "Blair",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jules",
    "Kai",
    "Lane",
    "Marlow",
)
TEST_ONLY = {"topicalchat_test"}

Source = Literal[
    "natcs",
    "taskmaster1",
    "taskmaster2",
    "taskmaster3",
    "abcd",
    "multiwoz22",
    "star",
    "ccpem",
    "harpervalley",
    "ami",
    "dipco",
    "dynasent_r2",
    "isarcasmeval",
    "topicalchat_test",
]
Role = Literal["user", "agent", "participant"]
Modality = Literal["spoken", "written", "asr"]
ROLES: tuple[str, ...] = Role.__args__
MODALITIES: tuple[str, ...] = Modality.__args__


class OpenLine(BaseModel):
    """Tone's corpus line for an open source, with the source's own label."""

    model_config = ConfigDict(extra="forbid")

    line_id: LineId
    source: Source
    source_ref: str
    licence: str
    split: Literal["train", "dev", "test"]
    group_key: str
    input: LineInput
    slices: list[str]
    twin_of: None = None
    generator: None = None
    native: dict | None = None


def role_of(line: dict) -> str:
    """An open line's role, the second of its slices ([source, role, modality, domains...])."""
    slices = line.get("slices") or []
    return slices[1] if len(slices) > 1 else ""


def modality_of(line: dict) -> str:
    """An open line's modality, the third of its slices."""
    slices = line.get("slices") or []
    return slices[2] if len(slices) > 2 else ""


@dataclass
class Turn:
    speaker: str  # `player`, `Agent`, or a participant's neutral name
    text: str
    role: Role
    modality: Modality
    ref: str  # where the turn is in the source: file, conversation, the source's own turn id
    native: dict | None = None


@dataclass
class Conversation:
    conv_id: str  # the source's own id, as given
    turns: list[Turn]
    participants: tuple[str, ...] = ()  # meetings only: every participant's neutral name
    domains: tuple[str, ...] = ()  # the conversation's domains, where the source gives them
    variant: str = ""  # a second reading of the same conversation (Harper Valley's machine transcript)
    no_previous: bool = False  # single sentences: no turn has a line before


@dataclass
class Built:
    lines: list[OpenLine] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)  # what the parsers skipped or cleaned, not turns dropped
    groups: dict = field(default_factory=dict)  # split: group keys with at least one line
    turns_seen: int = 0
    seen: set = field(default_factory=set)  # (text, previous) pairs kept so far
    conv_ids: Counter = field(default_factory=Counter)  # (conversation id, variant) seen so far

    @property
    def conversations(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.groups.items())}


def split_of(source: str, conv_id: str) -> str:
    """About 5% of a source's conversations dev and 5% test, fixed by a hash of the conversation's id; a test-only
    source (share-alike) never trains."""
    if source in TEST_ONLY:
        return "test"
    bucket = int(hashlib.sha256(f"{source}:{conv_id}".encode()).hexdigest()[:8], 16) % 100
    return "dev" if bucket < 5 else "test" if bucket < 10 else "train"


def safe_id(text: str) -> str:
    """Lower case, every character outside [a-z0-9_.-] made a hyphen, no leading punctuation: safe in a file name."""
    t = re.sub(r"[^a-z0-9_.-]+", "-", text.lower())
    return re.sub(r"-{2,}", "-", t).lstrip("-_.")


# Literal "\r", "\n", "\t" left in the text by an export (TM-3 has thousands).
ESCAPES = re.compile(r"\\+[rnt]")
ESCAPED_QUOTE = re.compile(r"\\+(?=['\"])")  # "I\'ve" (DynaSent, TM-3): the backslash goes


def norm(text: str | None) -> str:
    """Whitespace runs made one space and the ends trimmed: a kept line's text, and what texts are matched by."""
    return re.sub(r"\s+", " ", text or "").strip()


def domain_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def native(source: str, kind: str, label) -> dict | None:
    if label is None or label == [] or label == "":
        return None
    return {"scheme": f"{source}-{kind}", "label": label}


def build(source: str, convs, licence: str, built: Built | None = None) -> Built:
    """Corpus lines from conversations, one per turn, in order. A two-party turn is said to the other party; a meeting
    participant's to nobody in particular, every other participant a target. `previous` is the turn before as "Name:
    text", cut to its last 480 characters, and a turn dropped for its length still stands as the next one's. Dropped
    and counted: empty turns, turns over 400 characters, inputs LineInput refuses, and a (text, previous) pair already
    kept from the source, so one Built is passed through every call for a source."""
    out = built or Built()
    for conv in convs:
        out.conv_ids[(conv.conv_id, conv.variant)] += 1
        copy = out.conv_ids[(conv.conv_id, conv.variant)]
        # A source that reuses an id (versions of one dialog): the same group and split, its own line ids.
        if copy > 1:
            out.notes["repeated conversation id"] += 1
            conv.variant = f"{conv.variant}r{copy}"
        group_key = f"{source}:{conv.conv_id}"
        split = split_of(source, conv.conv_id)
        prev: tuple[str, str] | None = None
        two_party = not conv.participants
        for i, t in enumerate(conv.turns):
            out.turns_seen += 1
            raw, n = ESCAPES.subn(" ", t.text or "")
            raw, m = ESCAPED_QUOTE.subn("", raw)
            out.notes["literal escape sequences removed"] += n + m
            text = norm(raw)
            if not text:
                out.dropped["empty"] += 1
                continue
            previous = f"{prev[0]}: {prev[1]}"[-MAX_PREVIOUS:] if prev and not conv.no_previous else None
            prev = (t.speaker, text)
            if len(text) > MAX_TEXT:
                out.dropped["over_400_chars"] += 1
                continue
            if two_party:
                other = AGENT if t.speaker == PLAYER else PLAYER
                to, targets = other, [other]
            else:
                to, targets = None, [p for p in conv.participants if p != t.speaker]
            try:
                inp = LineInput.model_validate(
                    {
                        "speaker": t.speaker,
                        "speaker_kind": "player" if t.speaker == PLAYER else "npc",
                        "to": to,
                        "targets": targets,
                        "text": text,
                        "previous": previous,
                    }
                )
            except ValidationError:
                out.dropped["invalid_input"] += 1
                continue
            if (text, previous) in out.seen:
                out.dropped["duplicate_text_and_previous"] += 1
                continue
            out.seen.add((text, previous))
            tag = f"{conv.variant}-" if conv.variant else ""
            out.lines.append(
                OpenLine(
                    line_id=safe_id(f"{source}-{conv.conv_id}-{tag}{i}"),
                    source=source,
                    source_ref=t.ref,
                    licence=licence,
                    split=split,
                    group_key=group_key,
                    input=inp,
                    slices=[
                        source,
                        t.role,
                        t.modality,
                        *dict.fromkeys(d for d in map(domain_slug, conv.domains) if d),
                    ],
                    native=t.native,
                )
            )
            out.groups.setdefault(split, set()).add(group_key)
    return out


# --- the parsers: archive files (path inside the archive: bytes) to conversations, one per source ---------------------


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]


def csv_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


STANDARD_TAG = re.compile(r"[\[<][a-z ]{1,24}[\]>]")  # [noise], [laughter], <unk>, [clears throat]


def strip_tags(text: str | None, notes: Counter, pattern: re.Pattern) -> str:
    """Remove transcription tags ([noise], [laugh], ...) and count them. A standard tag is counted by name; anything
    else in brackets (a background speaker's words) only by number, so no source text reaches the counts."""

    def gone(m: re.Match) -> str:
        tag = m.group(0).lower()
        notes[f"tag removed {tag}" if STANDARD_TAG.fullmatch(tag) else "other bracketed text removed"] += 1
        return " "

    return pattern.sub(gone, text or "")


def _two_party(
    speaker_value: str, user: tuple[str, ...], agent: tuple[str, ...], notes: Counter
) -> tuple[str, str] | None:
    """The user's side speaks as `player`, the agent's as `Agent`; an unknown speaker is skipped and counted."""
    if speaker_value in user:
        return PLAYER, "user"
    if speaker_value in agent:
        return AGENT, "agent"
    notes[f"unknown speaker {speaker_value!r} skipped"] += 1
    return None


# NatCS (DSTC11 track 2): insurance is the development set, banking and finance the test sets. Banking and finance
# were spoken and then transcribed, insurance written as if spoken. Insurance
# marks fillers with "#" ("#Ah"); the mark goes, the word stays. Intents are gold; the InformIntent dialogue acts are
# the organisers' predictions (dstc11/faq.md). Sound tags in brackets and "~" cut-off marks are removed.
NATCS_FILES = {
    "insurance": "dstc11/development/dialogues.jsonl",
    "banking": "dstc11/test-banking/dialogues.jsonl",
    "finance": "dstc11/test-finance/dialogues.jsonl",
}
NATCS_MODALITY = {"insurance": "written", "banking": "spoken", "finance": "spoken"}
FILLER_MARK = re.compile(r"(?<![\w#])#(?=[A-Za-z])")
SOUND_TAG = re.compile(r"\[[^\]]*\]")  # [Laughter], [clears throat], and a few background speakers' words


def strip_tilde(text: str, notes: Counter) -> str:
    """NatCS and Harper Valley mark a cut-off word with "~" ("gon~", "wou~"): the mark goes, the fragment stays, as AMI
    writes them."""
    notes["cut-off marks (~) removed"] += text.count("~")
    return text.replace("~", "")


def natcs(members: dict[str, bytes], notes: Counter):
    for domain, path in NATCS_FILES.items():
        for row in jsonl(members[path]):
            did = str(row["dialogue_id"])
            conv_id = did if did.startswith(f"{domain}_") else f"{domain}_{did}"
            turns = []
            for t in row["turns"]:
                who = _two_party(t.get("speaker_role"), ("Customer",), ("Agent",), notes)
                if not who:
                    continue
                text, n = FILLER_MARK.subn("", t.get("utterance") or "")
                notes["filler marks removed"] += n
                text = strip_tilde(strip_tags(text, notes, SOUND_TAG), notes)
                intents, acts = t.get("intents") or [], t.get("dialogue_acts") or []
                nat = (
                    native("natcs", "intent", intents[0] if len(intents) == 1 else intents)
                    if intents
                    else native("natcs", "predicted-dialogue-act", sorted(acts))
                )
                turns.append(
                    Turn(who[0], text, who[1], NATCS_MODALITY[domain], f"{path}#{t.get('turn_id')}", nat)
                )
            yield Conversation(conv_id, turns, domains=(domain,))


# Taskmaster: USER and ASSISTANT (TM-3 lower case). TM-1's spoken dialogs (woz) and TM-2: the user's turns transcribed
# from speech, the assistant's typed; TM-1's self-dialogs and TM-3 written by one person for both sides (their READMEs).
def _taskmaster(path: str, convs: list[dict], user_mod: str, agent_mod: str, domain_of, notes: Counter):
    for c in convs:
        cid = c["conversation_id"]
        turns = []
        for u in sorted(c.get("utterances") or [], key=lambda u: u.get("index", 0)):
            who = _two_party((u.get("speaker") or "").upper(), ("USER",), ("ASSISTANT",), notes)
            if who:
                turns.append(
                    Turn(
                        who[0],
                        u.get("text") or "",
                        who[1],
                        user_mod if who[0] == PLAYER else agent_mod,
                        f"{path}#{cid}/{u.get('index')}",
                    )
                )
        dom = domain_of(c)
        yield Conversation(cid, turns, domains=(dom,) if dom else ())


def _tm1_domain(c: dict) -> str:
    return re.sub(r"-\d+[a-z]?$", "", c.get("instruction_id") or "")


def taskmaster1(members: dict[str, bytes], notes: Counter):
    yield from _taskmaster(
        "TM-1-2019/woz-dialogs.json",
        json.loads(members["TM-1-2019/woz-dialogs.json"]),
        "spoken",
        "written",
        _tm1_domain,
        notes,
    )
    yield from _taskmaster(
        "TM-1-2019/self-dialogs.json",
        json.loads(members["TM-1-2019/self-dialogs.json"]),
        "written",
        "written",
        _tm1_domain,
        notes,
    )


def taskmaster2(members: dict[str, bytes], notes: Counter):
    for path in sorted(n for n in members if re.fullmatch(r"TM-2-2020/data/[^/]+\.json", n)):
        domain = path.rsplit("/", 1)[1][:-5]
        yield from _taskmaster(
            path, json.loads(members[path]), "spoken", "written", lambda c, d=domain: d, notes
        )


def taskmaster3(members: dict[str, bytes], notes: Counter):
    for path in sorted(n for n in members if re.fullmatch(r"TM-3-2020/data/[^/]+\.json", n)):
        yield from _taskmaster(
            path,
            json.loads(members[path]),
            "written",
            "written",
            lambda c: c.get("vertical") or "movie tickets",
            notes,
        )


# ABCD: `original` holds the raw turns as [speaker, text]; "action" rows are the agent's system actions, not turns.
def abcd(members: dict[str, bytes], notes: Counter):
    data = json.loads(gzip.decompress(members["data/abcd_v1.1.json.gz"]))
    for part in ("train", "dev", "test"):
        for c in data[part]:
            cid = str(c["convo_id"])
            turns = []
            for i, (speaker, text) in enumerate(c["original"]):
                if speaker == "action":
                    notes["action rows skipped"] += 1
                    continue
                who = _two_party(speaker, ("customer",), ("agent",), notes)
                if who:
                    turns.append(Turn(who[0], text, who[1], "written", f"abcd_v1.1.json#{part}/{cid}/{i}"))
            yield Conversation(cid, turns, domains=((c.get("scenario") or {}).get("flow") or "",))


# MultiWOZ 2.2: USER and SYSTEM; the dialogue acts (act types without slots) from dialog_acts.json.
def multiwoz22(members: dict[str, bytes], notes: Counter):
    acts = json.loads(members["data/MultiWOZ_2.2/dialog_acts.json"])
    for path in sorted(
        n for n in members if re.fullmatch(r"data/MultiWOZ_2\.2/(train|dev|test)/dialogues_\d+\.json", n)
    ):
        for d in json.loads(members[path]):
            did = d["dialogue_id"]
            turns = []
            for t in d["turns"]:
                who = _two_party(t.get("speaker"), ("USER",), ("SYSTEM",), notes)
                if who:
                    labels = sorted(
                        ((acts.get(did) or {}).get(str(t.get("turn_id"))) or {}).get("dialog_act", {})
                    )
                    turns.append(
                        Turn(
                            who[0],
                            t.get("utterance") or "",
                            who[1],
                            "written",
                            f"{path}#{did}/{t.get('turn_id')}",
                            native("multiwoz22", "dialogue-act", labels),
                        )
                    )
            yield Conversation(did, turns, domains=tuple(d.get("services") or ()))


# STAR: the user's `utter` events, and what the wizard sent: `utter` (typed) and `pick_suggestion` (a suggested reply,
# with its action label). `request_suggestions` is the wizard's unsent draft; guide, query and knowledge-base events
# are not turns.
def star(members: dict[str, bytes], notes: Counter):
    paths = sorted(
        (n for n in members if re.fullmatch(r"dialogues/\d+\.json", n)), key=lambda n: int(n[10:-5])
    )
    for path in paths:
        d = json.loads(members[path])
        turns = []
        for i, e in enumerate(d.get("Events") or []):
            agent, action = e.get("Agent"), e.get("Action")
            if agent == "User" and action == "utter":
                turns.append(Turn(PLAYER, e.get("Text") or "", "user", "written", f"{path}#{i}"))
            elif agent == "Wizard" and action in ("utter", "pick_suggestion"):
                turns.append(
                    Turn(
                        AGENT,
                        e.get("Text") or "",
                        "agent",
                        "written",
                        f"{path}#{i}",
                        native("star", "wizard-action", e.get("ActionLabel")),
                    )
                )
            else:
                notes[f"events skipped {agent}/{action}"] += 1
        domains = tuple(x for x in ((d.get("Scenario") or {}).get("Domains") or []) if x)
        yield Conversation(str(d["DialogueID"]), turns, domains=domains)


# CCPE-M: the user spoke (transcribed), the assistant typed (README). Native: the turn's annotation types.
def ccpem(members: dict[str, bytes], notes: Counter):
    for c in json.loads(members["data.json"]):
        cid = c["conversationId"]
        turns = []
        for u in sorted(c.get("utterances") or [], key=lambda u: u.get("index", 0)):
            who = _two_party(u.get("speaker"), ("USER",), ("ASSISTANT",), notes)
            if who:
                types = sorted(
                    {
                        a.get("annotationType")
                        for s in u.get("segments") or []
                        for a in s.get("annotations") or []
                        if a.get("annotationType")
                    }
                )
                turns.append(
                    Turn(
                        who[0],
                        u.get("text") or "",
                        who[1],
                        "spoken" if who[0] == PLAYER else "written",
                        f"data.json#{cid}/{u.get('index')}",
                        native("ccpem", "annotation-types", types),
                    )
                )
        yield Conversation(cid, turns, domains=("movies",))


# Harper Valley Bank: each call's segments in time order, read twice: the transcriptionists' corrected text (spoken)
# and the machine transcript (asr), both in the call's group so they share its split. Dialogue acts are Gridspace's
# model outputs, not people's. Tags such as [noise] are removed.
HV_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")


def harpervalley(members: dict[str, bytes], notes: Counter):
    for path in sorted(n for n in members if re.fullmatch(r"data/transcript/[^/]+\.json", n)):
        sid = path.rsplit("/", 1)[1][:-5]
        meta = (
            json.loads(members[f"data/metadata/{sid}.json"]) if f"data/metadata/{sid}.json" in members else {}
        )
        tasks = tuple(sorted({t.get("task_type") for t in meta.get("tasks") or [] if t.get("task_type")}))
        segs = sorted(json.loads(members[path]), key=lambda s: (s.get("start_ms") or 0, s.get("index") or 0))
        human, machine = [], []
        for s in segs:
            who = _two_party(s.get("speaker_role"), ("caller",), ("agent",), notes)
            if not who:
                continue
            nat = native("harpervalley", "model-dialogue-act", sorted(s.get("dialog_acts") or []))
            ref = f"{path}#{s.get('index')}"
            human.append(
                Turn(
                    who[0],
                    strip_tilde(strip_tags(s.get("human_transcript"), notes, HV_TAG), notes),
                    who[1],
                    "spoken",
                    ref,
                    nat,
                )
            )
            machine.append(
                Turn(who[0], strip_tags(s.get("transcript"), notes, HV_TAG), who[1], "asr", ref + "/asr", nat)
            )
        yield Conversation(sid, human, domains=tasks)
        yield Conversation(sid, machine, domains=tasks, variant="asr")


# AMI (NXT manual annotations): every dialogue act of a meeting's four speakers, in time order (each act's first timed
# word), its words joined with their punctuation; vocal sounds, disfluency markers and gaps are not words. Speakers
# A to D become Alex, Blair, Casey and Drew for the meeting.
NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


SPELLED = re.compile(r"\b((?:[A-Za-z]_)+)")  # AMI spells letters out with underscores: "L_C_D_" is "LCD"


def join_words(words: list[tuple[str, bool]]) -> str:
    out = ""
    for w, punc in words:
        w = (w or "").strip()
        if w:
            out = out + w if punc and out else f"{out} {w}" if out else w
    return out


def unspell(text: str, notes: Counter) -> str:
    def join(m: re.Match) -> str:
        notes["spelled-out letters joined (L_C_D_ to LCD)"] += 1
        return m.group(1).replace("_", "")

    return SPELLED.sub(join, text)


def ami(members: dict[str, bytes], notes: Counter):
    import xml.etree.ElementTree as ET

    types = {
        e.get(f"{NITE}id"): e.get("name")
        for e in ET.fromstring(members["ontologies/da-types.xml"]).iter("da-type")
    }
    kinds = {
        m.get("observation"): m.get("type")
        for m in ET.fromstring(members["corpusResources/meetings.xml"]).iter("meeting")
    }
    files: dict[str, list[str]] = {}
    for n in sorted(members):
        m = re.fullmatch(r"dialogueActs/([^/.]+)\.([A-Z])\.dialog-act\.xml", n)
        if m:
            files.setdefault(m.group(1), []).append(n)
    for meeting, paths in sorted(files.items()):
        agents = sorted(p.split(".")[1] for p in paths)
        names = {a: NAMES[i] for i, a in enumerate(agents)}
        segs = []
        for path in paths:
            agent = path.split(".")[1]
            elems = list(ET.fromstring(members[f"words/{meeting}.{agent}.words.xml"]))
            index = {e.get(f"{NITE}id"): i for i, e in enumerate(elems)}
            last = 0.0
            for k, dact in enumerate(ET.fromstring(members[path]).iter("dact")):
                ptr = dact.find(f"{NITE}pointer")
                tid = re.search(r"id\(([^)]+)\)", ptr.get("href")).group(1) if ptr is not None else None
                words, times = [], []
                try:
                    for child in dact.findall(f"{NITE}child"):
                        m = _HREF.search(child.get("href") or "")
                        a, b = m.group(1), m.group(2) or m.group(1)
                        for e in elems[index[a] : index[b] + 1]:
                            if e.tag == "w":
                                words.append((e.text, e.get("punc") == "true"))
                                try:
                                    times.append(float(e.get("starttime")))
                                except (TypeError, ValueError):
                                    pass
                except (AttributeError, KeyError):
                    notes["dialogue acts with a broken word reference skipped"] += 1
                    continue
                if tid is None:
                    notes["dialogue acts without a type (kept, native null)"] += 1
                start = min(times) if times else last
                last = start
                segs.append(
                    (
                        start,
                        agent,
                        k,
                        unspell(join_words(words), notes),
                        types.get(tid) if tid else None,
                        f"{path}#{dact.get(f'{NITE}id')}",
                    )
                )
        segs.sort(key=lambda x: (x[0], x[1], x[2]))
        turns = [
            Turn(names[a], text, "participant", "spoken", ref, native("ami", "dialogue-act", label))
            for _, a, _, text, label, ref in segs
        ]
        yield Conversation(
            meeting,
            turns,
            participants=tuple(names[a] for a in agents),
            domains=(f"{kinds.get(meeting) or 'ami'} meeting",),
        )


# DiPCo: each session's utterances in the order they start (close-talk times); stored grouped by speaker. Participants
# get neutral names in the order of their ids. [noise], [laugh] and [unintelligible] are removed.
DIPCO_TAG = re.compile(r"\[[^\]]*\]")


def _seconds(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def dipco(members: dict[str, bytes], notes: Counter):
    paths = sorted(
        n for n in members if re.search(r"/transcriptions/(dev|eval)/S\d+\.json$", n) and "/._" not in n
    )
    for path in paths:
        utts = json.loads(members[path])
        sessions = {u.get("session_id") for u in utts}
        if len(sessions) != 1:
            raise ValueError(f"{path}: {len(sessions)} sessions in one file")
        speakers = sorted({u["speaker_id"] for u in utts})
        names = {sp: NAMES[i] for i, sp in enumerate(speakers)}
        order = sorted(
            range(len(utts)),
            key=lambda i: (_seconds(utts[i]["start_time"]["close-talk"]), utts[i]["speaker_id"], i),
        )
        turns = [
            Turn(
                names[utts[i]["speaker_id"]],
                strip_tags(utts[i].get("words"), notes, DIPCO_TAG),
                "participant",
                "spoken",
                f"{path}#{i}",
            )
            for i in order
        ]
        yield Conversation(
            sessions.pop(), turns, participants=tuple(names[sp] for sp in speakers), domains=("dinner party",)
        )


# DynaSent round 2 (crowd-written; round 1 is Yelp text and left out). Only the worker's sentence: never the Yelp
# prompt in `prompt_data`. Native: the majority label, null where the five workers had none.
def dynasent_r2(members: dict[str, bytes], notes: Counter):
    with zipfile.ZipFile(io.BytesIO(members["dynasent-v1.1.zip"])) as z:
        names = sorted(
            n
            for n in z.namelist()
            if re.fullmatch(r"dynasent-v1\.1/dynasent-v1\.1-round02-dynabench-(train|dev|test)\.jsonl", n)
        )
        for name in names:
            for row in jsonl(z.read(name)):
                tid = row["text_id"]
                if row.get("gold_label") is None:
                    notes["sentences without a majority label (kept, native null)"] += 1
                yield Conversation(
                    tid,
                    [
                        Turn(
                            PLAYER,
                            row.get("sentence") or "",
                            "user",
                            "written",
                            f"{name}#{tid}",
                            native("dynasent_r2", "sentiment", row.get("gold_label")),
                        )
                    ],
                    no_previous=True,
                )


# iSarcasmEval (English): each text is its own group, and an author's sincere rephrase joins its sarcastic original's
# group (turn 1, sarcastic 0), so the pair never straddles a split. In the test files the rephrases come in task C's
# pairs; where task A also lists the sarcastic text or its rephrase, that row joins the pair.
def isarcasmeval(members: dict[str, bytes], notes: Counter):
    import html

    def line(text, ref, label):
        clean = html.unescape(text or "")  # tweets keep Twitter's HTML escaping ("J&amp;J")
        notes["texts with HTML entities unescaped"] += clean != (text or "")
        return Turn(PLAYER, clean, "user", "written", ref, native("isarcasmeval", "sarcasm", label))

    for i, r in enumerate(csv_rows(members["train/train.En.csv"])):
        rid = (r.get("") or "").strip() or str(i)
        turns = [line(r.get("tweet"), f"train/train.En.csv#{rid}", int(r["sarcastic"]))]
        if r["sarcastic"] == "1" and (r.get("rephrase") or "").strip():
            turns.append(line(r["rephrase"], f"train/train.En.csv#{rid}/rephrase", 0))
        yield Conversation(f"train-{rid}", turns, no_previous=True)
    a_rows = csv_rows(members["test/task_A_En_test.csv"])
    a_by_text = {norm(r["text"]): i for i, r in enumerate(a_rows)}
    joined: set[int] = set()
    for j, r in enumerate(csv_rows(members["test/task_C_En_test.csv"])):
        s = int(r["sarcastic_id"])
        sarcastic, sincere = r[f"text_{s}"], r[f"text_{1 - s}"]
        ai, ri = a_by_text.get(norm(sarcastic)), a_by_text.get(norm(sincere))
        joined |= {x for x in (ai, ri) if x is not None}
        notes["task C pairs whose sarcastic text is in task A"] += ai is not None
        yield Conversation(
            f"test-a{ai}" if ai is not None else f"test-c{j}",
            [
                line(sarcastic, f"test/task_C_En_test.csv#{j}/text_{s}", 1),
                line(sincere, f"test/task_C_En_test.csv#{j}/text_{1 - s}", 0),
            ],
            no_previous=True,
        )
    for i, r in enumerate(a_rows):
        if i in joined:
            notes["task A rows joined to a task C pair"] += 1
            continue
        yield Conversation(
            f"test-a{i}",
            [line(r["text"], f"test/task_A_En_test.csv#{i}", int(r["sarcastic"]))],
            no_previous=True,
        )


# Topical-Chat, test splits only (share-alike: never trained on). Two crowd workers chatting as equals: agent_1 (who
# always opens) as `player`, agent_2 as `Agent`, both role "participant". Native: each writer's own sentiment label.
def topicalchat_test(members: dict[str, bytes], notes: Counter):
    for path in ("conversations/test_freq.json", "conversations/test_rare.json"):
        for cid, c in json.loads(members[path]).items():
            turns = []
            for i, t in enumerate(c.get("content") or []):
                who = _two_party(t.get("agent"), ("agent_1",), ("agent_2",), notes)
                if who:
                    turns.append(
                        Turn(
                            who[0],
                            t.get("message") or "",
                            "participant",
                            "written",
                            f"{path}#{cid}/{i}",
                            native("topicalchat_test", "sentiment", t.get("sentiment")),
                        )
                    )
            yield Conversation(cid, turns)


PARSERS = {
    "natcs": natcs,
    "taskmaster1": taskmaster1,
    "taskmaster2": taskmaster2,
    "taskmaster3": taskmaster3,
    "abcd": abcd,
    "multiwoz22": multiwoz22,
    "star": star,
    "ccpem": ccpem,
    "harpervalley": harpervalley,
    "ami": ami,
    "dipco": dipco,
    "dynasent_r2": dynasent_r2,
    "isarcasmeval": isarcasmeval,
    "topicalchat_test": topicalchat_test,
}


def parse(source: str, members: dict[str, bytes]) -> Built:
    """Every line of one source from its archive's files."""
    built = Built()
    licence = SOURCES[source]["licence"] + (" (test only: share-alike)" if source in TEST_ONLY else "")
    return build(source, PARSERS[source](members, built.notes), licence, built)


# Where each source is downloaded from (pinned GitHub commits and versions), which licence was verified for it, and
# how it is credited on the model card.
SOURCES: dict[str, dict] = {
    "natcs": {
        "kind": "github",
        "repo": "amazon-science/dstc11-track2-intent-induction",
        "commit": "969b95a0d7365fbc6cd0e05989f1be6b44e6680c",
        "licence": "CDLA-Permissive-2.0",
        "licence_files": ["dstc11/LICENSE.md"],
        "peek": ["dstc11/development/*", "dstc11/test-banking/*", "dstc11/test-finance/*"],
        "read": [
            "README.md",
            "LICENSE",
            "NOTICE",
            "dstc11/LICENSE.md",
            "dstc11/faq.md",
            "dstc11/task1-intent-clustering.md",
        ],
    },
    "taskmaster1": {
        "kind": "github",
        "repo": "google-research-datasets/Taskmaster",
        "commit": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "licence": "CC-BY-4.0",
        "licence_files": [],
        "peek": [
            "TM-1-2019/woz-dialogs.json",
            "TM-1-2019/self-dialogs.json",
            "TM-2-2020/data/flights.json",
            "TM-3-2020/data/data_00.json",
        ],
        "read": ["README.md", "TM-1-2019/README.md", "TM-2-2020/README.md", "TM-3-2020/README.md"],
    },
    "taskmaster2": {
        "kind": "github",
        "repo": "google-research-datasets/Taskmaster",
        "commit": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "licence": "CC-BY-4.0",
        "licence_files": [],
    },
    "taskmaster3": {
        "kind": "github",
        "repo": "google-research-datasets/Taskmaster",
        "commit": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "licence": "CC-BY-4.0",
        "licence_files": [],
    },
    "abcd": {
        "kind": "github",
        "repo": "asappresearch/abcd",
        "commit": "6b8700ce67c6b37b062dd7a60abc76d7ef832a97",
        "licence": "MIT",
        "licence_files": ["LICENSE"],
        "peek": ["data/abcd_v1.1.json.gz", "data/abcd_sample.json"],
        "read": ["README.md", "LICENSE"],
    },
    "multiwoz22": {
        "kind": "github",
        "repo": "budzianowski/multiwoz",
        "commit": "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b",
        "licence": "MIT",
        "licence_files": ["LICENSE"],
        "peek": ["data/MultiWOZ_2.2/train/dialogues_001.json", "data/MultiWOZ_2.2/dialog_acts.json"],
        "read": ["LICENSE", "data/MultiWOZ_2.2/README.md"],
    },
    "star": {
        "kind": "github",
        "repo": "RasaHQ/STAR",
        "commit": "3058975092cdd09798e4a3a9e4ae135755824545",
        "licence": "MIT",
        "licence_files": ["LICENSE.txt"],
        "peek": ["dialogues/1.json", "dialogues/2.json"],
        "read": ["README.md", "LICENSE.txt"],
    },
    "ccpem": {
        "kind": "github",
        "repo": "google-research-datasets/ccpe",
        "commit": "2c9cd30f33f3a154b5a27d015333679262ff36f5",
        "licence": "CC-BY-4.0",
        "licence_files": [],
        "peek": ["data.json"],
        "read": ["README.md"],
    },
    "harpervalley": {
        "kind": "github",
        "repo": "cricketclub/gridspace-stanford-harper-valley",
        "commit": "0bd721e877c4a85d8c13ff837e68661ea6200a98",
        "licence": "CC-BY-4.0",
        "licence_files": ["LICENSE"],
        "keep": ["data/transcript/*", "data/metadata/*", "LICENSE", "README.md"],  # no audio
        "peek": ["data/transcript/*", "data/metadata/*"],
        "read": ["README.md", "LICENSE"],
    },
    "ami": {
        "kind": "url",
        "urls": ["https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"],
        "licence_page": "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "licence": "CC-BY-4.0",
        "ua": "Wget/1.21.4",  # the corpus's own download page hands out wget scripts; other agents get 403 on GET
        "licence_files": ["*LICEN*"],
        "peek": [
            "*words/ES2002a.A.words.xml",
            "*dialogueActs/ES2002a.A.dialog-act.xml",
            "*ontologies/da-types.xml",
            "*corpusResources/meetings.xml",
        ],
        "read": ["*LICEN*", "*MANIFEST*", "*README*"],
    },
    "dipco": {
        "kind": "zenodo",
        "record": "8122551",
        "file": "DipCo.tgz",
        "licence": "CDLA-Permissive-1.0",
        "licence_files": ["*LICEN*"],
        # no audio
        "keep": ["*transcriptions/*", "*LICEN*", "*README*", "*readme*", "*.txt", "*.md", "*.pdf"],
        "peek": ["*transcriptions/*"],
        "read": ["*LICEN*", "*README*", "*readme*", "*.txt", "*.md"],
    },
    "dynasent_r2": {
        "kind": "github",
        "repo": "cgpotts/dynasent",
        "commit": "707b6208978bfe7ff251a0c3263c9f0552c27b48",
        "licence": "CC-BY-4.0",
        "licence_files": ["LICENSE", "dynasent-v1.1.zip!*LICEN*"],
        "peek": ["dynasent-v1.1.zip!*round02*", "dynasent-v1.1.zip!*"],
        "read": [
            "README.md",
            "LICENSE",
            "dynasent_datasheet.md",
            "dynasent-v1.1.zip!*README*",
            "dynasent-v1.1.zip!*LICEN*",
        ],
    },
    "isarcasmeval": {
        "kind": "github",
        "repo": "iabufarha/iSarcasmEval",
        "commit": "dfc708b53bde1bb571abfb5692f63231c2232195",
        "licence": "MIT",
        "licence_files": ["LICENSE"],
        "peek": ["train/train.En.csv", "test/task_A_En_test.csv", "test/task_C_En_test.csv"],
        "read": ["README.md", "LICENSE"],
    },
    "topicalchat_test": {
        "kind": "github",
        "repo": "alexa/Topical-Chat",
        "commit": "7c939229cbcf6f55f6977b341a5a2f2fe982d53f",
        "licence": "CDLA-Sharing-1.0",
        "licence_files": ["DATALICENSE"],
        "peek": ["conversations/test_freq.json", "conversations/test_rare.json"],
        "read": ["README.md", "DATALICENSE", "SOFTWARELICENSE"],
    },
}
# The archive files each parser reads; nothing else is unpacked.
FILES: dict[str, list[str]] = {
    "natcs": ["dstc11/*/dialogues.jsonl"],
    "taskmaster1": ["TM-1-2019/woz-dialogs.json", "TM-1-2019/self-dialogs.json"],
    "taskmaster2": ["TM-2-2020/data/*.json"],
    "taskmaster3": ["TM-3-2020/data/*.json"],
    "abcd": ["data/abcd_v1.1.json.gz"],
    "multiwoz22": ["data/MultiWOZ_2.2/*"],
    "star": ["dialogues/*.json"],
    "ccpem": ["data.json"],
    "harpervalley": ["data/transcript/*", "data/metadata/*"],
    "ami": ["words/*", "dialogueActs/*", "ontologies/da-types.xml", "corpusResources/meetings.xml"],
    "dipco": ["*/transcriptions/*"],
    "dynasent_r2": ["dynasent-v1.1.zip"],
    "isarcasmeval": ["train/train.En.csv", "test/task_A_En_test.csv", "test/task_C_En_test.csv"],
    "topicalchat_test": ["conversations/test_freq.json", "conversations/test_rare.json"],
}

# One line per source for the model card: licence and attribution travel with the lines.
CREDITS: dict[str, str] = {
    "natcs": "NatCS (Gung et al., 2023; Amazon, DSTC11 Track 2), CDLA-Permissive-2.0",
    "taskmaster1": "Taskmaster-1 (Byrne et al., 2019; Google), CC BY 4.0",
    "taskmaster2": "Taskmaster-2 (Byrne et al., 2020; Google), CC BY 4.0",
    "taskmaster3": "Taskmaster-3 (Byrne et al., 2020; Google), CC BY 4.0",
    "abcd": "ABCD, the Action-Based Conversations Dataset (Chen et al., 2021; ASAPP), MIT",
    "multiwoz22": "MultiWOZ 2.2 (Zang et al., 2020; Budzianowski et al., 2018), MIT",
    "star": "STAR (Mosig et al., 2020; Rasa), MIT",
    "ccpem": "CCPE-M (Radlinski et al., 2019; Google), CC BY 4.0",
    "harpervalley": "Harper Valley Bank (Wu et al., 2020; Gridspace and Stanford), CC BY 4.0",
    "ami": "AMI Meeting Corpus manual annotations (Carletta et al., 2005; the AMI Consortium), CC BY 4.0",
    "dipco": "DiPCo, the Dinner Party Corpus (Van Segbroeck et al., 2020; Amazon), CDLA-Permissive-1.0",
    "dynasent_r2": "DynaSent round 2 (Potts et al., 2021), CC BY 4.0",
    "isarcasmeval": "iSarcasmEval, SemEval-2022 Task 6 (Abu Farha et al., 2022), MIT",
    "topicalchat_test": "Topical-Chat (Gopalakrishnan et al., 2019; Amazon), CDLA-Sharing-1.0; its test splits only, for evaluation",
}

# What each source's own licence files said when it was downloaded. Every one matches the licence verified for it.
LICENCE_CHECKS: dict[str, str] = {
    "natcs": "dstc11/LICENSE.md is CDLA-Permissive-2.0 and NOTICE says the DSTC11 Track 2 data is under it; the repository LICENSE (Apache-2.0) is the code's",
    "taskmaster1": "no licence file; TM-1-2019/README.md's copyright notice: CC BY 4.0",
    "taskmaster2": "no licence file; TM-2-2020/README.md's copyright notice: CC BY 4.0",
    "taskmaster3": "no licence file; TM-3-2020/README.md's copyright notice: CC BY 4.0",
    "abcd": "LICENSE: MIT",
    "multiwoz22": "LICENSE: MIT",
    "star": "LICENSE.txt: MIT",
    "ccpem": "no licence file; README.md's copyright notice: CC BY 4.0",
    "harpervalley": "LICENSE: CC BY 4.0",
    "ami": "LICENCE.txt inside the zip and the corpus licence page: CC BY 4.0, corpus and annotations",
    "dipco": "Dipco/LICENSE.txt inside the archive: CDLA-Permissive-1.0; the Zenodo record says the same",
    "dynasent_r2": (
        "README.md (License) and dynasent_datasheet.md: the dataset is CC BY 4.0 and the code Apache 2.0, so the "
        "repository's LICENSE file (Apache-2.0) is the code's; no licence file inside dynasent-v1.1.zip"
    ),
    "isarcasmeval": "LICENSE: MIT",
    "topicalchat_test": (
        "DATALICENSE: CDLA-Sharing-1.0 except where a NOTICE.md says otherwise; the two NOTICE.md files cover "
        "the knowledge reading sets (src/wapo: Washington Post, CC BY-NC-ND 4.0; src/wiki: Wikipedia, CC BY-SA 3.0), "
        "not conversations/. Share-alike: test only"
    ),
}

# What a user of the lines should know (no line text).
CAVEATS: dict[str, str] = {
    "natcs": "insurance (948 dialogues) is written as if spoken, banking and finance spoken then transcribed; 117 insurance dialogues repeat others word for word under another id, so their turns are dropped as duplicates; the dialogue acts are the organisers' InformIntent predictions, the intents gold; '#' filler marks, '~' cut-off marks and bracketed sound tags removed",
    "taskmaster1": "woz-dialogs: the user's turns transcribed speech, the assistant's typed; self-dialogs: one worker wrote both sides; one woz id is used by six dialogs (kept in one group); 55 woz turns without a speaker skipped",
    "taskmaster2": "all spoken Wizard-of-Oz: the user's turns transcribed speech, the assistant's typed and played by text to speech; 15 ids appear twice in one domain file (9 identical, 6 versions), kept in one group",
    "taskmaster3": "self-dialogs: one worker wrote both sides; literal escape sequences (\\r, \\n, \\t) in the text replaced by spaces",
    "abcd": "the agent's action rows (system actions) are not turns and are left out",
    "multiwoz22": "native: the turn's dialogue act types from dialog_acts.json, without slots",
    "star": "the wizard's sent turns only (utter, pick_suggestion); drafts (request_suggestions), guide, query and knowledge-base events left out",
    "ccpem": "the user spoke (transcribed), the assistant typed; native: the turn's annotation types",
    "harpervalley": "each call read twice in one group: the corrected human transcript (spoken) and the machine transcript (asr); identical (text, previous) pairs kept once, so asr lines are where the machine differs or its line before does; dialogue acts are Gridspace's model outputs; tags ([noise], <unk>, ...) and '~' cut-off marks removed; audio not kept",
    "ami": "one line per dialogue act, in time order across the four speakers; vocal sounds, disfluency markers and gaps are not words; spelled-out letters joined (L_C_D_ to LCD); 28 acts in the corpus have no type (native null)",
    "dipco": "ten sessions, so the split by session puts whole sessions in one split; [noise], [laugh] and [unintelligible] removed; audio not kept",
    "dynasent_r2": "round 2 only; most sentences were written by workers shown a Yelp sentence as a prompt: only the worker's sentence is kept, never prompt_data; sentences without a majority label have native null; escaped apostrophes (I\\'ve) fixed",
    "isarcasmeval": "each author's sincere rephrase is its own line (sarcastic 0) in its sarcastic original's group; task C's test pairs are joined to task A's rows of the same texts; tweets keep their @-handles, links and emoji; HTML entities (&amp;) unescaped",
    "topicalchat_test": "test only (share-alike); two crowd workers chatting as equals (agent_1 as player, agent_2 as Agent, both 'participant'); turns can restate facts from the reading sets (Wikipedia, Washington Post, Reddit)",
}

# Sources that come from one download: the archive is kept once per source folder.
UNITS: dict[str, list[str]] = {}
for _s, _c in SOURCES.items():
    UNITS.setdefault(_c.get("repo") or _c.get("record") or _s, []).append(_s)


def unit_of(source: str) -> list[str]:
    """Every source that shares `source`'s download, the first one being where the download's description is read."""
    return next(v for v in UNITS.values() if source in v)


# --- counts (no text) and a reading sample ------------------------------------------------------------------------------

MARKUP = re.compile(r"[\[\]<>{}#*_|~\\]")


def _label(x) -> str:
    return "+".join(map(str, x)) if isinstance(x, list) else str(x)


def summarize(source: str, built: Built) -> dict:
    """Counts only: safe to keep in the repository (no line text)."""
    lines = built.lines
    words = sorted(len(ln.input.text.split()) for ln in lines) or [0]
    natives: dict[str, Counter] = {}
    for ln in lines:
        if ln.native:
            natives.setdefault(ln.native["scheme"], Counter())[_label(ln.native["label"])] += 1
    role_mod = Counter(f"{ln.slices[1]}/{ln.slices[2]}" for ln in lines)
    return {
        "source": source,
        "corpus_version": CORPUS_VERSION,
        "licence": SOURCES[source]["licence"],
        "test_only": source in TEST_ONLY,
        "credit": CREDITS[source],
        "lines": len(lines),
        "turns_read": built.turns_seen,
        "by_split": dict(sorted(Counter(ln.split for ln in lines).items())),
        "by_role": dict(sorted(Counter(ln.slices[1] for ln in lines).items())),
        "by_modality": dict(sorted(Counter(ln.slices[2] for ln in lines).items())),
        "by_role_and_modality": dict(sorted(role_mod.items())),
        "by_split_and_role": dict(sorted(Counter(f"{ln.split}/{ln.slices[1]}" for ln in lines).items())),
        "conversations": {"total": sum(built.conversations.values()), **built.conversations},
        "dropped": dict(sorted(built.dropped.items())),
        "notes": {k: v for k, v in sorted(built.notes.items()) if v},
        "domains": dict(Counter(d for ln in lines for d in ln.slices[3:]).most_common(60)),
        "native": {
            k: {"lines": sum(v.values()), "labels": len(v), "top": dict(v.most_common(40))}
            for k, v in sorted(natives.items())
        },
        "lines_without_native": sum(ln.native is None for ln in lines),
        "previous": {
            "none": sum(ln.input.previous is None for ln in lines),
            "cut_to_480": sum(len(ln.input.previous or "") == MAX_PREVIOUS for ln in lines),
        },
        "words": {
            "median": words[len(words) // 2],
            "p95": words[int(len(words) * 0.95)] if lines else 0,
            "max": words[-1],
            "one_to_three": sum(w <= 3 for w in words if lines),
        },
        "chars_max": max((len(ln.input.text) for ln in lines), default=0),
        "distinct_texts": len({ln.input.text for ln in lines}),
        "markup_characters": dict(
            Counter(ch for ln in lines for ch in set(MARKUP.findall(ln.input.text))).most_common()
        ),
    }


def sample(built: Built, seed: int = 512, runs: int = 3, run_len: int = 8, extra: int = 16) -> list[dict]:
    """Lines to read: a few runs of consecutive turns from random conversations (to check the line before), and some
    random lines. Printed in the session, never written to the repo."""
    import random

    rng = random.Random(seed)
    groups: dict[str, list[OpenLine]] = {}
    for ln in built.lines:
        groups.setdefault(ln.group_key, []).append(ln)
    keys = sorted(groups)
    picked: list[OpenLine] = []
    for key in rng.sample(keys, min(runs, len(keys))):
        g = groups[key]
        start = rng.randrange(max(1, len(g) - run_len + 1))
        picked += g[start : start + run_len]
    chosen = {ln.line_id for ln in picked}
    rest = [ln for ln in built.lines if ln.line_id not in chosen]
    # At least 40 to read.
    picked += rng.sample(rest, min(max(extra, runs * run_len + extra - len(picked)), len(rest)))
    return [
        {
            "line_id": ln.line_id,
            "split": ln.split,
            "speaker": ln.input.speaker,
            "to": ln.input.to,
            "targets": ln.input.targets,
            "slices": ln.slices,
            "previous": ln.input.previous,
            "text": ln.input.text,
            "native": ln.native,
            "source_ref": ln.source_ref,
        }
        for ln in picked
    ]
