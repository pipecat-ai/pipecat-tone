"""Noisy twins of r2's pool lines from a speech round trip: each drawn line spoken by Kokoro-82M and transcribed again by
Whisper, half through a telephone channel, so the student reads real speech-to-text errors beside the clean line. Pure
functions, no torch, no network, no volume: ``lab.reader.pipeline.speech`` feeds them lines and transcripts."""

import copy
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import TypeAdapter, ValidationError

from lab.reader.open_sources import modality_of, role_of
from lab.reader.schemas import LineId, LineInput
from lab.reader.sources import SPEECH_ID

SALT = "tonetext-speech-0.1"
MIN_CHARS, MAX_CHARS = 3, 200
MAX_WER = 0.6
MAX_PER_TEXT = 2  # lines per normalised text in the whole draw
TWIN_SUFFIX = "-asr"
TWIN_SLICE = "speech_twin"
LINE_ID = TypeAdapter(LineId)

# The models, pinned to a revision at their primary source, where their licences were checked too.
TTS = {
    "name": "Kokoro-82M",
    "repo": "hexgrad/Kokoro-82M",
    "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "weights": "kokoro-v1_0.pth",
    "licence": "Apache-2.0",
    "licence_source": "huggingface.co/hexgrad/Kokoro-82M at the pinned revision: README.md front matter `license: apache-2.0` "
    "and 'With Apache-licensed weights, Kokoro can be deployed anywhere'; trained on 'permissive/non-copyrighted "
    "audio data'; github.com/hexgrad/kokoro (the inference code, pip kokoro 0.9.4) Apache-2.0",
}
ASR = {
    "name": "Whisper large-v3-turbo",
    "repo": "openai/whisper-large-v3-turbo",
    "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "weights": "model.safetensors",
    "licence": "MIT",
    "licence_source": "huggingface.co/openai/whisper-large-v3-turbo at the pinned revision: README.md front matter `license: mit`; "
    "github.com/openai/whisper README: 'Whisper's code and model weights are released under the MIT License.'",
}
GENERATOR = f"tts {TTS['repo']}@{TTS['revision'][:7]} + asr {ASR['repo']}@{ASR['revision'][:7]}"

# Kokoro's English voices: US and UK, female and male, three each, its better-graded ones first (VOICES.md grades in
# brackets). The pipeline language: "a" American, "b" British (misaki's British G2P).
VOICES = (
    # US female (A, A-, C+)
    ("af_heart", "a"),
    ("af_bella", "a"),
    ("af_sarah", "a"),
    # US male (C+, C+, C+)
    ("am_michael", "a"),
    ("am_fenrir", "a"),
    ("am_puck", "a"),
    # UK female (B-, C, D)
    ("bf_emma", "b"),
    ("bf_isabella", "b"),
    ("bf_alice", "b"),
    # UK male (C, C, D+)
    ("bm_george", "b"),
    ("bm_fable", "b"),
    ("bm_lewis", "b"),
)

# The draw. Open sources: lines per source; two-party sources take USER_SHARE of them from the user's turns, the rest
# from the agent's (either side tops up the other when short). AMI's turns are a meeting's participants'.
OPEN_QUOTAS = {
    "natcs": 11_000,
    "taskmaster1": 6_000,
    "taskmaster2": 6_000,
    "taskmaster3": 5_000,
    "abcd": 8_000,
    "multiwoz22": 6_000,
    "star": 4_000,
    "ccpem": 3_000,
    "harpervalley": 5_000,
    "ami": 6_000,
}
USER_SHARE = 2 / 3
POOL_QUOTAS = {"player": 36_000, "npc": 4_000}  # the line classifier's pool, by who speaks
N_SAMPLES, SAMPLES_PER_SOURCE = 50, 5  # short audio clips kept for listening
SAMPLE_CHARS = (15, 90)


def h(salt: str, line_id: str) -> int:
    """A fixed hash of a line id under a salt: every job draws, voices and lowercases the same lines."""
    return int(hashlib.sha256(f"{SALT}:{salt}:{line_id}".encode()).hexdigest()[:16], 16)


def voice_for(line_id: str) -> tuple[str, str]:
    return VOICES[h("voice", line_id) % len(VOICES)]


def lowercased(line_id: str) -> bool:
    """Whether a twin is written as raw speech-to-text output (lowercase, no punctuation but the apostrophe)."""
    return h("lower", line_id) % 2 == 0


def channel_for(line_id: str) -> str:
    """What Whisper hears: half the lines as spoken (16 kHz, as a WebRTC call), half through a telephone channel (8 kHz,
    8-bit mu-law, as a phone call), so both of Pipecat's kinds of call carry their own errors."""
    return "phone" if h("channel", line_id) % 2 == 0 else "wideband"


_DOUBLE_QUOTES, _SINGLE_OPEN, _SINGLE_CLOSE = '"“”', "'‘", "'’"


def unwrap_quotes(text: str | None) -> tuple[str, bool]:
    """Whisper sometimes writes a whole utterance as a quotation ("'...'" or "..."), which no speech-to-text service
    does: the double quotes at either end go, then single quotes that open and close the whole text. Returns the text
    (spaces collapsed) and whether quotation marks were removed."""
    t = " ".join((text or "").split())
    u = t.strip(_DOUBLE_QUOTES).strip()
    if len(u) >= 2 and u[0] in _SINGLE_OPEN and u[-1] in _SINGLE_CLOSE:
        u = u[1:-1].strip()
    return u, u != t


_PUNCT = re.compile(r"[^\w\s']|_")


def norm(text: str | None) -> str:
    """Lowercase words and straight apostrophes, one space apart: the draw's text cap, WER and lowercased twins."""
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    return " ".join(_PUNCT.sub(" ", t).split())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance: substitutions, deletions and insertions."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, w in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != w))
        prev = cur
    return prev[-1]


def wer_counts(ref_text: str, hyp_text: str) -> tuple[int, int]:
    """(edits, reference words) after normalising both sides."""
    ref, hyp = norm(ref_text).split(), norm(hyp_text).split()
    return edit_distance(ref, hyp), len(ref)


def wer(ref_text: str, hyp_text: str) -> float:
    """Word error rate of a transcript against the clean text (normalised both). An empty reference: 0 if the
    transcript is empty too, else 1."""
    edits, n = wer_counts(ref_text, hyp_text)
    if n == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / n


def has_digit(text: str | None) -> bool:
    return any(c.isdigit() for c in text or "")


def speakable(text: str | None) -> bool:
    t = text or ""
    return MIN_CHARS <= len(t) <= MAX_CHARS and any(c.isalpha() for c in t)


# --- the draw ---------------------------------------------------------------------------------------------------------


def kind_of(ln: dict) -> str:
    """Who speaks, for the counts: the open lines' role (user, agent, participant), the pool lines' speaker kind."""
    return role_of(ln) if ln.get("source") in OPEN_QUOTAS else ln["input"]["speaker_kind"]


def group_of(ln: dict) -> str:
    return "open" if ln.get("source") in OPEN_QUOTAS else "pool"


def _ranked(lines: Iterable[dict], salt: str) -> list[dict]:
    return sorted(lines, key=lambda ln: h(salt, ln["line_id"]))


class _TextCap:
    """At most MAX_PER_TEXT lines per normalised text over the whole draw."""

    def __init__(self):
        self.seen: Counter = Counter()
        self.skipped = 0

    def take(self, ranked, n: int) -> list[dict]:
        """Up to `n` lines from an iterator, in its order; the iterator stops just after the last line taken, so a
        second call goes on from there."""
        out: list[dict] = []
        if n <= 0:
            return out
        for ln in ranked:
            k = norm(ln["input"]["text"])
            if self.seen[k] >= MAX_PER_TEXT:
                self.skipped += 1
                continue
            self.seen[k] += 1
            out.append(ln)
            if len(out) >= n:
                break
        return out


def _clean_train(ln: dict, notes: Counter) -> bool:
    if ln.get("twin_of"):
        notes["twin (not drawn)"] += 1
        return False
    if ln.get("split") != "train":
        notes["not train"] += 1
        return False
    if not speakable(ln["input"]["text"]):
        notes["outside 3 to 200 characters or no letter"] += 1
        return False
    return True


def draw(
    open_lines: Iterable[dict],
    pool_lines: Iterable[dict],
    open_quotas: dict[str, int] | None = None,
    pool_quotas: dict[str, int] | None = None,
    user_share: float = USER_SHARE,
) -> tuple[list[dict], dict]:
    """The fixed draw of clean train lines, by hashes of their ids: r2's open draw first, source by source in
    OPEN_QUOTAS order (the two-party sources' user and agent turns, AMI's participants'), then the line classifier's pool by
    who speaks. Texts of 3 to 200 characters with a letter; a machine transcript is never spoken again; a normalised
    text at most twice in the whole draw, so the common short replies do not crowd out the rest. Returns the lines and
    the counts (eligible and drawn per stratum, what was passed over and why)."""
    open_quotas = OPEN_QUOTAS if open_quotas is None else open_quotas
    pool_quotas = POOL_QUOTAS if pool_quotas is None else pool_quotas
    notes: Counter = Counter()
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ln in open_lines:
        src = ln.get("source")
        if src not in open_quotas:
            notes[f"open source not drawn: {src}"] += 1
            continue
        if modality_of(ln) == "asr":
            notes["machine transcript (not spoken again)"] += 1
            continue
        if not _clean_train(ln, notes):
            continue
        role = role_of(ln)
        if (src == "ami") != (role == "participant"):
            notes[f"role not drawn: {src}/{role}"] += 1
            continue
        strata[(src, role)].append(ln)
    for ln in pool_lines:
        if not _clean_train(ln, notes):
            continue
        kind = ln["input"]["speaker_kind"]
        if kind not in pool_quotas:
            notes[f"pool speaker kind not drawn: {kind}"] += 1
            continue
        strata[("pool", kind)].append(ln)

    cap, out = _TextCap(), []
    drawn: Counter = Counter()
    for src, quota in open_quotas.items():
        if src == "ami":
            got = cap.take(iter(_ranked(strata[(src, "participant")], "draw")), quota)
        else:
            users, agents = (
                iter(_ranked(strata[(src, "user")], "draw")),
                iter(_ranked(strata[(src, "agent")], "draw")),
            )
            got = cap.take(users, round(quota * user_share))
            got += cap.take(agents, quota - len(got))  # the user's side short: the agent's tops it up
            got += cap.take(users, quota - len(got))  # the agent's side short: the user's tops it up
        out += got
        for ln in got:
            drawn[f"{src}/{role_of(ln)}"] += 1
    for kind, quota in pool_quotas.items():
        got = cap.take(iter(_ranked(strata[("pool", kind)], "draw")), quota)
        out += got
        for ln in got:
            drawn[f"pool/{ln['source']}/{kind}"] += 1
    counts = {
        "lines": len(out),
        "eligible": {f"{a}/{b}": len(v) for (a, b), v in sorted(strata.items())},
        "drawn": dict(sorted(drawn.items())),
        "by_source": dict(Counter(ln["source"] for ln in out).most_common()),
        "by_group_and_kind": dict(sorted(Counter(f"{group_of(ln)}/{kind_of(ln)}" for ln in out).items())),
        "passed_over": dict(notes.most_common()),
        "repeated_text_skipped": cap.skipped,
        "quotas": {
            "open": open_quotas,
            "user_share": round(user_share, 4),
            "pool": pool_quotas,
            "max_per_normalised_text": MAX_PER_TEXT,
            "chars": [MIN_CHARS, MAX_CHARS],
        },
    }
    return out, counts


def pick_samples(lines: list[dict], n: int = N_SAMPLES, per_source: int = SAMPLES_PER_SOURCE) -> set[str]:
    """The line ids whose audio is kept for listening: short lines, a few per source, by hash."""
    lo, hi = SAMPLE_CHARS
    picked, per = [], Counter()
    for ln in _ranked((x for x in lines if lo <= len(x["input"]["text"]) <= hi), "sample"):
        if len(picked) >= n:
            break
        if per[ln["source"]] >= per_source:
            continue
        per[ln["source"]] += 1
        picked.append(ln["line_id"])
    return set(picked)


def task(ln: dict, sample: bool = False) -> dict:
    """What a speaking worker needs for one line."""
    voice, lang = voice_for(ln["line_id"])
    return {
        "line_id": ln["line_id"],
        "text": ln["input"]["text"],
        "voice": voice,
        "lang": lang,
        "bucket": h("shard", ln["line_id"]) % 1000,
        "sample": sample,
    }


# --- twins ------------------------------------------------------------------------------------------------------------

OUTCOMES = (
    "kept",
    "empty",
    "same_as_clean",
    "wer_over_0.6",
    "invalid_input",
    "no_audio",
    "tts_error",
    "no_result",
)


def make_twin(clean: dict, transcript: str | None) -> tuple[dict | None, str, float | None]:
    """(the twin or None, the outcome, the word error rate or None when there is no transcript). A twin is its clean
    line with line_id "<clean id>-asr", twin_of the clean id, slices plus speech_twin, GENERATOR, and the transcript as
    its text: Whisper's quotation marks around the whole of it removed, and for half the lines (a hash of the id)
    lowercased without punctuation but the apostrophe, as raw speech-to-text output often is. Dropped: an empty
    transcript, one equal to the clean text once normalised (it teaches nothing), one whose word error rate is over 0.6,
    and an input LineInput refuses."""
    hyp, _ = unwrap_quotes(transcript)
    if not norm(hyp):
        return None, "empty", None
    ref = clean["input"]["text"]
    w = wer(ref, hyp)
    if norm(hyp) == norm(ref):
        return None, "same_as_clean", w
    if w > MAX_WER:
        return None, "wer_over_0.6", w
    twin = copy.deepcopy(clean)
    twin["line_id"] = f"{clean['line_id']}{TWIN_SUFFIX}"
    twin["twin_of"] = clean["line_id"]
    twin["input"]["text"] = norm(hyp) if lowercased(clean["line_id"]) else hyp
    slices = list(clean.get("slices") or [])
    twin["slices"] = slices if TWIN_SLICE in slices else [*slices, TWIN_SLICE]
    twin["generator"] = GENERATOR
    try:
        LINE_ID.validate_python(twin["line_id"])
        LineInput.model_validate(twin["input"])
    except ValidationError:
        return None, "invalid_input", w
    return twin, "kept", w


def outcome_of(clean: dict, result: dict | None) -> tuple[dict | None, str, float | None]:
    """A worker's result for one drawn line ({line_id, transcript, error, seconds}) to its twin and outcome."""
    if result is None:
        return None, "no_result", None
    if result.get("error"):
        return None, "tts_error", None
    if not result.get("seconds"):
        return None, "no_audio", None
    return make_twin(clean, result.get("transcript"))


def heard(result: dict | None) -> str:
    """The transcript as the twin reads it, before lowercasing (quotation marks around the whole of it removed)."""
    return unwrap_quotes((result or {}).get("transcript"))[0]


# --- the counts ---------------------------------------------------------------------------------------------------------

WER_BINS = ((0.0, "0"), (0.1, "0-0.1"), (0.2, "0.1-0.2"), (0.3, "0.2-0.3"), (0.6, "0.3-0.6"), (1.0, "0.6-1"))


def wer_bin(w: float) -> str:
    if w == 0:
        return "0"
    for edge, name in WER_BINS[1:]:
        if w <= edge:
            return name
    return "over 1"


def distribution(rows: list[tuple[float, int, int]]) -> dict:
    """rows: (wer, edits, reference words) per line. Corpus WER is edits over reference words, summed."""
    if not rows:
        return {"lines": 0}
    ws = sorted(r[0] for r in rows)
    q = lambda p: round(ws[min(len(ws) - 1, int(p * len(ws)))], 4)  # noqa: E731
    bins = Counter(wer_bin(w) for w in ws)
    words = sum(r[2] for r in rows)
    return {
        "lines": len(rows),
        "corpus_wer": round(sum(r[1] for r in rows) / words, 4) if words else None,
        "mean": round(statistics.fmean(ws), 4),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.9),
        "p95": q(0.95),
        "bins": {name: bins.get(name, 0) for name in ["0", *[n for _, n in WER_BINS[1:]], "over 1"]},
    }


def summarize(drawn: list[dict], results: dict[str, dict]) -> tuple[list[dict], dict, list[dict]]:
    """Twins, the summary (counts only) and one detail row per drawn line (no text) from the draw and the results."""
    twins, details = [], []
    outcome: Counter = Counter()
    by: dict[str, dict[str, Counter]] = {
        "source": defaultdict(Counter),
        "kind": defaultdict(Counter),
        "channel": defaultdict(Counter),
    }
    wers: dict[str, list] = defaultdict(list)
    voices: Counter = Counter()
    seconds = []
    lower_kept = unwrapped = numerals = 0
    for ln in drawn:
        lid = ln["line_id"]
        res = results.get(lid)
        twin, out, w = outcome_of(ln, res)
        keys = {
            "source": ln["source"],
            "kind": f"{group_of(ln)}/{kind_of(ln)}",
            "channel": (res or {}).get("channel") or channel_for(lid),
        }
        outcome[out] += 1
        for k, v in keys.items():
            by[k][v][out] += 1
        if res and res.get("seconds"):
            seconds.append(float(res["seconds"]))
        if w is not None:
            unwrapped += unwrap_quotes(res.get("transcript"))[1]
            edits, n = wer_counts(ln["input"]["text"], heard(res))
            row = (w, edits, n)
            wers["all"].append(row)
            for k, v in keys.items():
                wers[f"{k}/{v}"].append(row)
            if out == "kept":
                wers["kept"].append(row)
                wers[f"kept_source/{keys['source']}"].append(row)
            elif out == "wer_over_0.6":
                numerals += has_digit(heard(res)) != has_digit(ln["input"]["text"])
        if twin is not None:
            twins.append(twin)
            voices[(res or {}).get("voice") or voice_for(lid)[0]] += 1
            lower_kept += lowercased(lid)
        details.append(
            {
                "line_id": lid,
                "source": keys["source"],
                "kind": keys["kind"],
                "voice": (res or {}).get("voice"),
                "channel": keys["channel"],
                "seconds": (res or {}).get("seconds"),
                "wer": None if w is None else round(w, 4),
                "outcome": out,
                "lowercased": lowercased(lid) if out == "kept" else None,
            }
        )
    kept = outcome["kept"]

    def counts(c: Counter) -> dict:
        return {
            "drawn": sum(c.values()),
            "kept": c["kept"],
            **{o: c[o] for o in OUTCOMES if o != "kept" and c[o]},
        }

    def dists(prefix: str) -> dict:
        return {
            k.split("/", 1)[1]: distribution(v) for k, v in sorted(wers.items()) if k.startswith(prefix + "/")
        }

    summary = {
        "speech_id": SPEECH_ID,
        "lines_drawn": len(drawn),
        "results": sum(1 for ln in drawn if ln["line_id"] in results),
        "twins_kept": kept,
        "dropped": {k: outcome[k] for k in OUTCOMES if k != "kept" and outcome[k]},
        "by_source": {k: counts(c) for k, c in sorted(by["source"].items())},
        "by_group_and_kind": {k: counts(c) for k, c in sorted(by["kind"].items())},
        "by_channel": {k: counts(c) for k, c in sorted(by["channel"].items())},
        "wer": {
            "note": "word error rate of the transcript against the clean text, both normalised (lowercase, punctuation but the "
            "apostrophe removed); over every line with a transcript, and over the kept twins",
            "all": distribution(wers["all"]),
            "kept": distribution(wers["kept"]),
            "by_source": dists("source"),
            "kept_by_source": dists("kept_source"),
            "by_group_and_kind": dists("kind"),
            "by_channel": dists("channel"),
        },
        "lowercased": {"twins": lower_kept, "share": round(lower_kept / kept, 4) if kept else None},
        "quotation_marks_removed": unwrapped,
        "wer_over_0.6_numerals_one_side": {
            "lines": numerals,
            "note": "drops over 0.6 with digits on one side only: mostly numbers, "
            "codes and dates Whisper writes as numerals where the clean line spells them (or the "
            "reverse), a formatting gap more than a mishearing; the plain word error rate counts it",
        },
        "voices_kept": dict(sorted(voices.items())),
        "audio_seconds": {
            "total": round(sum(seconds), 1),
            "mean": round(statistics.fmean(seconds), 2) if seconds else None,
            "max": round(max(seconds), 2) if seconds else None,
        },
        "generator": GENERATOR,
        "max_wer": MAX_WER,
    }
    return twins, summary, details
