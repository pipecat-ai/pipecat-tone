"""The reader's training and evaluation lines, one example per line with every committee and teacher label it has, and
the input format it reads them in. Pure Python, no torch."""

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from lab.reader import sources
from lab.reader.schemas import EMOTIONS, INTENSITIES, MOVE_INDEX
from pipecat_tone.reader.model import render  # noqa: F401

# The input format, the runtime's render: "oh great, thanks a lot" Before: "Sorry, we're out of rooms tonight."
RENDER_VERSION = "tonetext-input-0.1"
MAX_LEN = 160  # tokens


def previous_words(previous: str | None) -> str | None:
    """A label set's "Name: text" line before as its words only, so the input never carries names. A line without a
    name before its first colon passes through unchanged."""
    if not previous or not previous.strip():
        return None
    name, sep, rest = previous.partition(":")
    if sep and 0 < len(name.strip()) <= 40 and rest.strip():
        return rest.strip()
    return previous.strip()


EMOTION_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
INTENSITY_INDEX = {x: i for i, x in enumerate(INTENSITIES)}


@dataclass
class Example:
    """A line and every target it has, each of which can be missing: hard labels, the committee's majority, and soft
    labels, the teachers' mean."""

    line_id: str
    text: str
    previous: str | None  # the previous line's words, names cut
    split: str
    twin_of: str | None = None
    source: str = ""
    slices: tuple = ()
    sets: set = field(default_factory=set)
    emo: int = -100  # hard labels; -100 masked
    emo_int: int = -100  # masked for neutral, as in Tone
    back: int = -100
    move: int = -100
    move_w: float = 0.0
    sarc: int = -100
    sarc_w: float = 0.0
    s_emo: list | None = None  # soft labels; None absent
    s_emo_int: list | None = None
    s_back: float | None = None
    s_move: list | None = None
    s_sarc: float | None = None
    keep_previous: bool = False  # training never drops the line before (`LinesR2.keep_needed_previous`)

    @property
    def twin(self) -> bool:
        return self.twin_of is not None


def example(ln: dict, split: str) -> Example:
    """A corpus line as an example, without labels."""
    inp = ln["input"]
    return Example(
        ln["line_id"],
        inp["text"],
        previous_words(inp.get("previous")),
        split,
        ln.get("twin_of"),
        ln.get("source", ""),
        tuple(ln.get("slices") or ()),
    )


class Lines:
    """Examples by line id, built source by source. A line found in several sets is one example; a line the sets put in
    different splits is counted and kept out of every split."""

    def __init__(self):
        self.by_id: dict[str, Example] = {}
        self.split_clash: set[str] = set()
        self.counts: Counter = Counter()

    def get(self, ln: dict, split: str | None = None) -> Example:
        """The example for a corpus line, made on first sight. `split` overrides the line's own (the pool is train)."""
        lid = ln["line_id"]
        split = split or ln["split"]
        ex = self.by_id.get(lid)
        if ex is None:
            ex = example(ln, split)
            self.by_id[lid] = ex
        elif ex.split != split:
            self.split_clash.add(lid)
        return ex

    def add_tone_sets(self, jsonl: Callable[[str], list[dict]]) -> None:
        """Tone's committee labels: the act and emotion sets, the move set and the sarcasm sets."""
        for set_id in sources.ACT_EMOTION_SETS:
            self.add_act_emotion(set_id, jsonl(sources.labels_key(set_id)))
        self.add_exchange(jsonl(sources.labels_key(sources.EXCHANGE_SET)))
        for ids in sources.SARCASM_SETS.values():
            for set_id in ids:
                self.add_sarcasm(set_id, jsonl(sources.sarcasm_key(set_id)))

    def add_act_emotion(self, set_id: str, rows: Iterable[dict]) -> None:
        """The committee's emotion, its intensity (none for neutral) and backchannel."""
        for r in rows:
            ex = self.get(r["line"])
            if "act_emotion" in ex.sets:
                continue  # the first set to carry a line wins
            ex.sets.update({"act_emotion", set_id})
            emo = (r.get("emotion") or {}).get("label")
            inten = (r.get("emotion") or {}).get("intensity")
            ex.emo = EMOTION_INDEX[emo] if emo else -100
            ex.emo_int = INTENSITY_INDEX[inten] if emo and emo != "neutral" and inten else -100
            ex.back = {"true": 1, "false": 0}.get((r.get("backchannel") or {}).get("label"), -100)
            self.counts[f"hard_emotion_{ex.split}"] += ex.emo >= 0

    def add_exchange(self, rows: Iterable[dict]) -> None:
        """The committee's move, with its agreement as the line's weight."""
        for r in rows:
            ex = self.get(r["line"])
            ex.sets.add("exchange")
            lab = (r.get("exchange") or {}).get("label")
            ex.move = MOVE_INDEX[lab] if lab else -100
            ex.move_w = round(((r["exchange"].get("agree") or 3) / 3), 4) if lab else 0.0
            self.counts[f"hard_move_{ex.split}"] += ex.move >= 0

    def add_sarcasm(self, set_id: str, rows: Iterable[dict]) -> None:
        """The committee's sarcasm, with its agreement as the line's weight."""
        for r in rows:
            ex = self.get(r["line"])
            ex.sets.update({"sarcasm", set_id})
            lab = (r.get("sarcasm") or {}).get("label")
            ex.sarc = -100 if lab is None else int(bool(lab))
            ex.sarc_w = round(((r["sarcasm"].get("agree") or 3) / 3), 4) if lab is not None else 0.0
            self.counts[f"hard_sarcasm_{ex.split}"] += ex.sarc >= 0

    def add_pool(self, lines: Iterable[dict], soft: dict[str, dict]) -> None:
        """Tone's pool with its teachers' emotion, intensity and backchannel labels (twins take their clean line's)."""
        for ln in lines:
            s = soft.get(ln.get("twin_of") or ln["line_id"])
            if s is None:
                self.counts["pool_without_soft"] += 1
                continue
            ex = self.get(ln, split="train")
            ex.sets.add("pool")
            ex.s_emo, ex.s_emo_int, ex.s_back = list(s["emo"]), list(s["emo_int"]), float(s["back"])
            self.counts["soft_emotion"] += 1

    def add_move_soft(self, lines: Iterable[dict], means: dict[str, dict]) -> None:
        """The move teachers' mean (`mean_teachers`) on the move pool. Lines not yet seen join as training lines."""
        for ln in lines:
            mean = means.get(ln["line_id"])
            if mean is None:
                continue
            ex = self.get(ln, split="train")
            ex.sets.add("move_pool")
            ex.s_move = list(mean["p"])
        for ex in self.by_id.values():  # twins of move-pool lines
            if ex.s_move is None and ex.twin_of and ex.split == "train" and ex.twin_of in means:
                ex.s_move = list(means[ex.twin_of]["p"])
        self.counts["soft_move"] = sum(1 for e in self.by_id.values() if e.s_move is not None)

    def add_sarcasm_soft(self, means: dict[str, dict]) -> None:
        """The sarcasm teachers' mean (`mean_teachers`) on every training line without a committee sarcasm label (a
        noisy twin takes its clean line's)."""
        for ex in self.by_id.values():
            if ex.split != "train" or ex.sarc >= 0:
                continue
            mean = means.get(ex.line_id)
            if mean is None and ex.twin_of:
                mean = means.get(ex.twin_of)
            if mean is not None:
                ex.s_sarc = float(mean["p"])
        self.counts["soft_sarcasm"] = sum(1 for e in self.by_id.values() if e.s_sarc is not None)

    def split(self, name: str) -> list[Example]:
        return [e for e in self.by_id.values() if e.split == name and e.line_id not in self.split_clash]

    def summary(self) -> dict:
        tr = self.split("train")

        def has(e: Example) -> dict:
            return {
                "emotion": e.emo >= 0 or e.s_emo is not None,
                "move": e.move >= 0 or e.s_move is not None,
                "sarcasm": e.sarc >= 0 or e.s_sarc is not None,
            }

        cover = Counter(k for e in tr for k, v in has(e).items() if v)
        return {
            "lines": len(self.by_id),
            "train": len(tr),
            "dev": len(self.split("dev")),
            "test": len(self.split("test")),
            "split_clash": len(self.split_clash),
            "train_with": dict(cover),
            "train_with_previous": sum(1 for e in tr if e.previous),
            "counts": dict(self.counts),
        }


def mean_teachers(files: list[list[dict]], keys: tuple[str, ...]) -> dict[str, dict]:
    """Several teachers' files averaged on the lines every file has: by line id, each key's values summed in file order,
    then divided by the number of files."""
    if not files:
        return {}
    common = set.intersection(*({r["line_id"] for r in f} for f in files))
    sums: dict[str, dict] = {}
    for f in files:
        for r in f:
            if r["line_id"] not in common:
                continue
            acc = sums.setdefault(r["line_id"], {})
            for k in keys:
                v = r[k]
                if k not in acc:
                    acc[k] = v
                elif isinstance(v, list):
                    acc[k] = [a + b for a, b in zip(acc[k], v)]
                else:
                    acc[k] = acc[k] + v
    n = len(files)
    return {
        lid: {k: ([x / n for x in v] if isinstance(v, list) else v / n) for k, v in acc.items()}
        for lid, acc in sums.items()
    }


def build(
    jsonl: Callable[[str], list[dict]], sarcasm_teachers: tuple[str, ...] = (), pools: bool = True
) -> Lines:
    """Every line r1's student reads (`jsonl` reads one file): Tone's committee labels, then Tone's pool and the move
    pool with their teachers' labels, and the mean of the sarcasm teachers that `sarcasm_teachers` names. `pools=False`
    reads the committee's labelled sets only (dev and test for calibration and scoring)."""
    lines = Lines()
    lines.add_tone_sets(jsonl)
    if not pools:
        return lines
    soft = {r["line_id"]: r for r in jsonl(sources.soft_key())}
    lines.add_pool(jsonl(sources.pool_key()), soft)
    del soft
    move = mean_teachers([jsonl(sources.exchange_soft_key(t)) for t in sources.EXCHANGE_TEACHERS], ("p",))
    lines.add_move_soft(jsonl(sources.exchange_soft_key("lines")), move)
    if sarcasm_teachers:
        files = [jsonl(sources.sarcasm_soft_key(t)) for t in sarcasm_teachers]
        lines.add_sarcasm_soft(mean_teachers(files, ("p",)))
    return lines


def r1_training_lines(jsonl: Callable[[str], list[dict]], sarcasm: bool = True) -> Iterator[tuple[str, dict]]:
    """r1's training lines as (where from, line), noisy twins and repeats included: Tone's pool, the move pool, then the
    train splits of the act and emotion sets, the move set and, with `sarcasm`, the sarcasm set."""
    for ln in jsonl(sources.pool_key()):
        yield "pool", ln
    for ln in jsonl(sources.exchange_soft_key("lines")):
        yield "exchange_pool", ln
    labelled = [("act_emotion_train", sources.labels_key(s)) for s in sources.ACT_EMOTION_SETS]
    labelled.append(("exchange_train", sources.labels_key(sources.EXCHANGE_SET)))
    if sarcasm:
        labelled += [("sarcasm_train", sources.sarcasm_key(s)) for s in sources.SARCASM_SETS["train"]]
    for where, key in labelled:
        for r in jsonl(key):
            if r["line"]["split"] == "train":
                yield where, r["line"]


def clean_once(
    lines: Iterable[tuple[str, dict]], exclude_sources: tuple[str, ...] = ()
) -> Iterator[tuple[str, dict]]:
    """Each line the first time its id comes, without noisy twins (they take their clean line's labels) and without
    `exclude_sources`."""
    seen = set()
    for where, ln in lines:
        if ln.get("twin_of") or ln["line_id"] in seen or ln.get("source") in exclude_sources:
            continue
        seen.add(ln["line_id"])
        yield where, ln


# --- the reads used to choose a seed and to report it ------------------------------------------------------------------


def eval_groups(exs: list[Example], split: str) -> dict[str, list[int]]:
    """Indices into `exs` for each read: Tone's main test, player speech and unseen settings for emotion; the move's main
    test (no twins, no unseen settings); the held-out sarcasm and chain tests and the natural test for sarcasm. On dev:
    every clean labelled line per output."""

    def where(cond):
        return [i for i, e in enumerate(exs) if e.split == split and cond(e)]

    if split == "dev":
        return {
            "emotion": where(lambda e: e.emo >= 0 and not e.twin and "act_emotion" in e.sets),
            "move": where(lambda e: e.move >= 0 and not e.twin and "exchange" in e.sets),
            "sarcasm": where(lambda e: e.sarc >= 0 and "sarcasm" in e.sets),
        }
    emo = lambda e: e.emo >= 0 and not e.twin and "act_emotion" in e.sets  # noqa: E731
    return {
        "emotion_main": where(lambda e: emo(e) and not any(s in e.slices for s in sources.HELD_SLICES)),
        "emotion_player_speech": where(lambda e: emo(e) and "player_speech" in e.slices),
        "emotion_unseen": where(lambda e: emo(e) and "unseen_genre" in e.slices),
        "move": where(
            lambda e: e.move >= 0 and not e.twin and "exchange" in e.sets and "unseen_genre" not in e.slices
        ),
        "sarcasm_hard": where(lambda e: e.sarc >= 0 and bool(e.sets & set(sources.HARD_SARCASM_TEST))),
        "sarcasm_chain": where(lambda e: e.sarc >= 0 and sources.CHAIN_TEST in e.sets),
        "sarcasm_natural": where(lambda e: e.sarc >= 0 and sources.NATURAL_SARCASM[1] in e.sets),
    }


# --- r2 --------------------------------------------------------------------------------------------------------------

# In Example.sets: a line from r2's committee sets, where the voice-agent dev and test sets live.
R2_TAG = "r2"
# In Example.sets: a line from r3's support-call set, or from r3c's contrast pairs.
SUPPORT_TAG = "support"
CONTRAST_TAG = "contrast"
# The slices of lines written to be sarcastic only after the agent's line: the support set's short answers to news, and
# the contrast pairs' side after which the turn means the opposite.
NEEDS_PREVIOUS = ("reply_sarcastic_news", "reply_contrast_opposite")


def sha_order(lines: list[dict], salt: str) -> list[dict]:
    """Lines in a fixed order, by the sha256 of salt and line id, so every job draws the same lines."""
    return sorted(lines, key=lambda ln: hashlib.sha256(f"{salt}:{ln['line_id']}".encode()).hexdigest())


class LinesR2(Lines):
    def add_r2_committee(self, jsonl: Callable[[str], list[dict]]) -> None:
        """r2's committee sets: training turns, the voice-agent dev set and the real voice-agent test set."""
        for q, sid in sources.R2_SETS.items():
            rows = jsonl(sources.tonetext_labels_key(sid))
            before = set(self.by_id)
            if q == "committee":
                for r in rows:  # r2's labels stand even where a Tone set already carries the line
                    ex = self.get(r["line"])
                    ex.sets.discard("act_emotion")
                self.add_act_emotion(sid, rows)
            elif q == "exchange":
                self.add_exchange(rows)
            else:
                self.add_sarcasm(sid, rows)
            for r in rows:
                self.by_id[r["line"]["line_id"]].sets.add(R2_TAG)
            self.counts[f"r2_{q}_new_lines"] += len(set(self.by_id) - before)

    def add_support(
        self, jsonl: Callable[[str], list[dict]], set_ids: dict = sources.SUPPORT_SETS, tag: str = SUPPORT_TAG
    ) -> None:
        """r3's support-call set, or with r3c's ids and CONTRAST_TAG its contrast pairs: the committee's emotion,
        intensity and backchannel, and its sarcasm, asked under the Realtime Context Standard's perception. Its move
        is left to the teachers."""
        before = set(self.by_id)
        rows = {q: jsonl(sources.tonetext_labels_key(sid)) for q, sid in set_ids.items()}
        self.add_act_emotion(set_ids["committee"], rows["committee"])
        self.add_sarcasm(set_ids["sarcasm"], rows["sarcasm"])
        for rs in rows.values():
            for r in rs:
                self.by_id[r["line"]["line_id"]].sets.add(tag)
        self.counts[f"{tag}_new_lines"] = len(set(self.by_id) - before)

    def keep_needed_previous(self) -> int:
        """Marks the sarcastic lines whose sarcasm the agent's line before carries, as they were written
        (NEEDS_PREVIOUS), so training never drops that line: without it, their label would teach that the words alone
        are sarcastic. Returns how many."""
        n = 0
        for ex in self.by_id.values():
            if ex.sarc == 1 and ex.previous and any(s in ex.slices for s in NEEDS_PREVIOUS):
                ex.keep_previous = True
                n += 1
        self.counts["keep_previous"] = n
        return n

    def add_text(self, lines: Iterable[dict], tag: str) -> int:
        """Training lines with no label yet (their soft labels come from the teachers). Returns how many were new."""
        n = 0
        for ln in lines:
            new = ln["line_id"] not in self.by_id
            ex = self.get(ln, split="train")
            ex.sets.add(tag)
            n += new
        return n

    def add_multi_soft(self, means: dict[str, dict]) -> None:
        """The r2 teachers' mean (`mean_teachers`) on every training line, for each output the committee did not label
        (a noisy twin takes its clean line's)."""
        for ex in self.by_id.values():
            if ex.split != "train":
                continue
            key = ex.line_id if ex.line_id in means else (ex.twin_of if ex.twin_of in means else None)
            if key is None:
                continue
            p = means[key]
            if ex.emo < 0:
                ex.s_emo, ex.s_emo_int, ex.s_back = list(p["emo"]), list(p["emo_int"]), float(p["back"])
            if ex.move < 0:
                ex.s_move = list(p["move"])
            if ex.sarc < 0:
                ex.s_sarc = float(p["sarc"])

    def drop_sources(self, excluded: tuple[str, ...]) -> int:
        gone = [k for k, e in self.by_id.items() if e.source in excluded]
        for k in gone:
            del self.by_id[k]
        self.counts["dropped_by_source"] = len(gone)
        return len(gone)

    def drop_targetless(self) -> int:
        """Training lines with neither a committee label nor a teacher's: nothing to learn from."""
        gone = [
            k
            for k, e in self.by_id.items()
            if e.split == "train"
            and e.emo < 0
            and e.move < 0
            and e.sarc < 0
            and e.s_emo is None
            and e.s_move is None
            and e.s_sarc is None
        ]
        for k in gone:
            del self.by_id[k]
        self.counts["dropped_without_targets"] = len(gone)
        return len(gone)


def r2_open_lines(jsonl: Callable[[str], list[dict]]) -> list[dict]:
    """The open sets' train-split lines that reach the pool: each source's lines, or a fixed draw of them over its cap
    (`sources.OPEN_CAPS`)."""
    out = []
    for src in sources.OPEN_TRAIN_SOURCES:
        rows = [ln for ln in jsonl(sources.open_key(src)) if ln["split"] == "train"]
        cap = sources.OPEN_CAPS[src]
        out += rows if len(rows) <= cap else sha_order(rows, f"r2-{src}")[:cap]
    return out


def r2_soft_id(tag: str = "", recipe: str = "r2") -> str:
    """A recipe's teachers' labels' id: R2_SOFT_ID (R3_SOFT_ID for r3; r3c reads r2's), or its own for teachers
    built under a tag."""
    base = sources.R3_SOFT_ID if recipe == "r3" else sources.R2_SOFT_ID
    return base if not tag else base.replace("-0.1", f"-{tag}-0.1")


def build_r2(
    jsonl: Callable[[str], list[dict]],
    teachers: tuple[str, ...] = (),
    pools: bool = True,
    speech: bool = True,
    exclude_sources: tuple[str, ...] = (),
    soft_id: str = sources.R2_SOFT_ID,
    support: bool = False,
    contrast: bool = False,
) -> LinesR2:
    """Every line r2 reads, and with `support` r3's: Tone's committee sets, then r2's, then the support-call set's;
    with `contrast` (r3c) the contrast pairs too, and no line whose sarcasm the agent's line carries loses that line in
    training. With `pools`: Tone's pool and the move pool, the open sets' draws and, with `speech`, the speech round
    trip's twins, all soft-labelled by the teachers named in `teachers` (their files under `soft_id`); a training line
    left with no target is dropped. `exclude_sources` drops every line of those sources (such as "goemotions")."""
    lines = LinesR2()
    lines.add_tone_sets(jsonl)
    lines.add_r2_committee(jsonl)
    if support:
        lines.add_support(jsonl)
    if contrast:
        lines.add_support(jsonl, sources.CONTRAST_SETS, CONTRAST_TAG)
        lines.keep_needed_previous()
    if pools:
        lines.counts["pool_lines"] = lines.add_text(jsonl(sources.pool_key()), "pool")
        lines.counts["move_pool_lines"] = lines.add_text(
            jsonl(sources.exchange_soft_key("lines")), "move_pool"
        )
        lines.counts["open_lines"] = lines.add_text(r2_open_lines(jsonl), "open")
        if speech:
            lines.counts["speech_twins"] = lines.add_text(jsonl(sources.speech_key()), "speech")
        files = [jsonl(sources.r2_soft_key(t, soft_id)) for t in teachers]
        means = mean_teachers(files, ("emo", "emo_int", "back", "move", "sarc"))
        lines.add_multi_soft(means)
        lines.counts["teacher_labelled"] = len(means)
    if exclude_sources:
        lines.drop_sources(tuple(exclude_sources))
    if pools:
        lines.drop_targetless()
    return lines


def r2_lines_to_label(
    jsonl: Callable[[str], list[dict]], exclude_sources: tuple[str, ...] = (), support: bool = False
) -> list[dict]:
    """The clean training lines the r2 teachers label, each once: r1's (`r1_training_lines`), then the r2 committee
    sets' train splits and the open sets' draws; with `support`, r3's, the support-call set's train split last."""

    def walk() -> Iterator[tuple[str, dict]]:
        yield from r1_training_lines(jsonl)
        for sid in sources.R2_SETS.values():
            for r in jsonl(sources.tonetext_labels_key(sid)):
                if r["line"]["split"] == "train":
                    yield "r2_train", r["line"]
        for ln in r2_open_lines(jsonl):
            yield "open", ln
        if support:
            for sid in sources.SUPPORT_SETS.values():
                for r in jsonl(sources.tonetext_labels_key(sid)):
                    if r["line"]["split"] == "train":
                        yield "support_train", r["line"]

    return [ln for _, ln in clean_once(walk(), exclude_sources)]


# The support-call set's moments whose agent line carries no news (``support.MOMENTS``).
SUPPORT_PLAIN = ("moment_status", "moment_question", "moment_procedure", "moment_kind")


def eval_groups_r2(exs: list[Example], split: str) -> dict[str, list[int]]:
    """r1's reads on Tone's lines, plus the voice-agent reads on r2's committee lines (dev: the voice-agent dev set; test:
    the real voice-agent test set), and when the support-call set is read its own: emotion and sarcasm, on test also
    sarcasm after a plain agent line, after news, and on the held domains; the contrast pairs' alike. Neither set's
    lines join the others' reads, so r2's rule chooses an epoch as it did."""
    tone = [
        e for e in exs if R2_TAG not in e.sets and SUPPORT_TAG not in e.sets and CONTRAST_TAG not in e.sets
    ]
    base = eval_groups(tone, split)
    index = {id(e): i for i, e in enumerate(exs)}
    out = {k: [index[id(tone[i])] for i in v] for k, v in base.items()}

    def where(cond):
        return [i for i, e in enumerate(exs) if e.split == split and R2_TAG in e.sets and cond(e)]

    out["voice_emotion"] = where(lambda e: e.emo >= 0 and not e.twin)
    out["voice_move"] = where(lambda e: e.move >= 0 and not e.twin)
    out["voice_sarcasm"] = where(lambda e: e.sarc >= 0 and not e.twin)

    def support(cond):
        return [i for i, e in enumerate(exs) if e.split == split and SUPPORT_TAG in e.sets and cond(e)]

    plain = lambda e: any(s in e.slices for s in SUPPORT_PLAIN)  # noqa: E731
    groups = {
        "support_emotion": support(lambda e: e.emo >= 0),
        "support_sarcasm": support(lambda e: e.sarc >= 0),
    }
    if split == "test":
        groups["support_sarcasm_after_plain"] = support(lambda e: e.sarc >= 0 and plain(e))
        groups["support_sarcasm_after_news"] = support(
            lambda e: e.sarc >= 0 and not plain(e) and "reply_ordinary" not in e.slices
        )
        groups["support_sarcasm_held_domains"] = support(lambda e: e.sarc >= 0 and "held_domain" in e.slices)

    def contrast(cond):
        return [i for i, e in enumerate(exs) if e.split == split and CONTRAST_TAG in e.sets and cond(e)]

    groups["contrast_emotion"] = contrast(lambda e: e.emo >= 0)
    groups["contrast_sarcasm"] = contrast(lambda e: e.sarc >= 0)
    if split == "test":
        groups["contrast_sarcasm_held_domains"] = contrast(
            lambda e: e.sarc >= 0 and "held_domain" in e.slices
        )
    out.update({k: v for k, v in groups.items() if v})
    return out
