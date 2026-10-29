"""The reader's data without others' text, and the files that hold it rebuilt from the original sources. What reads
the data finds every file at its key as before: the files that are ours as they are; the open sets rebuilt from their
archives; and eight files that mix our lines with others' published as stand-ins. A stand-in mirrors its file line for
line in the same JSON style, our own rows as they are and in place of each value from a third-party source a hole that
says how to make it again from the source at its pin. Pure Python: ``lab.reader.publish_data`` writes the stand-ins,
``lab.reader.pipeline.fetch`` rebuilds the files on the volume.

    uv run python -m lab.reader.rebuild check --copy <copy> --staged <dir> --upstream <dir>
    uv run python -m lab.reader.rebuild audit <dir> --copy <copy>
"""

import argparse
import copy
import difflib
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from lab.reader import open_sources, sources, speech
from lab.reader.sources import MANIFEST_KEY, standin_key
from lab.reader.tone_sources import noisy_base

# Each file's rows are json.dumps of themselves in one of these, and each file ends with a newline.
STYLES = {
    "compact_utf8": {"separators": (",", ":"), "ensure_ascii": False},
    "spaced_ascii": {},
}

# Tone's sources by who wrote the lines: ours, and the third-party ones with the holes their text and label take.
OURS = ("mix", "generated", "handwritten")
TEXT_HOLES = {"goemotions": "ge", "civil_comments": "cc", "gutenberg": "gb"}
LABEL_HOLES = {"goemotions": "gel", "civil_comments": "ccl"}
PREVIOUS_HOLE = "gbp"  # Gutenberg's line before
TWIN_HOLE = "v"  # a noisy twin's text, made from its clean line's
OPEN_ROW, ASR_ROW = "open", "asr"  # whole-row holes: an open set's line, a speech twin

# rc1's and rc2's runs as trained and exported: weights, results and the files the runtime loads.
RUN_ID = sources.run_id("r2", "student", "minilm-l6", 1)
RC2_RUN_ID = sources.run_id("r3c", "student", "minilm-l6", 2)
RUN_FILES = (
    "config.json",
    "model.pt",
    "result.json",
    "export/config.json",
    "export/export_result.json",
    "export/model.onnx",
    "export/model.opt.onnx",
    "export/tokenizer.json",
    "precision/model_fp16.onnx",
)
# Tone's label sets and the r2 sets that hold others' lines.
TONE_MIXED = ("linecls-corpus-0.1-full-committee", "linecls-corpus-0.1-thin-committee")
SARCASM_MIXED = sources.SARCASM_SETS["train"]
# rc2's support-call and contrast sets' calls as written: each line's turns before it and its sarcasm as written.
CALLS = tuple(sources.support_key("meta.jsonl", s) for s in (sources.SUPPORT_ID, sources.CONTRAST_ID))

# Published as they are: every file here is ours.
KEEP = (
    *(sources.run_key(RUN_ID, f) for f in RUN_FILES),
    *(sources.r2_soft_key(t) for t in sources.R2_TEACHERS),
    sources.exchange_soft_key("lines"),
    *(sources.labels_key(s) for s in sources.ACT_EMOTION_SETS if s not in TONE_MIXED),
    sources.labels_key(sources.EXCHANGE_SET),
    *(sources.sarcasm_key(s) for split in ("dev", "test") for s in sources.SARCASM_SETS[split]),
    *(sources.tonetext_labels_key(s) for s in sources.VOICE_GENERATED.values()),
    *(sources.run_key(RC2_RUN_ID, f) for f in RUN_FILES),
    *(
        sources.tonetext_labels_key(s)
        for sets in (sources.SUPPORT_SETS, sources.CONTRAST_SETS)
        for s in sets.values()
    ),
    *CALLS,
)
# Not published: each rebuilt from its source's archive (``lab.reader.pipeline.open_ingest``).
OPEN = {s: sources.open_key(s) for s in open_sources.SOURCES}
# Published as stand-ins, rebuilt in this order: the speech twins read the pool's lines and the open sets'.
STANDIN = {
    sources.pool_key(): "tone",
    **{sources.labels_key(s): "tone" for s in TONE_MIXED},
    **{sources.sarcasm_key(s): "tone" for s in SARCASM_MIXED},
    **{sources.tonetext_labels_key(s): "r2" for s in sources.R2_SETS.values()},
    sources.speech_key(): "speech",
}


class Mismatch(ValueError):
    """A row that can't be made again exactly, or a source that isn't the one a stand-in was made from."""


def dumps(obj, style: str) -> str:
    return json.dumps(obj, **STYLES[style])


def rows_of(body: bytes) -> list[str]:
    text = body.decode()
    if not text.endswith("\n"):
        raise Mismatch("the file doesn't end with a newline")
    return text[:-1].split("\n")


def style_of(body: bytes) -> str:
    """The style every row of a file is written in."""
    lines = rows_of(body)
    for style in STYLES:
        if all(dumps(json.loads(x), style) == x for x in lines):
            return style
    raise Mismatch("no one style writes every row")


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# --- holes -------------------------------------------------------------------------------------------------------------


def diff(base: str, target: str) -> list[list]:
    """The edits that make `target` from `base`: [start, end, insert], offsets in `base`'s code points, in order."""
    ops = difflib.SequenceMatcher(None, base, target, autojunk=False).get_opcodes()
    return [[i1, i2, target[j1:j2]] for op, i1, i2, j1, j2 in ops if op != "equal"]


def patch(base: str, edits: list) -> str:
    """`base` with `edits` made, the last first so the offsets of the rest hold."""
    out = base
    for start, end, insert in reversed(edits):
        out = out[:start] + insert + out[end:]
    return out


def fold(text: str) -> str:
    """Lower case, one code point for each, so offsets in it are offsets in the text."""
    low = text.lower()
    return low if len(low) == len(text) else "".join(c if len(c.lower()) != 1 else c.lower() for c in text)


def edits_from(base: str, target: str) -> dict:
    """A hole's edits: from the base, or from the base in lower case when that puts in fewer characters (speech-to-text
    can write a line in another case, and edits from the base would put in most of its letters again). {"f": 1}
    marks the lower case; neither key is there when there's nothing to say."""
    edits, low = diff(base, target), fold(base)
    if low != base:
        folded = diff(low, target)
        if sum(len(e[2]) for e in folded) < sum(len(e[2]) for e in edits):
            return {"f": 1, "e": folded}
    return {"e": edits} if edits else {}


def made(h: dict, base: str) -> str:
    """A hole's value from its base."""
    return patch(fold(base) if h.get("f") else base, h.get("e", []))


def tag(base: str) -> str:
    """What a hole keeps of its base: enough to tell the source's text has changed, not to read it."""
    return hashlib.sha256(base.encode()).hexdigest()[:12]


def hole(kind: str, base: str, value: str) -> dict:
    """A value from a source as the edits from its base, the source's own text. A label's hole keeps no tag: with so
    few labels, a tag would name one."""
    tagged = kind not in LABEL_HOLES.values()
    return {"$": kind, **edits_from(base, value), **({"h": tag(base)} if tagged else {})}


def is_hole(value) -> bool:
    return isinstance(value, dict) and ("$" in value or "$row" in value)


def fill(h: dict, base: str, where: str) -> str:
    if "h" in h and h["h"] != tag(base):
        raise Mismatch(f"{where}: the source's {h['$']} isn't the one the stand-in was made from")
    return made(h, base)


def twin_base(clean: dict) -> str:
    """What a speech twin's text is edited from: its clean line's text, as speech-to-text writes it for half the lines."""
    text = clean["input"]["text"]
    return speech.norm(text) if speech.lowercased(clean["line_id"]) else text


def speech_twin(clean: dict, h: dict) -> dict:
    """A speech twin from its clean line, as ``lab.reader.speech`` makes one, and its hole's edits to the text."""
    twin = copy.deepcopy(clean)
    twin["line_id"] = f"{clean['line_id']}{speech.TWIN_SUFFIX}"
    twin["twin_of"] = clean["line_id"]
    twin["input"]["text"] = made(h, twin_base(clean))
    slices = list(clean.get("slices") or [])
    twin["slices"] = slices if speech.TWIN_SLICE in slices else [*slices, speech.TWIN_SLICE]
    twin["generator"] = speech.GENERATOR
    return twin


# --- what a file's holes are filled from --------------------------------------------------------------------------------


class Context:
    """What holes are filled from: the sources at their pins (``tone_sources.Upstream``) and the files rebuilt before,
    read by key, of which only the rows asked for are kept."""

    def __init__(self, upstream=None, read: Callable[[str], bytes] | None = None):
        self.upstream = upstream
        self.read = read
        self._rows: dict[str, dict[str, dict]] = {}

    def rows(self, key: str, ids: set[str]) -> dict[str, dict]:
        have = self._rows.setdefault(key, {})
        if ids - have.keys():
            for x in rows_of(self.read(key)):
                r = json.loads(x)
                if r["line_id"] in ids:
                    have[r["line_id"]] = r
        missing = ids - have.keys()
        if missing:
            raise Mismatch(f"{key} has no line {sorted(missing)[0]} (and {len(missing) - 1} more)")
        return have


def needs(standin: bytes, kind: str) -> dict[str, set[str]]:
    """The rows of files rebuilt before that a stand-in's whole-row holes take, by key."""
    out: dict[str, set[str]] = {}
    if kind == "tone":
        return out
    for x in rows_of(standin):
        r = json.loads(x)
        h = r.get("line") if kind == "r2" else r
        if is_hole(h) and "$row" in h:
            key = sources.pool_key() if h["src"] == "pool" else sources.open_key(h["src"])
            out.setdefault(key, set()).add(h["id"] if h["$row"] == OPEN_ROW else h["of"])
    return out


# --- stand-ins and rebuilds ------------------------------------------------------------------------------------------


def _line(row: dict) -> dict:
    return row["line"] if "line" in row else row


def _tone_out(line: dict, up, clean: dict[str, str], counts: Counter) -> None:
    """A third-party line's text, line before and label made holes, in place."""
    src, ref, inp = line["source"], line["source_ref"], line["input"]
    where = line["line_id"]
    if line.get("twin_of"):
        if line["twin_of"] not in clean:
            raise Mismatch(f"{where}: its clean line isn't in the file")
        inp["text"] = hole(TWIN_HOLE, noisy_base(clean[line["twin_of"]]), inp["text"])
    else:
        inp["text"] = hole(TEXT_HOLES[src], up.text(src, ref), inp["text"])
    if inp["previous"] is not None:
        if src != "gutenberg":
            raise Mismatch(f"{where}: a {src} line with a line before")
        inp["previous"] = hole(PREVIOUS_HOLE, f"{inp['to']}: {up.previous(src, ref)}", inp["previous"])
    if src in LABEL_HOLES:
        label = up.label(src, ref)
        if label == line["generator"]:
            line["generator"] = hole(LABEL_HOLES[src], label, label)
        else:
            counts["label_kept"] += 1
    elif line["generator"] is not None:
        raise Mismatch(f"{where}: a {src} line with a generator")


def _tone_in(line: dict, up, clean: dict[str, str]) -> None:
    src, ref, inp = line["source"], line["source_ref"], line["input"]
    where = line["line_id"]
    if is_hole(inp["text"]):
        h = inp["text"]
        if h["$"] == TWIN_HOLE:
            if line["twin_of"] not in clean:
                raise Mismatch(f"{where}: its clean line isn't in the file")
            base = noisy_base(clean[line["twin_of"]])
        else:
            base = up.text(src, ref)
        inp["text"] = fill(h, base, where)
    if is_hole(inp["previous"]):
        inp["previous"] = fill(inp["previous"], f"{inp['to']}: {up.previous(src, ref)}", where)
    if is_hole(line["generator"]):
        line["generator"] = fill(line["generator"], up.label(src, ref), where)


# The sources each kind of file may hold beside ours.
ALLOWED = {
    "tone": tuple(TEXT_HOLES),
    "r2": tuple(open_sources.SOURCES),
    "speech": (*TEXT_HOLES, *open_sources.SOURCES),
}


def _source(line: dict, kind: str) -> str:
    src = line.get("source")
    if src not in OURS and src not in ALLOWED[kind]:
        raise Mismatch(
            f"{line.get('line_id')}: source {src!r} isn't ours or one this file's holes are made from"
        )
    return src


def _from(src: str) -> str:
    """The rebuilt file a third-party speech twin's or r2 row's line is read from."""
    return sources.open_key(src) if src in open_sources.SOURCES else sources.pool_key()


def stand_in_file(original: bytes, kind: str, ctx: Context) -> tuple[bytes, Counter, list]:
    """A file's stand-in, the counts of its holes and the sizes of the edited ones. Each row is made again from its
    stand-in row and checked against the original."""
    style = style_of(original)
    lines = rows_of(original)
    rows = [json.loads(x) for x in lines]
    theirs = [(x, r) for x, r in zip(lines, rows) if _source(_line(r), kind) not in OURS]
    counts: Counter = Counter(ours=len(rows) - len(theirs))
    sizes: list = []
    if kind == "tone":
        clean = {
            _line(r)["line_id"]: _line(r)["input"]["text"] for _, r in theirs if not _line(r).get("twin_of")
        }
    else:
        wanted: dict[str, set[str]] = {}
        for _, r in theirs:
            ln = _line(r)
            wanted.setdefault(_from(ln["source"]), set()).add(
                ln["line_id"] if kind == "r2" else ln["twin_of"]
            )
        found = {key: ctx.rows(key, ids) for key, ids in wanted.items()}
    stands = {}
    for x, r in theirs:
        ln = _line(r)
        lid = ln["line_id"]
        if kind == "tone":
            stand = copy.deepcopy(r)
            _tone_out(_line(stand), ctx.upstream, clean, counts)
            back = copy.deepcopy(stand)
            _tone_in(_line(back), ctx.upstream, clean)
            if dumps(back, style) != x:
                raise Mismatch(f"{lid}: its stand-in row doesn't rebuild it")
            for h in (
                _line(stand)["input"]["text"],
                _line(stand)["input"]["previous"],
                _line(stand)["generator"],
            ):
                _count(h, lid, counts, sizes)
        elif kind == "r2":
            if dumps(found[_from(ln["source"])][lid], style) != dumps(ln, style):
                raise Mismatch(f"{lid}: the row's line isn't the open set's")
            stand = {**r, "line": {"$row": OPEN_ROW, "src": ln["source"], "id": lid}}
            counts[OPEN_ROW] += 1
        else:
            base_line = found[_from(ln["source"])][ln["twin_of"]]
            src = ln["source"] if ln["source"] in open_sources.SOURCES else "pool"
            stand = {"$row": ASR_ROW, "src": src, "of": ln["twin_of"]}
            stand.update(edits_from(twin_base(base_line), ln["input"]["text"]))
            if dumps(speech_twin(base_line, stand), style) != x:
                raise Mismatch(f"{lid}: its clean line and edits don't rebuild it")
            _count(stand, lid, counts, sizes)
        stands[id(r)] = dumps(stand, style)
    out = "".join(stands.get(id(r), x) + "\n" for x, r in zip(lines, rows))
    return out.encode(), counts, sizes


def _count(h, line_id: str, counts: Counter, sizes: list) -> None:
    if not is_hole(h):
        return
    kind = h.get("$") or h["$row"]
    counts[kind] += 1
    if h.get("f"):
        counts[f"{kind}_lower_case"] += 1
    edits = h.get("e") or []
    if edits:
        deleted, inserted = sum(e[1] - e[0] for e in edits), sum(len(e[2]) for e in edits)
        counts.update(
            {f"{kind}_edited": 1, "edits": len(edits), "chars_deleted": deleted, "chars_inserted": inserted}
        )
        sizes.append((deleted + inserted, line_id, kind, deleted, inserted))


def rebuild_file(standin: bytes, kind: str, ctx: Context) -> bytes:
    """The file a stand-in stands in for, from the sources (`ctx.upstream`) and the files rebuilt before it."""
    style = style_of(standin)
    rows = [json.loads(x) for x in rows_of(standin)]
    if kind == "tone":
        # Clean lines first: a noisy twin's text is made from its clean line's.
        clean: dict[str, str] = {}
        for r in rows:
            ln = _line(r)
            if is_hole(ln["input"]["text"]) and ln["input"]["text"]["$"] != TWIN_HOLE:
                _tone_in(ln, ctx.upstream, clean)
                clean[ln["line_id"]] = ln["input"]["text"]
        for r in rows:
            ln = _line(r)
            if is_hole(ln["input"]["text"]):
                _tone_in(ln, ctx.upstream, clean)
        return ("".join(dumps(r, style) + "\n" for r in rows)).encode()
    found = {key: ctx.rows(key, ids) for key, ids in needs(standin, kind).items()}
    out = []
    for r in rows:
        if kind == "r2" and is_hole(r["line"]):
            h = r["line"]
            r["line"] = found[sources.open_key(h["src"])][h["id"]]
        elif kind == "speech" and is_hole(r):
            key = sources.pool_key() if r["src"] == "pool" else sources.open_key(r["src"])
            r = speech_twin(found[key][r["of"]], r)
        out.append(dumps(r, style) + "\n")
    return "".join(out).encode()


# --- the local check -------------------------------------------------------------------------------------------------


def local_members(copy_root: Path, source: str) -> dict[str, bytes]:
    """An open set's archive files as the parser reads them, from a local copy of the volume's open-raw folder."""
    from lab.reader.pipeline import open_ingest

    return open_ingest.load_raw(source, open_sources.FILES[source], root=copy_root)


def check(copy_root: Path, staged: Path, upstream: Path, log=print) -> dict:
    """Every file rebuilt in memory from the staged tree, the sources and the copy's open-set archives, and compared
    with the copy byte for byte and with the manifest; every kept file in the staged tree checked against the
    manifest."""
    from lab.reader.pipeline import open_ingest
    from lab.reader.tone_sources import Upstream

    manifest = json.loads((staged / MANIFEST_KEY).read_text())
    res: dict = {"kept": {}, "open": {}, "standins": {}}
    for key, want in manifest["kept"].items():
        got = {(len(b), sha256(b)) for b in ((staged / key).read_bytes(), (copy_root / key).read_bytes())}
        res["kept"][key] = "pass" if got == {(want["bytes"], want["sha256"])} else "fail"
    log(f"kept: {dict(Counter(res['kept'].values()))}")
    rebuilt: dict[str, bytes] = {}
    for source, pin in manifest["open"].items():
        members = local_members(copy_root, source)
        bad = sorted(n for n, s in pin["members"].items() if n not in members or sha256(members[n]) != s)
        extra = sorted(set(members) - set(pin["members"]))
        body = open_ingest.build_lines(source, members)
        same = body == (copy_root / pin["key"]).read_bytes()
        ok = sha256(body) == pin["sha256"] and same and not bad and not extra
        res["open"][pin["key"]] = {
            "result": "pass" if ok else "fail",
            "members_differ": len(bad) + len(extra),
        }
        rebuilt[pin["key"]] = body
        log(f"{pin['key']}: {res['open'][pin['key']]['result']}")
    ctx = Context(Upstream(upstream), read=lambda key: rebuilt[key])
    for key in manifest["order"]:
        entry = manifest["standins"][key]
        body = rebuild_file((staged / standin_key(key)).read_bytes(), entry["kind"], ctx)
        same = body == (copy_root / key).read_bytes()
        ok = sha256(body) == entry["sha256"] and same
        res["standins"][key] = "pass" if ok else "fail"
        if key == sources.pool_key():
            rebuilt[key] = body
        log(f"{key}: {res['standins'][key]}")
    res["passed"] = all(v == "pass" for v in (*res["kept"].values(), *res["standins"].values())) and all(
        v["result"] == "pass" for v in res["open"].values()
    )
    return res


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="rebuild every file in memory and compare it with the copy")
    checking.add_argument("--copy", type=Path, required=True, help="a full copy of data/reader/ as it was")
    checking.add_argument("--staged", type=Path, required=True, help="the staged data/reader/ folder")
    checking.add_argument(
        "--upstream", type=Path, required=True, help="the sources, as tone_sources.download leaves them"
    )
    auditing = commands.add_parser("audit", help="check the staged tree holds no third-party text")
    auditing.add_argument("staged", type=Path)
    auditing.add_argument("--copy", type=Path, help="the copy, whose third-party texts are looked for")
    args = parser.parse_args()
    if args.command == "check":
        res = check(args.copy, args.staged, args.upstream)
    else:
        from lab.reader import publish_data

        res = publish_data.audit(args.staged, args.copy)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
