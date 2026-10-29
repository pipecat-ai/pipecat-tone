"""The reader's data as published on Hugging Face, made locally from a full copy of ``data/reader/`` as it was: the
files that are ours as they are, stand-ins for those that hold others' text, a manifest and a README
(``lab.reader.rebuild``). Nothing here writes to Hugging Face but `push`.

    uv run --env-file .env python -m lab.reader.publish_data verify --copy <copy>
    uv run python -m lab.reader.publish_data stage --copy <copy> --upstream <dir> --out <dir>
    uv run --env-file .env python -m lab.reader.publish_data push <dir> --branch <name>
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.reader import open_sources, rebuild, sources, tone_sources
from lab.reader.sources import MANIFEST_KEY, standin_key
from lab.reader.tone_sources import sha256_file
from pipecat_tone.hub import REPO

PREFIX = "data/reader/"
README_KEY = "README.md"
FORMAT = "tone-reader-data-0.2"
GRAM = 10  # words in a run that counts as a leak
FIRST_PUSH = "Reader data without third-party text: stand-ins and a manifest to rebuild it"


def _shown(key: str) -> str:
    """A key as printed: the old line classifier's files by their folder only."""
    return "models/linecls/" if key.startswith("models/linecls/") else key


def files_in(folder: Path) -> dict[str, Path]:
    return {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.is_file()}


def git_blob(path: Path) -> str:
    """A file's git blob id, which the Hub gives for files outside LFS."""
    body = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def remote_files(revision: str) -> dict:
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        REPO, path_in_repo=PREFIX.rstrip("/"), recursive=True, revision=revision, expand=True
    )
    return {f.path.removeprefix(PREFIX): f for f in tree if hasattr(f, "size")}


def same_as_remote(path: Path, f) -> bool:
    if path.stat().st_size != f.size:
        return False
    return sha256_file(path) == f.lfs.sha256 if f.lfs else git_blob(path) == f.blob_id


def verify_copy(copy: Path, revision: str = "main") -> dict:
    """The copy against ``data/reader/`` on Hugging Face at `revision`: the same paths, each the same size, the LFS
    files the same sha256 and the rest the same git blob."""
    remote, local = remote_files(revision), files_in(copy)
    differ = sorted(k for k in remote.keys() & local.keys() if not same_as_remote(local[k], remote[k]))
    missing, extra = sorted(remote.keys() - local.keys()), sorted(local.keys() - remote.keys())
    return {
        "revision": revision,
        "files": len(remote),
        "lfs_files": sum(1 for f in remote.values() if f.lfs),
        "gb": round(sum(f.size for f in remote.values()) / 1e9, 2),
        "missing": sorted({_shown(k) for k in missing}),
        "extra": sorted({_shown(k) for k in extra}),
        "differ": sorted({_shown(k) for k in differ}),
        "passed": not (missing or extra or differ),
    }


# --- staging -----------------------------------------------------------------------------------------------------------


def _open_pin(copy: Path, source: str) -> dict:
    """What rebuilds an open set: where its archive is from, at its pin, the sha256 of each file the parser reads, and
    the lines' sha256 as rc1 had them."""
    meta = json.loads((copy / sources.open_raw_key(source, "download.json")).read_bytes())
    summary = json.loads((copy / sources.open_key(source, "summary.json")).read_bytes())
    body = (copy / sources.open_key(source)).read_bytes()
    if rebuild.sha256(body) != summary["sha256"]:
        raise RuntimeError(f"{source}: the copy's lines aren't the ones its summary.json records")
    conf = open_sources.SOURCES[source]
    if conf["kind"] == "github":
        origin = {
            "repo": conf["repo"],
            "commit": conf["commit"],
            "url": f"https://github.com/{conf['repo']}/tree/{conf['commit']}",
        }
    elif conf["kind"] == "url":
        origin = {
            "url": conf["urls"][0],
            "zip_sha256": meta["download"]["sha256"],
            "zip_bytes": meta["download"]["bytes"],
        }
    else:
        full = meta["full_archive"]
        origin = {
            "record": conf["record"],
            "file": conf["file"],
            "url": f"https://zenodo.org/records/{conf['record']}",
            "doi": meta.get("doi"),
            "md5": full["md5"],
            "stream_sha256": full["sha256"],
            "stream_bytes": full["bytes"],
        }
    members = rebuild.local_members(copy, source)
    return {
        "key": sources.open_key(source),
        "bytes": len(body),
        "sha256": summary["sha256"],
        "lines": summary["lines"],
        "kind": conf["kind"],
        "origin": origin,
        "members": {name: rebuild.sha256(members[name]) for name in sorted(members)},
        "licence": open_sources.SOURCES[source]["licence"],
        "credit": open_sources.CREDITS[source],
        "rc1_archive": {
            "name": meta["archive"]["name"],
            "sha256": meta["archive"]["sha256"],
            "bytes": meta["archive"]["bytes"],
        },
    }


def _upstream() -> dict:
    return {
        name: {
            "repo": s["repo"],
            "revision": s["revision"],
            "files": {
                path: {**want, "key": sources.upstream_key(tone_sources.file_rel(name, path))}
                for path, want in s["files"].items()
            },
            "licence": s["licence"],
            "credit": s["credit"],
        }
        for name, s in tone_sources.SOURCES.items()
    }


HOLES = {
    "$": "a field hole: the value made again from the source by its row's source_ref, then the edits e made",
    "e": "[start, end, insert] in the base's code points, in order, made last first; none when the value is the base",
    "f": "1 when the edits are made to the base in lower case, for a text speech-to-text wrote in another case",
    "h": "the first 12 hex digits of the base's sha256, checked before the edits are made",
    "ge": "a GoEmotions comment's text, whitespace runs made one space",
    "cc": "a Civil Comments comment's text, whitespace runs made one space",
    "gb": "a Gutenberg utterance, detokenised as the Tone pipeline did, whitespace runs made one space",
    "gbp": 'a Gutenberg line\'s line before: "<input.to>: " and the utterance before, detokenised, first 400 characters',
    "v": "a noisy twin's text: its clean line's text (twin_of, same file) without punctuation or capitals",
    "gel": "a GoEmotions comment's first label",
    "ccl": "a Civil Comments comment's band (hostile, heated, civil) from its toxicity, insult and threat scores",
    "$row": "a whole-row hole: open, the open set's line `id` from `src`; asr, a speech twin of line `of` from the "
    "open set `src` or the pool, its text the clean text (lowercased without punctuation for half the lines, as "
    "lab.reader.speech does) with f and e as in a field hole",
}


def stage(copy: Path, upstream: Path, out: Path, log=print) -> dict:
    """The published folder in `out`: the kept files hardlinked from the copy, the stand-ins (each rebuilt and checked
    against the copy before it's written), the manifest and the README. Returns the hole counts and edit sizes."""
    out.mkdir(parents=True, exist_ok=True)
    kept = {}
    for key in rebuild.KEEP:
        src, dst = copy / key, out / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        kept[key] = {"bytes": src.stat().st_size, "sha256": sha256_file(src)}
    log(f"kept: {len(kept)} files hardlinked")
    opened = {}
    for source in rebuild.OPEN:
        opened[source] = _open_pin(copy, source)
        log(f"open {source}: {len(opened[source]['members'])} archive files pinned")
    ctx = rebuild.Context(tone_sources.Upstream(upstream), read=lambda key: (copy / key).read_bytes())
    standins, holes, sizes = {}, Counter(), []
    for key, kind in rebuild.STANDIN.items():
        original = (copy / key).read_bytes()
        body, counts, found = rebuild.stand_in_file(original, kind, ctx)
        if rebuild.rebuild_file(body, kind, ctx) != original:
            raise rebuild.Mismatch(f"{key}: its stand-in doesn't rebuild it")
        dest = out / standin_key(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        standins[key] = {
            "standin": standin_key(key),
            "kind": kind,
            "style": rebuild.style_of(original),
            "bytes": len(original),
            "sha256": rebuild.sha256(original),
            "lines": original.count(b"\n"),
            "standin_bytes": len(body),
            "standin_sha256": rebuild.sha256(body),
            "holes": dict(sorted(counts.items())),
        }
        holes.update({k: v for k, v in counts.items()})
        sizes += [(*s, key) for s in found]
        log(f"{key}: stand-in of {len(body):,} bytes, {dict(counts)}")
    published = {*rebuild.KEEP, *(standin_key(k) for k in rebuild.STANDIN), MANIFEST_KEY, README_KEY}
    rebuilt = {*rebuild.OPEN.values(), *rebuild.STANDIN}
    dropped = {k: p for k, p in files_in(copy).items() if k not in published and k not in rebuilt}
    manifest = {
        "format": FORMAT,
        "styles": rebuild.STYLES,
        "kept": kept,
        "open": opened,
        "standins": standins,
        "order": list(rebuild.STANDIN),
        "upstream": _upstream(),
        "holes": HOLES,
        "dropped": {
            "files": len(dropped),
            "bytes": sum(p.stat().st_size for p in dropped.values()),
            "sha256": {
                k: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for k, p in dropped.items()
                if _shown(k) == k
            },
            "folders": {
                "models/linecls/": {
                    "files": sum(1 for k in dropped if _shown(k) != k),
                    "bytes": sum(p.stat().st_size for k, p in dropped.items() if _shown(k) != k),
                    "note": "the line classifier Tone shipped, not read by this pipeline",
                }
            },
        },
    }
    (out / MANIFEST_KEY).write_text(json.dumps(manifest, indent=1) + "\n")
    (out / README_KEY).write_text(readme(manifest))
    total = sum(p.stat().st_size for p in files_in(out).values())
    return {"holes": dict(sorted(holes.items())), "sizes": sorted(sizes, reverse=True), "staged_bytes": total}


def sources_table(manifest: dict) -> list[tuple[str, str, str, str, str]]:
    """(source, where at its pin, its URL, licence, credit) for every source the rebuilt files are made from."""
    rows = []
    for source, pin in manifest["open"].items():
        o = pin["origin"]
        at = (
            f"{o['repo']}@{o['commit'][:7]}"
            if "commit" in o
            else f"Zenodo {o['record']}"
            if "record" in o
            else o["url"].rsplit("/", 1)[1]
        )
        rows.append((source, at, o["url"], pin["licence"], open_sources.CREDITS[source]))
    for name, s in manifest["upstream"].items():
        url = f"https://huggingface.co/datasets/{s['repo']}/tree/{s['revision']}"
        rows.append((name, f"{s['repo']}@{s['revision'][:7]}", url, s["licence"], s["credit"]))
    return rows


def readme(manifest: dict) -> str:
    table = "\n".join(
        f"| {s} | [{at}]({url}) | {lic} | {c} |" for s, at, url, lic, c in sources_table(manifest)
    )
    n_open, n_stand, n_kept = len(manifest["open"]), len(manifest["standins"]), len(manifest["kept"])
    return f"""# The reader's data

What `lab.reader` trains, checks and scores the tone reader with, each file at its key on the `tone-reader-data`
volume. It holds no third-party text:

- **{n_kept} files that are ours**, as they are: rc1's and rc2's runs (weights, results, each export and 16-bit
  file), the five r2 teachers' labels, the move pool, the label sets whose lines are all ours, and rc2's support-call
  and contrast sets: their committee labels and each line's call as it was written.
- **{n_stand} stand-ins**, in `rebuild/standins/` at their files' keys: files that mix our lines with others'. Each
  mirrors its file line for line in the same JSON style. Our rows are as they are; each value from a third-party
  source is a hole that says how to make it again from the source at its pin (`holes` in the manifest).
- **The {n_open} open sets are not here**: they are rebuilt from their sources' archives.

`rebuild/manifest.json` holds every file's sha256, the sources' pins, and the sha256 of each archive file the open
sets are read from.

## Rebuilding

`lab.reader.pipeline.pull` copies this folder to the volume. `lab.reader.pipeline.fetch` downloads each source at
its pin, rebuilds the open sets and the stand-ins' files, and writes a file only if its sha256 is the manifest's;
it also checks the files `pull` copied. Both run as pipecat-tone's
[`src/lab/README.md`](https://github.com/pipecat-ai/pipecat-tone/blob/main/src/lab/README.md#the-reader) says.

## Sources

| Source | At | Licence | Credit |
|---|---|---|---|
{table}

Topical-Chat is share-alike: its test splits are read for evaluation only and never trained on. The rest of this
folder is ours.
"""


# --- the audit ----------------------------------------------------------------------------------------------------------


def _strings(x, inside_hole=False):
    """Every string in a JSON value, and whether it's an edit's insert inside a hole."""
    if isinstance(x, str):
        yield x, inside_hole
    elif isinstance(x, dict):
        hole = inside_hole or rebuild.is_hole(x)
        for v in x.values():
            yield from _strings(v, hole)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v, inside_hole)


def grams(text: str) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(words[i : i + GRAM])) for i in range(len(words) - GRAM + 1)}


def _staged_rows(staged: Path):
    """(file, row id, string, inside a hole) for every string in the staged tree's JSON and Markdown files."""
    for key, p in files_in(staged).items():
        if key.endswith(".md"):
            yield key, None, p.read_text(), False
        elif key.endswith(".jsonl"):
            with p.open() as f:
                for i, x in enumerate(f):
                    r = json.loads(x)
                    rid = _row_id(r) or i
                    for s, hole in _strings(r):
                        yield key, rid, s, hole
        elif key.endswith(".json"):
            for s, hole in _strings(json.loads(p.read_bytes())):
                yield key, None, s, hole


def _their_texts(copy: Path):
    """(source, line id, text) for every third-party text in the copy as its source has it: the open sets', and the
    third-party rows' of Tone's files (r2's rows are open-set lines; the speech twins' texts are speech-to-text's)."""
    keys = [*rebuild.OPEN.values(), *(k for k, kind in rebuild.STANDIN.items() if kind == "tone")]
    for key in keys:
        with (copy / key).open() as f:
            for x in f:
                ln = json.loads(x)
                ln = ln.get("line", ln)
                if ln["source"] not in rebuild.OURS:
                    for t in (ln["input"]["text"], ln["input"]["previous"]):
                        if t:
                            yield ln["source"], ln["line_id"], t


def _holes(x):
    """Every hole in a JSON value."""
    if isinstance(x, dict):
        if rebuild.is_hole(x):
            yield x
        for v in x.values():
            yield from _holes(v)
    elif isinstance(x, list):
        for v in x:
            yield from _holes(v)


def _row_id(r: dict) -> str | None:
    """A row's line id; a whole-row hole's, the line it is made from."""
    ln = r.get("line", r) if isinstance(r.get("line", r), dict) else r
    return ln.get("line_id") or ln.get("id") or ln.get("of")


def _long_inserts(staged: Path) -> list[str]:
    """The stand-ins' rows with an edit that puts in GRAM words or more."""
    out = set()
    for key in rebuild.STANDIN:
        with (staged / standin_key(key)).open() as f:
            for x in f:
                r = json.loads(x)
                for h in _holes(r):
                    if any(len(re.findall(r"[a-z0-9]+", e[2].lower())) >= GRAM for e in h.get("e", [])):
                        out.add(_row_id(r))
    return sorted(out)


def _structure(staged: Path) -> dict:
    found = set(files_in(staged))
    expected = {*rebuild.KEEP, *(standin_key(k) for k in rebuild.STANDIN), MANIFEST_KEY, README_KEY}
    bad: Counter = Counter()
    for key, kind in rebuild.STANDIN.items():
        with (staged / standin_key(key)).open() as f:
            for x in f:
                r = json.loads(x)
                if kind == "r2":
                    bad[f"{kind}: line not an open-set hole"] += not (
                        rebuild.is_hole(r["line"]) and r["line"].get("$row") == "open"
                    )
                    continue
                if kind == "speech" and rebuild.is_hole(r):
                    bad[f"{kind}: row hole not a speech twin"] += r.get("$row") != "asr"
                    continue
                ln = r.get("line", r)
                if ln["source"] in rebuild.OURS:
                    continue
                if kind == "speech":
                    bad["speech: a third-party twin kept as a row"] += 1
                    continue
                for field, v in (
                    ("text", ln["input"]["text"]),
                    ("previous", ln["input"]["previous"]),
                    ("generator", ln["generator"]),
                ):
                    bad[f"third-party row keeps its {field}"] += isinstance(v, str)
    ours = set()  # the kept lines' ids, so a call is checked against the lines it was written for
    for key in rebuild.KEEP:
        if not key.endswith(".jsonl"):
            continue
        with (staged / key).open() as f:
            for x in f:
                r = json.loads(x)
                ln = r.get("line", r)
                if "input" in ln:
                    mine = ln.get("source") in rebuild.OURS
                    bad["kept file with a third-party line"] += not mine
                    if mine:
                        ours.add(ln["line_id"])
                elif key in rebuild.CALLS:
                    bad["call of a line not kept as ours"] += r.get("line_id") not in ours
                else:
                    strings = [s for s, _ in _strings(r)]
                    bad["kept file with text besides line ids"] += strings != [r.get("line_id")]
    return {
        "missing": sorted(_shown(k) for k in expected - found),
        "extra": sorted({_shown(k) for k in found - expected}),
        "problems": {k: v for k, v in bad.items() if v},
    }


def audit(staged: Path, copy: Path | None = None, log=print) -> dict:
    """The staged tree's files are the ones it should hold, and no third-party row keeps its text, line before or
    label. With `copy`: every word 10-gram of every third-party text in the copy looked for in every string of the
    staged tree, which should find only stock phrases our own lines share."""
    res: dict = {"structure": _structure(staged)}
    log(f"structure: {res['structure']}")
    ok = not any(res["structure"].values())
    if copy is not None:
        ours: set[int] = set()
        for _, _, s, _ in _staged_rows(staged):
            ours |= grams(s)
        log(f"staged: {len(ours):,} distinct word {GRAM}-grams")
        hits: dict[int, set] = defaultdict(set)
        for where, lid, t in _their_texts(copy):
            for g in grams(t) & ours:
                hits[g].add((where, lid))
        found: dict[int, list] = defaultdict(list)
        for key, rid, s, hole in _staged_rows(staged):
            for g in grams(s) & hits.keys():
                found[g].append((key, rid, hole))
        shared = Counter(len(v) for v in hits.values())
        res["leaks"] = {
            "grams_matched": len(hits),
            "third_party_lines": len({x for v in hits.values() for x in v}),
            "by_source": dict(Counter(w for v in hits.values() for w, _ in v).most_common()),
            "grams_by_lines_sharing": {
                "1": shared[1],
                "2-9": sum(n for k, n in shared.items() if 2 <= k <= 9),
                "10-99": sum(n for k, n in shared.items() if 10 <= k <= 99),
                "100+": sum(n for k, n in shared.items() if k >= 100),
            },
            "grams_by_sources_sharing": dict(
                sorted(Counter(len({w for w, _ in v}) for v in hits.values()).items())
            ),
            "staged_strings": len({(k, r) for v in found.values() for k, r, _ in v}),
            "by_staged_file": dict(Counter(k for v in found.values() for k, _, _ in v).most_common()),
            "inside_holes": sorted({str(r) for v in found.values() for _, r, h in v if h}),
            "long_inserts": _long_inserts(staged),
        }
        log(f"leaks: {res['leaks']}")
        ok = ok and not res["leaks"]["inside_holes"]
    res["passed"] = ok
    return res


# --- publishing --------------------------------------------------------------------------------------------------------


def push(out: Path, branch: str, revision: str = "main", message: str = FIRST_PUSH) -> str:
    """The staged folder as one commit on `branch`, made from `revision` unless it's main: files new or changed added,
    every other file under data/reader/ deleted. Returns the commit's URL."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    if branch != "main":
        api.create_branch(REPO, branch=branch, revision=revision)
    remote, local = remote_files(branch), files_in(out)
    ops = [CommitOperationDelete(PREFIX + k) for k in sorted(remote.keys() - local.keys())]
    ops += [
        CommitOperationAdd(PREFIX + k, str(p))
        for k, p in local.items()
        if k not in remote or not same_as_remote(p, remote[k])
    ]
    commit = api.create_commit(
        REPO,
        operations=ops,
        revision=branch,
        commit_message=message,
    )
    return commit.commit_url


def main():
    from dotenv import load_dotenv

    from lab import ROOT

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verifying = commands.add_parser("verify", help="check the copy against Hugging Face")
    verifying.add_argument("--copy", type=Path, required=True)
    verifying.add_argument("--revision", default="main")
    staging = commands.add_parser("stage", help="write the published folder")
    staging.add_argument("--copy", type=Path, required=True)
    staging.add_argument("--upstream", type=Path, required=True)
    staging.add_argument("--out", type=Path, required=True)
    pushing = commands.add_parser("push", help="commit the staged folder to a new branch, or to main")
    pushing.add_argument("out", type=Path)
    pushing.add_argument("--branch", required=True)
    pushing.add_argument("--message", default=FIRST_PUSH, help="the commit's message")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    if args.command == "verify":
        print(json.dumps(verify_copy(args.copy, args.revision), indent=1))
    elif args.command == "stage":
        res = stage(args.copy, args.upstream, args.out)
        print(json.dumps({**res, "sizes": res["sizes"][:20]}, indent=1))
    else:
        print(push(args.out, args.branch, message=args.message))


if __name__ == "__main__":
    main()
