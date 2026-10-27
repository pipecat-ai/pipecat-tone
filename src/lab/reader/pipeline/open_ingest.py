"""The open conversation sets onto the data volume as corpus lines for r2: each source downloaded at a pinned version,
parsed (``lab.reader.open_sources``), checked and counted.

    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.open_ingest::survey
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.open_ingest::download --sources all-but-dipco
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.open_ingest::download --sources dipco
    MODAL_PROFILE=daily uv run modal run --detach --env tone-tagger -m lab.reader.pipeline.open_ingest::ingest
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.open_ingest::verify
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.open_ingest::manifests
    MODAL_PROFILE=daily uv run modal run --env tone-tagger -m lab.reader.pipeline.open_ingest::show
"""

import json
import pathlib

from lab.reader import open_sources, storage
from lab.reader.app import DATA, app, job, save, save_yaml
from lab.reader.sources import open_key, open_raw_key

UA = {"User-Agent": "pipecat-tone-open-data (research; contact via repository owner)"}


def _agent(conf: dict) -> dict:
    """The User-Agent a source's server is sent: the source's own where it needs one, else ours."""
    return {"User-Agent": conf["ua"]} if conf.get("ua") else UA


@job(DATA, cpu=1.0, memory=2048, timeout=10 * 60)
def survey_sources() -> dict:
    """Each source's metadata at its primary location: GitHub's API for a repository's size, head commit and file
    tree, Zenodo's record, a URL's size and licence page. Nothing is downloaded or stored."""
    import re
    from collections import Counter

    import httpx

    out: dict = {"github": {}, "zenodo": {}, "url": {}}
    with httpx.Client(
        follow_redirects=True, timeout=60, headers={**UA, "Accept": "application/vnd.github+json"}
    ) as c:
        for repo in sorted({s["repo"] for s in open_sources.SOURCES.values() if s["kind"] == "github"}):
            r = c.get(f"https://api.github.com/repos/{repo}")
            info: dict = {
                "status": r.status_code,
                "ratelimit_remaining": r.headers.get("x-ratelimit-remaining"),
            }
            if r.status_code == 200:
                j = r.json()
                info.update(
                    size_kb=j["size"],
                    default_branch=j["default_branch"],
                    github_licence=(j.get("license") or {}).get("spdx_id"),
                    pushed_at=j["pushed_at"],
                )
                rc = c.get(f"https://api.github.com/repos/{repo}/commits/{j['default_branch']}")
                if rc.status_code == 200:
                    info["head"] = rc.json()["sha"]
                    info["head_date"] = rc.json()["commit"]["committer"]["date"]
                t = c.get(
                    f"https://api.github.com/repos/{repo}/git/trees/{info.get('head', j['default_branch'])}",
                    params={"recursive": "1"},
                )
                if t.status_code == 200:
                    tree = t.json()
                    files = [(e["path"], e.get("size", 0)) for e in tree["tree"] if e["type"] == "blob"]
                    info["files"] = len(files)
                    info["truncated"] = tree.get("truncated")
                    info["bytes"] = sum(s for _, s in files)
                    agg: Counter = Counter()
                    cnt: Counter = Counter()
                    for p, s in files:
                        key = "/".join(p.split("/")[:2]) if "/" in p else p
                        agg[key] += s
                        cnt[key] += 1
                    info["top"] = [(k, v, cnt[k]) for k, v in agg.most_common(40)]
                    info["licence_like"] = [
                        p
                        for p, _ in files
                        if re.search(r"(?i)(licen[cs]e|copying|notice|readme)", p.split("/")[-1])
                    ][:40]
                    info["gitattributes"] = any(p.endswith(".gitattributes") for p, _ in files)
            out["github"][repo] = info
        for record in sorted({s["record"] for s in open_sources.SOURCES.values() if s["kind"] == "zenodo"}):
            r = c.get(f"https://zenodo.org/api/records/{record}", headers={"Accept": "application/json"})
            if r.status_code == 200:
                j = r.json()
                out["zenodo"][record] = {
                    "title": j["metadata"].get("title"),
                    "version": j["metadata"].get("version"),
                    "licence": j["metadata"].get("license"),
                    "doi": j.get("doi"),
                    "created": j.get("created"),
                    "files": [
                        {
                            "key": f["key"],
                            "size": f["size"],
                            "checksum": f["checksum"],
                            "link": f["links"]["self"],
                        }
                        for f in j["files"]
                    ],
                }
            else:
                out["zenodo"][record] = {"status": r.status_code}
    for source, conf in open_sources.SOURCES.items():
        if conf["kind"] != "url":
            continue
        with httpx.Client(follow_redirects=True, timeout=60, headers=_agent(conf)) as c:
            for url in conf["urls"]:
                h = c.head(url)
                out["url"][url] = {
                    "status": h.status_code,
                    "length": h.headers.get("content-length"),
                    "last_modified": h.headers.get("last-modified"),
                }
            page = c.get(conf["licence_page"])
        text = re.sub(r"<[^>]+>", " ", page.text)
        out["url"][f"{source}_licence_page"] = {
            "status": page.status_code,
            "text": re.sub(r"\s+", " ", text)[:4000],
        }
    return out


class _Resumable:
    """A file-like reader over one HTTP download that hashes every byte (sha256, md5) and resumes with a Range request
    when the connection drops."""

    def __init__(self, url: str, headers: dict = UA, tries: int = 8):
        import hashlib

        import httpx

        self.url, self.tries, self.pos = url, tries, 0
        self.sha, self.md5 = hashlib.sha256(), hashlib.md5()
        self.client = httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(300, connect=60), headers=headers
        )
        self.headers: dict = {}
        self._cm = None
        self._open()

    def _open(self) -> None:
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._cm = self.client.stream(
            "GET", self.url, headers={"Range": f"bytes={self.pos}-"} if self.pos else {}
        )
        r = self._cm.__enter__()
        r.raise_for_status()
        if self.pos and r.status_code != 206:
            raise RuntimeError(f"no range support at byte {self.pos}")
        if r.headers.get("content-encoding") not in (None, "identity"):
            raise RuntimeError(f"unexpected content-encoding {r.headers.get('content-encoding')}")
        if not self.headers:
            self.headers = {
                k: r.headers.get(k) for k in ("content-length", "content-type", "last-modified", "etag")
            }
            self.final_url = str(r.url)
        self._it = r.iter_raw(1 << 20)
        self._buf = b""

    def read(self, n: int = -1) -> bytes:
        import time

        import httpx

        tries = 0
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return b""
            except (httpx.HTTPError, OSError) as e:  # a dropped connection: resume where we were
                tries += 1
                if tries > self.tries:
                    raise
                print(f"resume at {self.pos} after {type(e).__name__}", flush=True)
                time.sleep(min(60, 5 * tries))
                self._open()
        if n is None or n < 0:
            n = len(self._buf)
        out, self._buf = self._buf[:n], self._buf[n:]
        self.sha.update(out)
        self.md5.update(out)
        self.pos += len(out)
        return out

    def drain(self) -> None:
        while self.read(1 << 20):
            pass

    def summary(self) -> dict:
        return {
            "url": self.url,
            "final_url": getattr(self, "final_url", self.url),
            "bytes": self.pos,
            "sha256": self.sha.hexdigest(),
            "md5": self.md5.hexdigest(),
            "headers": self.headers,
        }


def _fetch(url: str, dest: pathlib.Path, headers: dict = UA) -> dict:
    src = _Resumable(url, headers)
    with dest.open("wb") as f:
        while chunk := src.read(1 << 20):
            f.write(chunk)
    return src.summary()


def _pack(files: dict[str, bytes], dest: pathlib.Path) -> None:
    """A deterministic .tar.gz of the kept files (sorted, no times or owners), so the same files hash the same."""
    import gzip
    import io
    import tarfile

    with (
        dest.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf,
    ):
        for name in sorted(files):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(files[name]), 0, 0o644
            tf.addfile(ti, io.BytesIO(files[name]))


def _sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def _rel(name: str, strip_top: bool) -> str:
    return name.split("/", 1)[1] if strip_top and "/" in name else name


def _match(rel: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def read_members(path: pathlib.Path, strip_top: bool, patterns: list[str] | None = None) -> dict[str, bytes]:
    """The files in a .tar.gz or .zip (those matching `patterns`, or all), by their path inside the archive (GitHub's
    top folder cut)."""
    import tarfile
    import zipfile

    def wanted(rel: str) -> bool:
        return patterns is None or _match(rel, patterns)

    out: dict[str, bytes] = {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if not i.is_dir() and wanted(_rel(i.filename, strip_top)):
                    out[_rel(i.filename, strip_top)] = z.read(i)
        return out
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if m.isfile() and wanted(_rel(m.name, strip_top)):
                out[_rel(m.name, strip_top)] = tf.extractfile(m).read()
    return out


def _shape(x, depth: int = 0):
    if isinstance(x, dict):
        items = list(x.items())
        d = {k: _shape(v, depth + 1) for k, v in items[: 25 if depth < 3 else 8]}
        if len(items) > len(d):
            d["..."] = f"{len(items)} keys"
        return d
    if isinstance(x, list):
        k = 4 if depth < 2 else 2
        return [_shape(v, depth + 1) for v in x[:k]] + ([f"... {len(x)} items"] if len(x) > k else [])
    if isinstance(x, str):
        return x[:160]
    return x


def _peek(name: str, data: bytes) -> dict:
    import csv
    import gzip
    import io
    import zipfile

    if name.endswith(".gz"):
        data, name = gzip.decompress(data), name[:-3]
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return {"zip_members": [(i.filename, i.file_size) for i in z.infolist()][:80]}
        if name.endswith(".json"):
            j = json.loads(data)
            if isinstance(j, dict):
                first = next(iter(j)) if j else None
                return {
                    "type": "dict",
                    "n": len(j),
                    "keys": list(j)[:12],
                    "first_key": first,
                    "first": _shape(j[first]) if first is not None else None,
                }
            return {
                "type": "list",
                "n": len(j),
                "first": _shape(j[0]) if j else None,
                "second": _shape(j[1]) if len(j) > 1 else None,
            }
        if name.endswith(".jsonl"):
            rows = [x for x in data.decode().splitlines() if x.strip()]
            return {"type": "jsonl", "n": len(rows), "first": [_shape(json.loads(x)) for x in rows[:2]]}
        if name.endswith((".csv", ".tsv")):
            delimiter = "\t" if name.endswith(".tsv") else ","
            rows = list(csv.reader(io.StringIO(data.decode("utf-8", "replace")), delimiter=delimiter))
            return {"type": "csv", "n": len(rows), "first": rows[:5]}
    except Exception as e:  # noqa: BLE001 - a peek never fails the download
        return {"error": f"{type(e).__name__}: {e}", "head": data[:1500].decode("utf-8", "replace")}
    return {"head": data[:2500].decode("utf-8", "replace")}


def _select(members: dict[str, bytes], patterns: list[str], limit: int) -> dict[str, bytes]:
    """Members matching the patterns; `outer.zip!inner` patterns look inside a zip member."""
    import io
    import zipfile

    out: dict[str, bytes] = {}
    for pat in patterns:
        if "!" in pat:
            outer, inner = pat.split("!", 1)
            for name in sorted(n for n in members if _match(n, [outer])):
                with zipfile.ZipFile(io.BytesIO(members[name])) as z:
                    for i in z.infolist():
                        if not i.is_dir() and _match(i.filename, [inner]) and len(out) < limit:
                            out[f"{name}!{i.filename}"] = z.read(i)
        else:
            for name in sorted(n for n in members if _match(n, [pat]))[:limit]:
                out[name] = members[name]
    return out


def describe(members: dict[str, bytes], conf: dict) -> dict:
    """What the archive holds: folders with counts and bytes, the licence and readme files' text, the first records."""
    from collections import Counter

    agg: Counter = Counter()
    cnt: Counter = Counter()
    for n, b in members.items():
        key = "/".join(n.split("/")[:2]) if "/" in n else n
        agg[key] += len(b)
        cnt[key] += 1
    texts = {
        n: b.decode("utf-8", "replace")[:7000] for n, b in _select(members, conf.get("read", []), 12).items()
    }
    licence = {
        n: {"sha256": _sha(b), "bytes": len(b)}
        for n, b in _select(members, conf.get("licence_files", []), 12).items()
    }
    peeks = {n: _peek(n.split("!")[-1], b) for n, b in _select(members, conf.get("peek", []), 4).items()}
    return {
        "files": len(members),
        "bytes": sum(len(b) for b in members.values()),
        "folders": [(k, agg[k], cnt[k]) for k in sorted(agg)][:120],
        "licence_files": licence,
        "texts": texts,
        "peeks": peeks,
    }


@job(DATA, cpu=2.0, memory=16384, timeout=4 * 60 * 60)
def download_unit(unit: list[str]) -> dict:
    """One download's archive (`unit`: the sources sharing it) from its primary location onto the volume, described."""
    import datetime
    import tarfile
    import time

    import httpx

    t0 = time.perf_counter()
    first = unit[0]
    conf = open_sources.SOURCES[first]
    work = pathlib.Path("/tmp/open") / first
    work.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "sources": unit,
        "kind": conf["kind"],
        "downloaded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if conf["kind"] == "github":
        repo, commit = conf["repo"], conf["commit"]
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        info.update(repo=repo, commit=commit, source_url=f"https://github.com/{repo}")
        if conf.get("keep"):  # audio in the repository: stream, hash it all, keep the text
            src = _Resumable(url)
            kept: dict[str, bytes] = {}
            with tarfile.open(fileobj=src, mode="r|gz") as tf:
                for m in tf:
                    if m.isfile() and _match(_rel(m.name, True), conf["keep"]):
                        kept[_rel(m.name, True)] = tf.extractfile(m).read()
            src.drain()
            info["full_archive"] = {
                **src.summary(),
                "kept_files": len(kept),
                "note": "streamed and hashed whole; only the kept files are stored",
            }
            name = f"{repo.split('/')[1]}-{commit[:12]}-text.tar.gz"
            _pack(kept, work / name)
            members = kept
        else:
            name = f"{repo.split('/')[1]}-{commit[:12]}.tar.gz"
            info["download"] = _fetch(url, work / name)
            members = read_members(work / name, strip_top=True)
    elif conf["kind"] == "url":
        name = conf["urls"][0].rsplit("/", 1)[1]
        info["download"] = _fetch(conf["urls"][0], work / name, _agent(conf))
        info["source_url"] = conf["urls"][0]
        members = read_members(work / name, strip_top=False)
        with httpx.Client(follow_redirects=True, timeout=60, headers=_agent(conf)) as c:
            page = c.get(conf["licence_page"])
        info["licence_page"] = {
            "url": conf["licence_page"],
            "status": page.status_code,
            "sha256": _sha(page.content),
            "says_cc_by_4": "Creative Commons Attribution 4.0" in page.text,
        }
    elif conf["kind"] == "zenodo":
        with httpx.Client(
            follow_redirects=True, timeout=60, headers={**UA, "Accept": "application/json"}
        ) as c:
            rec = c.get(f"https://zenodo.org/api/records/{conf['record']}").json()
        f = next(x for x in rec["files"] if x["key"] == conf["file"])
        info.update(
            record=conf["record"],
            doi=rec.get("doi"),
            zenodo_licence=(rec["metadata"].get("license") or {}).get("id"),
            source_url=f"https://zenodo.org/records/{conf['record']}",
            zenodo_checksum=f["checksum"],
            zenodo_size=f["size"],
        )
        src = _Resumable(f["links"]["self"])
        kept = {}
        with tarfile.open(fileobj=src, mode="r|gz") as tf:
            for m in tf:
                if (
                    m.isfile()
                    and _match(m.name, conf["keep"])
                    and not m.name.lower().endswith((".wav", ".flac"))
                ):
                    kept[m.name] = tf.extractfile(m).read()
        src.drain()
        summ = src.summary()
        info["full_archive"] = {
            **summ,
            "kept_files": len(kept),
            "md5_matches_zenodo": f"md5:{summ['md5']}" == f["checksum"],
            "note": "streamed and hashed whole; only the kept files are stored",
        }
        name = f"{conf['file'].rsplit('.', 1)[0]}-text.tar.gz"
        _pack(kept, work / name)
        members = kept
    else:
        raise ValueError(conf["kind"])
    path = work / name
    info["archive"] = {"name": name, "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}
    desc = describe(members, conf)
    info["licence_files"] = desc["licence_files"]
    info["files"], info["bytes_unpacked"] = desc["files"], desc["bytes"]
    for i, source in enumerate(unit):
        key = open_raw_key(source, name)
        if i == 0:
            storage.upload(path, key)
        else:
            storage.copy(open_raw_key(first, name), key)
        storage.put_json(
            open_raw_key(source, "download.json"), {**info, "source": source, "archive_key": key}
        )
    # Holds sample text: printed by `show`, never in the repo.
    storage.put_json(open_raw_key(first, "describe.json"), desc)
    info["seconds"] = round(time.perf_counter() - t0)
    return {"info": info}


def load_raw(
    source: str, patterns: list[str] | None = None, root: pathlib.Path | None = None
) -> dict[str, bytes]:
    """A source's kept archive (its sha256 checked against the download), its files by their path inside: from the
    volume, or from `root`, a local copy of it."""
    get = (lambda key: (root / key).read_bytes()) if root else storage.get
    meta = json.loads(get(open_raw_key(source, "download.json")))
    path = root / meta["archive_key"] if root else storage.path(meta["archive_key"])
    got = _sha(path.read_bytes())
    if got != meta["archive"]["sha256"]:
        raise RuntimeError(
            f"{source}: archive sha256 {got} is not the downloaded {meta['archive']['sha256']}"
        )
    return read_members(
        path, strip_top=meta["kind"] == "github" and "full_archive" not in meta, patterns=patterns
    )


def lines_body(built: open_sources.Built) -> bytes:
    """A source's lines.jsonl: each line as pydantic writes it."""
    return "".join(ln.model_dump_json() + "\n" for ln in built.lines).encode()


def build_lines(source: str, members: dict[str, bytes] | None = None) -> bytes:
    """A source's lines.jsonl from its archive on the volume, or from `members`, the files the parser reads, already
    read."""
    return lines_body(
        open_sources.parse(
            source, load_raw(source, open_sources.FILES[source]) if members is None else members
        )
    )


@job(DATA, cpu=2.0, memory=32768, timeout=60 * 60)
def ingest_source(source: str) -> dict:
    """One source's corpus lines from its archive to runs/tonetext/open/<source>-0.1/, with summary.json."""
    import time

    storage.reload()
    t0 = time.perf_counter()
    meta = json.loads(storage.get(open_raw_key(source, "download.json")))
    built = open_sources.parse(source, load_raw(source, open_sources.FILES[source]))
    body = lines_body(built)
    storage.put(open_key(source), body)
    full = meta.get("full_archive")
    summary = {
        **open_sources.summarize(source, built),
        "key": open_key(source),
        "sha256": _sha(body),
        "bytes": len(body),
        "source_url": meta.get("source_url"),
        "commit": meta.get("commit"),
        "record": meta.get("record"),
        "raw": {
            "key": meta["archive_key"],
            "sha256": meta["archive"]["sha256"],
            "bytes": meta["archive"]["bytes"],
        },
        "full_archive": {k: full[k] for k in ("url", "bytes", "sha256", "md5")} if full else None,
        "built_by": "lab/reader/pipeline/open_ingest.py ingest",
        "seconds": round(time.perf_counter() - t0),
    }
    storage.put_json(open_key(source, "summary.json"), summary)
    # Line text: never in the repo.
    storage.put_json(open_key(source, "read_sample.json"), open_sources.sample(built))
    return {"summary": summary}


@job(DATA, cpu=2.0, memory=16384, timeout=30 * 60)
def verify_source(source: str) -> dict:
    """One source's lines re-read and every rule checked again, apart from the code that wrote them."""
    import hashlib
    from collections import Counter

    from lab.reader.schemas import MAX_PREVIOUS, MAX_TEXT, PLAYER, LineInput

    storage.reload()
    body = storage.get(open_key(source))
    summ = json.loads(storage.get(open_key(source, "summary.json")))
    bad: Counter = Counter()
    ids, pairs = set(), set()
    texts_by_split: dict[str, set] = {}
    rows = [json.loads(x) for x in body.decode().splitlines() if x.strip()]
    for r in rows:
        try:
            open_sources.OpenLine.model_validate(r)
            LineInput.model_validate(r["input"])
        except Exception:  # noqa: BLE001 - counted
            bad["invalid row"] += 1
            continue
        inp = r["input"]
        role = open_sources.role_of(r)
        bad["repeated line_id"] += r["line_id"] in ids
        ids.add(r["line_id"])
        bad["repeated (text, previous)"] += (inp["text"], inp["previous"]) in pairs
        pairs.add((inp["text"], inp["previous"]))
        source_, cid = r["group_key"].split(":", 1)
        bad["group key of another source"] += source_ != source
        bad["split not the hash's"] += r["split"] != open_sources.split_of(source, cid)
        bad["text empty or over 400"] += (
            not (0 < len(inp["text"]) <= MAX_TEXT) or inp["text"] != inp["text"].strip()
        )
        bad["previous over 480"] += len(inp["previous"] or "") > MAX_PREVIOUS
        bad["twin_of or generator set"] += r["twin_of"] is not None or r["generator"] is not None
        bad["slices not source, role, modality"] += (
            r["slices"][:1] != [source]
            or role not in open_sources.ROLES
            or open_sources.modality_of(r) not in open_sources.MODALITIES
        )
        two_party = inp["to"] is not None
        bad["two-party speaker not player or Agent"] += two_party and {inp["speaker"], inp["to"]} != {
            PLAYER,
            open_sources.AGENT,
        }
        bad["meeting line with a player"] += not two_party and (
            inp["speaker"] == PLAYER or PLAYER in inp["targets"]
        )
        bad["single sentence with a line before"] += (
            source in ("dynasent_r2", "isarcasmeval") and inp["previous"] is not None
        )
        bad["role and speaker disagree"] += (role == "user" and inp["speaker"] != PLAYER) or (
            role == "agent" and inp["speaker"] != open_sources.AGENT
        )
        texts_by_split.setdefault(r["split"], set()).add(inp["text"])
    if source in open_sources.TEST_ONLY:
        bad["test-only source line outside test"] += sum(r["split"] != "test" for r in rows)
    train = texts_by_split.get("train", set())
    return {
        "source": source,
        "rows": len(rows),
        "sha256_matches_summary": hashlib.sha256(body).hexdigest() == summ["sha256"],
        "lines_match_summary": len(rows) == summ["lines"],
        "problems": {k: v for k, v in bad.items() if v},
        "texts_also_in_train": {k: len(v & train) for k, v in texts_by_split.items() if k != "train"},
        "distinct_texts_by_split": {k: len(v) for k, v in sorted(texts_by_split.items())},
    }


def manifest(source: str) -> dict:
    """A source's manifest from its summary.json and download.json: counts and hashes only, no line text."""
    summ = json.loads(storage.get(open_key(source, "summary.json")))
    meta = json.loads(storage.get(open_raw_key(source, "download.json")))
    out = {
        "corpus_id": f"tonetext-open-{source}-0.1",
        "source": source,
        "format": open_sources.CORPUS_VERSION,
        "line_shape": "the line classifier's corpus line plus `native`; every input checked with lab.reader.schemas.LineInput",
        "key": open_key(source),
        "sha256": summ["sha256"],
        "bytes": summ["bytes"],
        "summary_key": open_key(source, "summary.json"),
        "lines": summ["lines"],
        "turns_read": summ["turns_read"],
        "by_split": summ["by_split"],
        "by_role": summ["by_role"],
        "by_modality": summ["by_modality"],
        "by_role_and_modality": summ["by_role_and_modality"],
        "by_split_and_role": summ["by_split_and_role"],
        "conversations": summ["conversations"],
        "dropped": summ["dropped"],
        "parser_notes": summ["notes"],
        "previous": summ["previous"],
        "native": {k: {"lines": v["lines"], "labels": v["labels"]} for k, v in summ["native"].items()},
        "domains": dict(list(summ["domains"].items())[:25]),
        "split_rule": "test only"
        if summ["test_only"]
        else "by conversation: int(sha256('<source>:<conversation id>')[:8], 16) % 100; 0-4 dev, 5-9 test, 10-99 train",
        "licence": summ["licence"],
        "test_only": summ["test_only"],
        "licence_check": open_sources.LICENCE_CHECKS[source],
        "licence_files": meta.get("licence_files"),
        "credit": summ["credit"],
        "caveats": open_sources.CAVEATS[source],
        "source_url": meta.get("source_url"),
        "version": (
            {"commit": meta["commit"]}
            if meta.get("commit")
            else {"zenodo_record": meta.get("record"), "doi": meta.get("doi")}
            if meta.get("record")
            else {
                "url": meta.get("source_url"),
                "last_modified": (meta.get("download") or {}).get("headers", {}).get("last-modified"),
            }
        ),
        "downloaded_at": meta.get("downloaded_at"),
        "raw": {
            "key": meta["archive_key"],
            "sha256": meta["archive"]["sha256"],
            "bytes": meta["archive"]["bytes"],
        },
        "built_by": "lab/reader/pipeline/open_ingest.py (download, ingest); lab/reader/open_sources.py",
    }
    if meta.get("full_archive"):
        fa = meta["full_archive"]
        out["raw"]["kept_from"] = {
            "url": fa["url"],
            "sha256": fa["sha256"],
            "md5": fa["md5"],
            "bytes": fa["bytes"],
            "note": "streamed and hashed whole; only the transcript, metadata, licence and readme files kept",
        }
    if meta.get("licence_page"):
        out["licence_page"] = meta["licence_page"]
    return out


def _chosen(sources: str) -> list[str]:
    """`all`, `all-but-dipco` or a comma-separated list of sources."""
    if sources == "all":
        return list(open_sources.SOURCES)
    if sources == "all-but-dipco":
        return [s for s in open_sources.SOURCES if s != "dipco"]
    return sources.split(",")


def _each(fn, items: list):
    """A job's results over `items`, one container each; a failure is printed and passed over."""
    for res in fn.map(items, return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED", repr(res))
        else:
            yield res


@app.local_entrypoint()
def survey():
    """Each source's metadata at its primary location, to results/reader/open_ingest/survey.json; a licence page's text
    is printed to read, not kept."""
    res = survey_sources.remote()
    print(json.dumps(res, indent=1))
    for found in res["url"].values():
        found.pop("text", None)
    save("open_ingest", "survey.json", res)


@app.local_entrypoint()
def download(sources: str = "all"):
    """Each source's archive at its pinned commit or version, hashed and kept as downloaded at
    runs/tonetext/open-raw/<source>/ with download.json beside it (URL, commit, sha256, size, the licence files' hashes);
    one container per download, which the sources of one repository share. Harper Valley and DiPCo ship audio in the
    same archive: it is streamed and hashed whole, and only the transcripts, metadata, licence and readme files are
    kept, packed as <name>-text.tar.gz."""
    chosen = _chosen(sources)
    for res in _each(download_unit, [u for u in open_sources.UNITS.values() if any(s in chosen for s in u)]):
        info = res["info"]
        save("open_ingest", f"download-{info['sources'][0]}.json", info)
        print(json.dumps(info, indent=1))


@app.local_entrypoint()
def ingest(sources: str = "all"):
    """Each source's corpus lines, one container per source, to runs/tonetext/open/<source>-0.1/lines.jsonl with
    summary.json beside it (counts only, also saved here) and a reading sample of 40 lines in read_sample.json (on the
    volume only)."""
    for res in _each(ingest_source, _chosen(sources)):
        summ = res["summary"]
        save("open_ingest", f"summary-{summ['source']}.json", summ)
        print(
            f"\n######## {summ['source']}: {summ['lines']} lines from {summ['turns_read']} turns, "
            f"conversations {summ['conversations']}, {summ['seconds']} s"
        )
        keys = (
            "by_split",
            "by_role_and_modality",
            "dropped",
            "notes",
            "previous",
            "words",
            "markup_characters",
        )
        print(json.dumps({k: summ[k] for k in keys}))


@app.local_entrypoint()
def verify(sources: str = "all"):
    """Each source's lines re-read apart from the code that wrote them; the counts of all sources saved together."""
    out = {}
    for res in _each(verify_source, _chosen(sources)):
        out[res["source"]] = res
        print(json.dumps(res))
    if sources == "all":
        save("open_ingest", "verify.json", out)


@app.local_entrypoint()
def manifests(sources: str = "all"):
    """results/reader/manifests/tonetext-open-<source>-0.1.yaml from the counts and hashes on the volume, and with all
    sources their totals: no line text."""
    totals = {}
    for source in _chosen(sources):
        m = manifest(source)
        save_yaml("manifests", f"tonetext-open-{source}-0.1.yaml", m)
        totals[source] = {
            k: m[k]
            for k in ("lines", "by_split", "by_role", "by_modality", "conversations", "dropped", "sha256")
        }
    if sources == "all":
        train_lines = sum(
            v["by_split"].get("train", 0) for s, v in totals.items() if s not in open_sources.TEST_ONLY
        )
        save(
            "open_ingest",
            "summary-open-data.json",
            {
                "lines": sum(v["lines"] for v in totals.values()),
                "train_lines": train_lines,
                "sources": totals,
            },
        )
    print(json.dumps({s: (v["lines"], v["by_split"]) for s, v in totals.items()}, indent=1))


@app.local_entrypoint()
def show(sources: str = "all"):
    """What each download holds (folders, licence and readme text, the shape of a few records), printed to read here,
    never saved."""
    for source in _chosen(sources):
        if source != open_sources.unit_of(source)[0]:
            continue
        desc = json.loads(storage.get(open_raw_key(source, "describe.json")))
        print(f"\n######## {source}: {desc['files']} files, {desc['bytes']} bytes unpacked")
        print("folders:", json.dumps(desc["folders"][:80]))
        print("licence files:", json.dumps(desc["licence_files"]))
        for n, t in desc["texts"].items():
            print(f"\n----- TEXT {n} -----\n{t}")
        for n, pk in desc["peeks"].items():
            print(f"\n----- PEEK {n} -----\n{json.dumps(pk, indent=1)[:9000]}")
