"""The sources of Tone's third-party lines, at the revisions its pool and label sets were built from: GoEmotions and
Civil Comments comments and Gutenberg dialogue, on Hugging Face. ``download`` fetches each file and checks it against
its sha256; ``Upstream`` reads a line's text, the line before and its label from them by the line's source_ref."""

import hashlib
import os
import re
import tempfile
from array import array
from pathlib import Path

SOURCES: dict[str, dict] = {
    "goemotions": {
        "repo": "google-research-datasets/go_emotions",
        "revision": "add492243ff905527e67aeb8b80c082af02207c3",
        "files": {
            "simplified/train-00000-of-00001.parquet": {
                "bytes": 2_767_678,
                "sha256": "b7d74279616ae7c9b8374ab62ea9f9d6504d36a577bb17f745d720dc2b0d4e76",
            },
        },
        "licence": "Apache-2.0",
        "credit": "GoEmotions (Demszky et al., 2020; Google), Apache 2.0",
    },
    "civil_comments": {
        "repo": "google/civil_comments",
        "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
        "files": {
            "data/validation-00000-of-00001.parquet": {
                "bytes": 20_955_569,
                "sha256": "2e0eb65474e7e1290df8689fc93eea783160c898c09ae15382c0a9662367bd03",
            },
            "data/train-00000-of-00002.parquet": {
                "bytes": 193_527_147,
                "sha256": "c20f01c3aecbdd942886cacb0ee67e995df33bc712a622fde75927ed3d6ccefe",
            },
            "data/train-00001-of-00002.parquet": {
                "bytes": 186_778_090,
                "sha256": "a76af69d7b24758bc38b196bc422d4767935c1dd7d3c1465554649c546d99ccb",
            },
        },
        "licence": "CC0-1.0",
        "credit": "Civil Comments (Borkan et al., 2019; Jigsaw), CC0 1.0",
    },
    "gutenberg": {
        "repo": "willwade/Gutenberg-dialog-en",
        "revision": "f3a0a1df18275cd9f61f7d161ad6eed3ce35f88c",
        "files": {
            "dev.txt": {
                "bytes": 74_692_317,
                "sha256": "70be07c77a9de393a1cb367612fdc3267f7aa3ee7637ee0de5a5469c4eb4a7cf",
            },
            "test.txt": {
                "bytes": 75_354_424,
                "sha256": "94b6ba76e1bf5ac4889be30b21e9bc038ddd119adb580f4d7b6fec1d606e3631",
            },
            "train.txt": {
                "bytes": 1_379_240_316,
                "sha256": "7ba5e57ef0f2d57adfeb942ec922c0e8fb4ce6eff21024ffa3b7785a44741ab2",
            },
        },
        "licence": "MIT; the novels are public domain (Project Gutenberg)",
        "credit": "The Gutenberg Dialogue Dataset (Csaky and Recski, 2021), English, from public-domain Project Gutenberg "
        "books, MIT",
    },
}
# Where each source_ref's file is: Civil Comments rows untagged (the label sets) are its validation file's; Gutenberg
# refs start with the file's name. Train.txt is read to its first 400 million characters, as the pool's was.
CIVIL_FILES = {
    "": "data/validation-00000-of-00001.parquet",
    "train0": "data/train-00000-of-00002.parquet",
    "train1": "data/train-00001-of-00002.parquet",
}
GUTENBERG_CHARS = {"dev": None, "test": None, "train": 400_000_000}

# From the Tone pipeline: the labels, the bands, the detokeniser, how dialogues are split, and what clean_text and
# noisy do to a text.
# fmt: off
GOEMOTIONS_LABELS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                     "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                     "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                     "remorse", "sadness", "surprise", "neutral"]
# fmt: on


def civil_bands(r: dict) -> str:
    if (r.get("threat") or 0) >= 0.3 or (r.get("insult") or 0) >= 0.5 or (r.get("toxicity") or 0) >= 0.6:
        return "hostile"
    if (r.get("toxicity") or 0) >= 0.2:
        return "heated"
    return "civil"


def detok(text: str) -> str:
    """The dataset is lower case with spaces before punctuation. Restore something readable."""
    t = re.sub(r"\s+([.,!?;:])", r"\1", text.strip())
    t = re.sub(r"\s+(n't|'s|'re|'ll|'ve|'d|'m)\b", r"\1", t)
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t


def gutenberg_dialogues(text: str) -> list[list[str]]:
    return [[u for u in block.splitlines() if u.strip()] for block in text.split("\n\n") if block.strip()]


def ws(text: str) -> str:
    """clean_text's change to a text it keeps: whitespace runs made one space, the ends trimmed."""
    return re.sub(r"\s+", " ", text).strip()


def noisy_base(text: str) -> str:
    """noisy's change to a clean line's text, before it drops a small word: no punctuation, no capitals."""
    return re.sub(r"[,.?!;:\"]", "", " ".join(text.split())).lower().strip()


def file_rel(source: str, name: str) -> str:
    """A source file's place in the upstream folder: hf/<repo>@<revision>/<path>."""
    s = SOURCES[source]
    return f"hf/{s['repo']}@{s['revision']}/{name}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(source: str, name: str, cache: Path) -> Path:
    """One source file from Hugging Face at its revision into a cache folder, its size and sha256 checked."""
    from huggingface_hub import hf_hub_download

    s, want = SOURCES[source], SOURCES[source]["files"][name]
    local = Path(
        hf_hub_download(s["repo"], name, repo_type="dataset", revision=s["revision"], cache_dir=cache)
    ).resolve()
    got = (local.stat().st_size, sha256_file(local))
    if got != (want["bytes"], want["sha256"]):
        raise RuntimeError(f"{s['repo']}@{s['revision']}/{name}: {got} is not the pinned {want}")
    return local


def download(folder: Path) -> dict:
    """Every source file into `folder` at its place (`file_rel`), checked; a file already there and right is kept."""
    folder = Path(folder)
    out = {}
    for source, s in SOURCES.items():
        for name, want in s["files"].items():
            dest = folder / file_rel(source, name)
            if (
                dest.is_file()
                and dest.stat().st_size == want["bytes"]
                and sha256_file(dest) == want["sha256"]
            ):
                out[file_rel(source, name)] = "kept"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder) as cache:
                os.replace(fetch_file(source, name, Path(cache)), dest)
            out[file_rel(source, name)] = "downloaded"
    return out


class Dialogues:
    """One Gutenberg file's dialogues as ``gutenberg_dialogues`` splits them, found by index without the whole list."""

    _TEXT = re.compile(r"\S")

    def __init__(self, text: str):
        self.text = text
        self.spans = array("q")
        pos = 0
        while True:
            end = text.find("\n\n", pos)
            stop = len(text) if end < 0 else end
            if self._TEXT.search(text, pos, stop):
                self.spans.extend((pos, stop))
            if end < 0:
                break
            pos = end + 2

    def __len__(self) -> int:
        return len(self.spans) // 2

    def __getitem__(self, i: int) -> list[str]:
        block = self.text[self.spans[2 * i] : self.spans[2 * i + 1]]
        return [u for u in block.splitlines() if u.strip()]


class Upstream:
    """The source files in a local folder, as `download` leaves them, each read on first use. By a line's source and
    source_ref: its text as the Tone pipeline took it before changing it (whitespace runs made one space, Gutenberg's
    detokenised), the line before as it cut it (Gutenberg), and the label it gave the line (GoEmotions, Civil
    Comments)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._files: dict[str, object] = {}

    def path(self, source: str, name: str) -> Path:
        return self.folder / file_rel(source, name)

    def text(self, source: str, ref: str) -> str:
        if source == "goemotions":
            return ws(self._goemotions(ref)["text"])
        if source == "civil_comments":
            return ws(self._civil(ref)["text"])
        if source == "gutenberg":
            d, ui = self._gutenberg(ref)
            return ws(detok(d[ui]))
        raise KeyError(source)

    def previous(self, source: str, ref: str) -> str:
        """The line before, detokenised and cut to 400 characters; the line's own previous is "<to>: " and this."""
        if source != "gutenberg":
            raise KeyError(source)
        d, ui = self._gutenberg(ref)
        return detok(d[ui - 1])[:400]

    def label(self, source: str, ref: str) -> str | None:
        if source == "goemotions":
            return GOEMOTIONS_LABELS[self._goemotions(ref)["labels"][0]]
        if source == "civil_comments":
            return civil_bands(self._civil(ref))
        return None

    def _read(self, name: str, load):
        if name not in self._files:
            self._files[name] = load()
        return self._files[name]

    def _goemotions(self, ref: str) -> dict:
        def load():
            import pyarrow.parquet as pq

            name = next(iter(SOURCES["goemotions"]["files"]))
            rows = pq.read_table(self.path("goemotions", name), columns=["id", "text", "labels"]).to_pylist()
            return {r["id"]: r for r in rows}

        return self._read("goemotions", load)[ref]

    def _civil(self, ref: str) -> dict:
        m = re.fullmatch(r"(?:(train[01]):)?row(\d+)", ref)
        if not m:
            raise KeyError(ref)
        name = CIVIL_FILES[m.group(1) or ""]

        def load():
            import pyarrow.parquet as pq

            return pq.read_table(
                self.path("civil_comments", name), columns=["text", "toxicity", "insult", "threat"]
            )

        return self._read(name, load).slice(int(m.group(2)), 1).to_pylist()[0]

    def _gutenberg(self, ref: str) -> tuple[Dialogues, int]:
        tag, di, ui = ref.split(":")

        def load():
            with self.path("gutenberg", f"{tag}.txt").open(encoding="utf-8") as f:
                chars = GUTENBERG_CHARS[tag]
                return Dialogues(f.read() if chars is None else f.read(chars))

        return self._read(tag, load)[int(di)], int(ui)
