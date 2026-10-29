"""Stand-ins and rebuilds on made-up rows: each file is made again byte for byte from its stand-in, the stand-in keeps
none of the third-party text, and every file the reader's jobs read is published or rebuilt. No network, no volume.
Every text here is made up for the test."""

import gzip
import hashlib
import io
import json
import tarfile

import pytest

from lab import ROOT
from lab.reader import data, open_sources, publish_data, rebuild, scoring, sources, speech
from lab.reader.pipeline import open_ingest
from lab.reader.rebuild import Context, Mismatch, dumps, rebuild_file, stand_in_file
from pipecat_tone.reader.model import MODEL_FILES

from . import open_line


def body(rows, style) -> bytes:
    return "".join(dumps(r, style) + "\n" for r in rows).encode()


# --- styles, edits and holes -------------------------------------------------------------------------------------------


def test_each_file_has_one_style():
    rows = [{"a": "Café", "b": [1, 2.5, None]}, {"a": "plain"}]
    assert rebuild.style_of(body(rows, "compact_utf8")) == "compact_utf8"
    assert rebuild.style_of(body(rows, "spaced_ascii")) == "spaced_ascii"
    mixed = body(rows[:1], "compact_utf8") + body(rows[:1], "spaced_ascii")
    with pytest.raises(Mismatch):
        rebuild.style_of(mixed)
    with pytest.raises(Mismatch):
        rebuild.style_of(body(rows, "compact_utf8")[:-1])


@pytest.mark.parametrize(
    "base,target",
    [
        ("Thanks [NAME], that made my day!", "Thanks Mara, that made my day!"),
        ("", "new"),
        ("gone", ""),
        ("same", "same"),
        ("naïve café — ok", "naive cafe, ok"),
        ("i want to book a table for two", "I want to book a table for 2."),
    ],
)
def test_edits_make_the_target_from_the_base(base, target):
    edits = rebuild.diff(base, target)
    assert rebuild.patch(base, edits) == target
    assert edits == sorted(edits) and all(a[1] <= b[0] for a, b in zip(edits, edits[1:]))
    assert (edits == []) == (base == target)


def test_a_text_in_another_case_is_edited_from_the_base_in_lower_case():
    base = "PLEASE BOOK A TABLE FOR TWO AT SEVEN TONIGHT AND CALL ME BACK"
    target = "Please book a table for two at seven tonight, and call me back."
    h = rebuild.hole("cc", base, target)
    assert h["f"] == 1 and sum(len(e[2]) for e in h["e"]) == 3  # "P", ",", "."
    assert rebuild.fill(h, base, "x") == target
    assert "f" not in rebuild.hole("ge", "Thanks [NAME], see you.", "Thanks Mara, see you.")
    assert len(rebuild.fold("İstanbul")) == len("İstanbul") and rebuild.fold("ABC") == "abc"


def test_a_hole_keeps_edits_and_a_check_of_its_base():
    h = rebuild.hole("ge", "Thanks [NAME]!", "Thanks Mara!")
    assert list(h) == ["$", "e", "h"] and h["h"] == hashlib.sha256(b"Thanks [NAME]!").hexdigest()[:12]
    assert rebuild.fill(h, "Thanks [NAME]!", "x") == "Thanks Mara!"
    assert "e" not in rebuild.hole("cc", "same", "same")
    with pytest.raises(Mismatch):
        rebuild.fill(h, "Thanks [NAME]?", "x")


# --- Tone's files ---------------------------------------------------------------------------------------------------------


class FakeUpstream:
    """Source texts, lines before and labels by (source, source_ref), as tone_sources.Upstream gives them."""

    def __init__(self, texts: dict, previous: dict | None = None, labels: dict | None = None):
        self.texts, self.prev, self.labels = texts, previous or {}, labels or {}

    def text(self, source, ref):
        return self.texts[(source, ref)]

    def previous(self, source, ref):
        return self.prev[(source, ref)]

    def label(self, source, ref):
        return self.labels.get((source, ref))


def tone_line(lid, source, ref, text, previous=None, generator=None, twin_of=None, to="Mara") -> dict:
    return {
        "line_id": lid,
        "source": source,
        "source_ref": ref,
        "licence": "x",
        "split": "train",
        "group_key": f"{source}:{ref}",
        "input": {
            "speaker": "player",
            "speaker_kind": "player",
            "to": to,
            "targets": [to],
            "text": text,
            "previous": previous,
        },
        "slices": ["player"],
        "twin_of": twin_of,
        "generator": generator,
    }


UP = FakeUpstream(
    {
        ("goemotions", "e1"): "Thanks [NAME], that made my whole day!",
        ("civil_comments", "train0:row7"): "You are wrong about the bridge vote.",
        ("gutenberg", "test:3:1"): "You will see me at the old mill tomorrow.",
    },
    previous={("gutenberg", "test:3:1"): "Where are you going tonight?"},
    labels={("goemotions", "e1"): "gratitude", ("civil_comments", "train0:row7"): "heated"},
)
THEIRS = ("made my whole day", "wrong about the bridge", "old mill tomorrow", "going tonight")


def pool_rows() -> list[dict]:
    clean = tone_line(
        "go-1", "goemotions", "e1", "Thanks Mara, that made my whole day!", generator="gratitude"
    )
    twin = tone_line("go-1-v", "goemotions", "e1", "thanks mara that made whole day", generator="gratitude")
    twin.update(twin_of="go-1", slices=["noisy", "player"])
    return [
        tone_line(
            "mi-1", "mix", "m#1", "Is the café still open?", previous="Mara: Evening.", generator="LW-01"
        ),
        clean,
        tone_line(
            "ci-1",
            "civil_comments",
            "train0:row7",
            "You are wrong about the bridge vote.",
            generator="heated",
        ),
        tone_line(
            "gu-1",
            "gutenberg",
            "test:3:1",
            "You will see me at the old mill tomorrow.",
            "Mara: Where are you going tonight?",
        ),
        twin,
    ]


def test_tone_file_stands_in_without_their_text_and_rebuilds():
    original = body(pool_rows(), "compact_utf8")
    standin, counts, sizes = stand_in_file(original, "tone", Context(UP))
    assert not any(t.encode() in standin for t in THEIRS)
    assert b"Is the caf\xc3\xa9 still open?" in standin  # ours, as it was
    assert {
        k: counts[k] for k in ("ours", "ge", "ge_edited", "cc", "gb", "gbp", "v", "v_edited", "gel", "ccl")
    } == {
        "ours": 1,
        "ge": 1,
        "ge_edited": 1,
        "cc": 1,
        "gb": 1,
        "gbp": 1,
        "v": 1,
        "v_edited": 1,
        "gel": 2,
        "ccl": 1,
    }
    assert {s[1] for s in sizes} == {"go-1", "go-1-v"}
    assert rebuild_file(standin, "tone", Context(UP)) == original
    rows = [json.loads(x) for x in standin.decode().splitlines()]
    assert rows[4]["input"]["text"]["$"] == "v" and all(e[2] == "" for e in rows[4]["input"]["text"]["e"])


def test_tone_label_sets_rebuild_with_the_line_inside_each_row():
    rows = [{"line": ln, "complete": True, "emotion": {"label": "joy", "agree": 3}} for ln in pool_rows()]
    original = body(rows, "spaced_ascii")
    standin, _, _ = stand_in_file(original, "tone", Context(UP))
    assert not any(t.encode() in standin for t in THEIRS)
    assert rebuild_file(standin, "tone", Context(UP)) == original


def test_a_label_the_source_disagrees_with_is_kept():
    up = FakeUpstream(UP.texts, UP.prev, {**UP.labels, ("civil_comments", "train0:row7"): "civil"})
    original = body(pool_rows(), "compact_utf8")
    standin, counts, _ = stand_in_file(original, "tone", Context(up))
    assert counts["label_kept"] == 1 and counts["ccl"] == 0
    assert rebuild_file(standin, "tone", Context(up)) == original


def test_a_changed_source_is_refused():
    original = body(pool_rows(), "compact_utf8")
    standin, _, _ = stand_in_file(original, "tone", Context(UP))
    changed = FakeUpstream(
        {**UP.texts, ("civil_comments", "train0:row7"): "Something else."}, UP.prev, UP.labels
    )
    with pytest.raises(Mismatch, match="ci-1"):
        rebuild_file(standin, "tone", Context(changed))


def test_an_unknown_source_is_refused():
    rows = [tone_line("xx-1", "reddit", "r1", "Hello there.")]
    with pytest.raises(Mismatch, match="reddit"):
        stand_in_file(body(rows, "compact_utf8"), "tone", Context(UP))


# --- r2's label sets and the speech twins --------------------------------------------------------------------------------


def open_body(lines: list[dict]) -> bytes:
    """An open set as pydantic writes it."""
    return "".join(open_sources.OpenLine.model_validate(ln).model_dump_json() + "\n" for ln in lines).encode()


NATCS = [
    open_line("natcs-c1-0", "Hi, I need to reset my card PIN please."),
    open_line(
        "natcs-c1-1", "Sure, can you confirm the last four digits?", role="agent", previous="player: Hi."
    ),
]


def files(store: dict):
    return lambda key: store[key]


def test_r2_rows_point_at_the_open_sets_lines():
    store = {sources.open_key("natcs"): open_body(NATCS)}
    rows = [
        {"line": json.loads(x), "complete": True, "emotion": {"label": "neutral", "agree": 3}}
        for x in store[sources.open_key("natcs")].decode().splitlines()
    ]
    original = body(rows, "spaced_ascii")
    standin, counts, _ = stand_in_file(original, "r2", Context(read=files(store)))
    assert counts["open"] == 2 and b"PIN" not in standin
    assert json.loads(standin.decode().splitlines()[0])["line"] == {
        "$row": "open",
        "src": "natcs",
        "id": "natcs-c1-0",
    }
    assert rebuild.needs(standin, "r2") == {sources.open_key("natcs"): {"natcs-c1-0", "natcs-c1-1"}}
    assert rebuild_file(standin, "r2", Context(read=files(store))) == original
    rows[1]["line"]["input"]["text"] = "Changed."
    with pytest.raises(Mismatch, match="natcs-c1-1"):
        stand_in_file(body(rows, "spaced_ascii"), "r2", Context(read=files(store)))


def test_speech_twins_rebuild_from_their_clean_lines():
    pool_clean = tone_line(
        "ci-1", "civil_comments", "train0:row7", "You are wrong about the bridge vote.", generator="heated"
    )
    ours = tone_line("ge-9", "generated", "g#9", "Where did you leave the lantern?", generator="GEN-1")
    store = {
        sources.open_key("natcs"): open_body(NATCS),
        sources.pool_key(): body([pool_clean, ours], "compact_utf8"),
    }
    cleans = [json.loads(x) for x in store[sources.open_key("natcs")].decode().splitlines()] + [
        pool_clean,
        ours,
    ]
    shouted = next(f"natcs-c2-{i}" for i in range(100) if not speech.lowercased(f"natcs-c2-{i}"))
    cleans.append(
        {
            **cleans[0],
            "line_id": shouted,
            "input": {**cleans[0]["input"], "text": "I NEED TO CANCEL MY ORDER"},
        }
    )
    store[sources.open_key("natcs")] = open_body([*NATCS, cleans[-1]])
    heard = [
        "Hi, I need to reset my car pin, please.",
        "Sure. Can you confirm the last 4 digits?",
        "You're wrong about the bridge vote.",
        "Where'd you leave the lantern?",
        "I need to cancel my orders.",
    ]
    twins = [speech.make_twin(c, t)[0] for c, t in zip(cleans, heard)]
    assert all(twins) and {speech.lowercased(c["line_id"]) for c in cleans} == {True, False}
    original = body(twins, "spaced_ascii")
    standin, counts, _ = stand_in_file(original, "speech", Context(read=files(store)))
    assert counts["asr"] == 4 and counts["ours"] == 1 and counts["asr_lower_case"] == 1
    assert b"bridge" not in standin and b"lantern" in standin and b"cancel" not in standin
    assert rebuild.needs(standin, "speech") == {
        sources.open_key("natcs"): {"natcs-c1-0", "natcs-c1-1", shouted},
        sources.pool_key(): {"ci-1"},
    }
    assert rebuild_file(standin, "speech", Context(read=files(store))) == original


# --- the tables ------------------------------------------------------------------------------------------------------


def test_the_tables_split_the_files_three_ways():
    keep, opened, standin = set(rebuild.KEEP), set(rebuild.OPEN.values()), list(rebuild.STANDIN)
    assert (len(rebuild.KEEP), len(keep), len(opened), len(standin)) == (49, 49, 14, 8)
    assert not (keep & opened or keep & set(standin) or opened & set(standin))
    assert standin[0] == sources.pool_key() and standin[-1] == sources.speech_key()
    assert {rebuild.STANDIN[k] for k in standin} == {"tone", "r2", "speech"}


def published_tables() -> set[str]:
    """The files published or rebuilt: the staged manifest's, if this machine has one, else rebuild's tables."""
    staged = ROOT / "data" / "release" / "hf-reader-public" / "data" / "reader" / rebuild.MANIFEST_KEY
    tables = {*rebuild.KEEP, *rebuild.OPEN.values(), *rebuild.STANDIN}
    if staged.is_file():
        m = json.loads(staged.read_text())
        found = {*m["kept"], *(v["key"] for v in m["open"].values()), *m["standins"]}
        assert found == tables
    return tables


def test_every_file_the_jobs_read_is_published_or_rebuilt():
    read = []

    def jsonl(key):
        read.append(key)
        return []

    data.build_r2(
        jsonl, teachers=sources.R2_TEACHERS
    )  # train::spawn_r2, and export's pools=False is a part of it
    data.build_r2(jsonl, teachers=sources.R2_TEACHERS, support=True, contrast=True)  # rc2's, --recipe r3c
    scoring.test_rows(jsonl)  # score and precision
    for set_id in (sources.SUPPORT_ID, sources.CONTRAST_ID):
        scoring.support_rows(jsonl, set_id)  # compare's support-call and contrast reads
    jsonl(sources.sarcasm_key(sources.CHAIN_TEST))  # score's chain read
    jsonl(sources.open_key("topicalchat_test"))  # score's TopicalChat read
    run = []
    for run_id in (rebuild.RUN_ID, rebuild.RC2_RUN_ID):
        run += [
            sources.run_key(run_id, f)
            for f in ("config.json", "model.pt", "export/config.json", "export/tokenizer.json")
        ]
        run += [sources.run_key(run_id, "export", MODEL_FILES["fp32"])]
        run += [sources.run_key(run_id, "precision", MODEL_FILES["fp16"])]
    assert len(set(read)) > 40
    assert set(read) | set(run) <= published_tables()


# --- reading archives from a local copy -----------------------------------------------------------------------------


def test_open_sets_build_from_a_local_copy_of_their_archive(tmp_path):
    conf = open_sources.SOURCES["isarcasmeval"]
    csv_files = {
        "train/train.En.csv": b",tweet,sarcastic,rephrase\n0,oh great another monday,1,i dislike mondays\n",
        "test/task_A_En_test.csv": b"text,sarcastic\nlovely weather for a picnic,1\n",
        "test/task_C_En_test.csv": b"text_0,text_1,sarcastic_id\n",
    }
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for name, b in csv_files.items():
            info = tarfile.TarInfo(f"iSarcasmEval-x/{name}")
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    archive = sources.open_raw_key("isarcasmeval", "iSarcasmEval-x.tar.gz")
    meta = {
        "kind": conf["kind"],
        "archive_key": archive,
        "archive": {"sha256": hashlib.sha256(raw.getvalue()).hexdigest()},
    }
    (tmp_path / archive).parent.mkdir(parents=True)
    (tmp_path / archive).write_bytes(raw.getvalue())
    (tmp_path / sources.open_raw_key("isarcasmeval", "download.json")).write_text(json.dumps(meta))
    members = rebuild.local_members(tmp_path, "isarcasmeval")
    assert members == csv_files
    lines = open_ingest.build_lines("isarcasmeval", members)
    assert lines == open_ingest.lines_body(open_sources.parse("isarcasmeval", members))
    assert len(lines.splitlines()) == 3
    meta["archive"]["sha256"] = "0" * 64
    (tmp_path / sources.open_raw_key("isarcasmeval", "download.json")).write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="sha256"):
        rebuild.local_members(tmp_path, "isarcasmeval")


# --- the audit's parts ------------------------------------------------------------------------------------------------


def test_git_blob_ids_are_gits(tmp_path):
    (tmp_path / "f").write_bytes(b"hello\n")
    assert publish_data.git_blob(tmp_path / "f") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_strings_inside_holes_are_told_apart():
    row = {
        "line_id": "go-1",
        "input": {"text": {"$": "ge", "e": [[7, 13, "Mara"]], "h": "abc"}, "previous": None},
    }
    found = list(publish_data._strings(row))
    assert ("go-1", False) in found and ("Mara", True) in found


def test_word_grams_ignore_case_and_punctuation():
    ten = "one two three four five six seven eight nine ten"
    assert publish_data.grams("one two three") == set()
    assert len(publish_data.grams(ten)) == 1 and publish_data.grams(
        ten.upper().replace(" ", ", ")
    ) == publish_data.grams(ten)
    assert len(publish_data.grams(ten + " eleven")) == 2
