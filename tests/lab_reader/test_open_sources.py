"""The open conversation sets' parsers on tiny inline fixtures: no network, no volume. Every text here is made up for
the test."""

import csv
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter

import pytest

from lab.reader import open_sources

from . import open_line


def run(source: str, members: dict[str, bytes]) -> open_sources.Built:
    return open_sources.parse(source, members)


def by_id(built: open_sources.Built) -> dict:
    return {ln.line_id: ln for ln in built.lines}


def j(obj) -> bytes:
    return json.dumps(obj).encode()


def csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode()


# --- the shared rules ------------------------------------------------------------------------------------------------


def test_split_rule():
    for cid in ("a", "dlg-1", "PMUL4398.json", "S01"):
        bucket = int(hashlib.sha256(f"natcs:{cid}".encode()).hexdigest()[:8], 16) % 100
        want = "dev" if bucket < 5 else "test" if bucket < 10 else "train"
        assert open_sources.split_of("natcs", cid) == want
    shares = Counter(open_sources.split_of("star", str(i)) for i in range(20000))
    assert 800 < shares["dev"] < 1200 and 800 < shares["test"] < 1200
    assert {open_sources.split_of("topicalchat_test", str(i)) for i in range(200)} == {"test"}


def test_role_and_modality_are_the_second_and_third_slices():
    line = open_line("x-1", "Hello.", role="agent", modality="asr")
    assert open_sources.role_of(line) == "agent" and open_sources.modality_of(line) == "asr"
    assert open_sources.role_of({"slices": ["natcs"]}) == "" and open_sources.modality_of({}) == ""
    assert open_sources.ROLES == ("user", "agent", "participant")


def test_safe_ids():
    assert open_sources.safe_id("natcs-insurance_000-3") == "natcs-insurance_000-3"
    assert open_sources.safe_id("MultiWOZ-PMUL4398.json-2") == "multiwoz-pmul4398.json-2"
    assert open_sources.safe_id("x-a/b:c d-1") == "x-a-b-c-d-1"


def two(*turns):
    return [
        open_sources.Turn(sp, text, "user" if sp == open_sources.PLAYER else "agent", "written", f"r{i}")
        for i, (sp, text) in enumerate(turns)
    ]


def test_previous_chain_targets_and_drops():
    # Over 400: dropped; with "Agent: " over 480: cut to its last 480 as the next previous.
    long_turn = "x" * 500
    convs = [
        open_sources.Conversation(
            "c1",
            two(
                (open_sources.AGENT, "Hello, how can I help?"),
                (open_sources.PLAYER, "  my card   is lost "),
                (open_sources.AGENT, ""),
                (open_sources.AGENT, long_turn),
                (open_sources.PLAYER, "ok"),
            ),
        )
    ]
    b = open_sources.build("abcd", convs, "MIT")
    lines = b.lines
    assert [ln.input.speaker for ln in lines] == ["Agent", "player", "player"]
    first, second, third = lines
    assert first.input.previous is None and first.input.to == "player" and first.input.targets == ["player"]
    assert second.input.text == "my card is lost" and second.input.previous == "Agent: Hello, how can I help?"
    assert second.input.to == "Agent" and second.input.speaker_kind == "player"
    # the empty turn is not a turn; the long turn is dropped but stands as the next turn's previous, its last 480 chars
    assert third.input.previous == f"Agent: {long_turn}"[-480:] and len(third.input.previous) == 480
    assert b.dropped == Counter({"empty": 1, "over_400_chars": 1})
    assert [ln.line_id for ln in lines] == ["abcd-c1-0", "abcd-c1-1", "abcd-c1-4"]
    assert first.slices == ["abcd", "agent", "written"] and first.group_key == "abcd:c1"
    assert b.conversations == {open_sources.split_of("abcd", "c1"): 1}
    again = open_sources.build(
        "abcd",
        [open_sources.Conversation("c2", two((open_sources.AGENT, "Hello, how can I help?")))],
        "MIT",
        b,
    )
    assert again.dropped["duplicate_text_and_previous"] == 1 and len(again.lines) == 3


def test_meeting_targets():
    turns = [
        open_sources.Turn(n, f"line {i}", "participant", "spoken", f"r{i}")
        for i, n in enumerate(("Alex", "Blair", "Alex"))
    ]
    b = open_sources.build(
        "dipco",
        [open_sources.Conversation("S01", turns, participants=("Alex", "Blair", "Casey"))],
        "CDLA-Permissive-1.0",
    )
    a0, b1, a2 = b.lines
    assert a0.input.to is None and a0.input.targets == ["Blair", "Casey"] and a0.input.speaker_kind == "npc"
    assert b1.input.targets == ["Alex", "Casey"] and b1.input.previous == "Alex: line 0"
    assert a2.input.previous == "Blair: line 1"


# --- one fixture per source --------------------------------------------------------------------------------------------


def test_natcs():
    dev = {
        "dialogue_id": "000",
        "turns": [
            {
                "turn_id": "insurance_000_000",
                "speaker_role": "Agent",
                "utterance": "Hi, this is the insurer.",
                "dialogue_acts": [],
                "intents": [],
            },
            {
                "turn_id": "insurance_000_001",
                "speaker_role": "Customer",
                "utterance": "#Ah I want to pay my bill.",
                "dialogue_acts": ["InformIntent"],
                "intents": ["PayBill"],
            },
            {
                "turn_id": "insurance_000_002",
                "speaker_role": "Customer",
                "utterance": "Is there #2 left?",
                "dialogue_acts": ["InformIntent"],
                "intents": [],
            },
        ],
    }
    bank = {
        "dialogue_id": "banking_0000",
        "turns": [
            {
                "turn_id": "banking_0000_000",
                "speaker_role": "Customer",
                "utterance": "balance please",
                "dialogue_acts": [],
                "intents": [],
            }
        ],
    }
    b = run(
        "natcs",
        {
            "dstc11/development/dialogues.jsonl": j(dev),
            "dstc11/test-banking/dialogues.jsonl": j(bank),
            "dstc11/test-finance/dialogues.jsonl": b"",
        },
    )
    ids = by_id(b)
    pay = ids["natcs-insurance_000-1"]
    assert (
        pay.input.text == "Ah I want to pay my bill."
        and pay.input.previous == "Agent: Hi, this is the insurer."
    )
    assert pay.native == {"scheme": "natcs-intent", "label": "PayBill"} and pay.slices == [
        "natcs",
        "user",
        "written",
        "insurance",
    ]
    assert ids["natcs-insurance_000-2"].native == {
        "scheme": "natcs-predicted-dialogue-act",
        "label": ["InformIntent"],
    }
    assert ids["natcs-insurance_000-2"].input.text == "Is there #2 left?"
    assert ids["natcs-banking_0000-0"].slices == ["natcs", "user", "spoken", "banking"]
    assert ids["natcs-insurance_000-0"].input.speaker == "Agent" and b.notes["filler marks removed"] == 1


def tm_conv(cid, speakers, extra=None):
    return {
        "conversation_id": cid,
        **(extra or {}),
        "utterances": [
            {"index": i, "speaker": sp, "text": f"turn {i} of {cid}"} for i, sp in enumerate(speakers)
        ],
    }


def test_taskmaster():
    tm1 = {
        "TM-1-2019/woz-dialogs.json": j(
            [tm_conv("dlg-w", ["ASSISTANT", "USER"], {"instruction_id": "movie-tickets-2"})]
        ),
        "TM-1-2019/self-dialogs.json": j(
            [tm_conv("dlg-s", ["USER", "ASSISTANT"], {"instruction_id": "restaurant-table-2"})]
        ),
    }
    ids = by_id(run("taskmaster1", tm1))
    assert ids["taskmaster1-dlg-w-1"].slices == ["taskmaster1", "user", "spoken", "movie_tickets"]
    assert ids["taskmaster1-dlg-w-0"].slices == ["taskmaster1", "agent", "written", "movie_tickets"]
    assert ids["taskmaster1-dlg-s-0"].slices == ["taskmaster1", "user", "written", "restaurant_table"]
    assert ids["taskmaster1-dlg-w-1"].input.previous == "Agent: turn 0 of dlg-w"
    tm2 = {"TM-2-2020/data/food-ordering.json": j([tm_conv("dlg-f", ["USER", "ASSISTANT", "ASSISTANT"])])}
    ids = by_id(run("taskmaster2", tm2))
    assert ids["taskmaster2-dlg-f-0"].slices == ["taskmaster2", "user", "spoken", "food_ordering"]
    # The same speaker twice in a row.
    assert ids["taskmaster2-dlg-f-2"].input.previous == "Agent: turn 1 of dlg-f"
    tm3 = {
        "TM-3-2020/data/data_00.json": j(
            [tm_conv("dlg-t", ["user", "assistant"], {"vertical": "Movie Tickets"})]
        )
    }
    ids = by_id(run("taskmaster3", tm3))
    assert ids["taskmaster3-dlg-t-1"].input.speaker == "Agent"
    assert ids["taskmaster3-dlg-t-0"].slices == ["taskmaster3", "user", "written", "movie_tickets"]


def test_abcd():
    data = {
        "train": [
            {
                "convo_id": 7,
                "scenario": {"flow": "product_defect", "subflow": "return_size"},
                "original": [
                    ["agent", "How can I help?"],
                    ["customer", "I need a return."],
                    ["action", "Searching the FAQ pages ..."],
                    ["agent", "Sure, your name?"],
                ],
            }
        ],
        "dev": [],
        "test": [],
    }
    b = run("abcd", {"data/abcd_v1.1.json.gz": gzip.compress(j(data))})
    ids = by_id(b)
    assert sorted(ids) == ["abcd-7-0", "abcd-7-1", "abcd-7-2"]
    assert ids["abcd-7-2"].input.previous == "player: I need a return."  # the action row is not a turn
    assert (
        ids["abcd-7-2"].source_ref == "abcd_v1.1.json#train/7/3"
        and ids["abcd-7-1"].slices[-1] == "product_defect"
    )
    assert b.notes["action rows skipped"] == 1


def test_multiwoz22():
    dlg = [
        {
            "dialogue_id": "SNG1.json",
            "services": ["hotel"],
            "turns": [
                {"speaker": "USER", "turn_id": "0", "utterance": "I need a hotel.", "frames": []},
                {"speaker": "SYSTEM", "turn_id": "1", "utterance": "Which area?", "frames": []},
            ],
        }
    ]
    acts = {
        "SNG1.json": {
            "0": {"dialog_act": {"Hotel-Inform": [["type", "hotel"]]}},
            "1": {"dialog_act": {"Hotel-Request": [["area", "?"]], "general-greet": []}},
        }
    }
    ids = by_id(
        run(
            "multiwoz22",
            {
                "data/MultiWOZ_2.2/train/dialogues_001.json": j(dlg),
                "data/MultiWOZ_2.2/dialog_acts.json": j(acts),
            },
        )
    )
    assert ids["multiwoz22-sng1.json-1"].native == {
        "scheme": "multiwoz22-dialogue-act",
        "label": ["Hotel-Request", "general-greet"],
    }
    assert ids["multiwoz22-sng1.json-0"].slices == ["multiwoz22", "user", "written", "hotel"]
    assert ids["multiwoz22-sng1.json-1"].group_key == "multiwoz22:SNG1.json"


def test_star():
    d = {
        "DialogueID": 12,
        "Scenario": {"Domains": ["bank"]},
        "Events": [
            {"Agent": "UserGuide", "Action": "instruct", "Text": "Pretend you are worried."},
            {"Agent": "User", "Action": "utter", "Text": "Hi, my balance?"},
            {"Agent": "Wizard", "Action": "request_suggestions", "Text": "draft never sent"},
            {
                "Agent": "Wizard",
                "Action": "pick_suggestion",
                "ActionLabel": "bank_ask_pin",
                "Text": "What is your PIN?",
            },
            {"Agent": "Wizard", "Action": "query", "Constraints": []},
            {"Agent": "User", "Action": "utter", "Text": "1234"},
            {"Agent": "Wizard", "Action": "utter", "Text": "Thanks."},
        ],
    }
    b = run("star", {"dialogues/12.json": j(d)})
    texts = [ln.input.text for ln in b.lines]
    assert texts == ["Hi, my balance?", "What is your PIN?", "1234", "Thanks."]
    pin = b.lines[1]
    assert (
        pin.native == {"scheme": "star-wizard-action", "label": "bank_ask_pin"}
        and pin.input.previous == "player: Hi, my balance?"
    )
    assert b.lines[3].native is None and pin.slices == ["star", "agent", "written", "bank"]


def test_ccpem():
    data = [
        {
            "conversationId": "CCPE-1",
            "utterances": [
                {"index": 0, "speaker": "ASSISTANT", "text": "what movies do you like"},
                {
                    "index": 1,
                    "speaker": "USER",
                    "text": "I like thrillers",
                    "segments": [
                        {
                            "annotations": [
                                {
                                    "annotationType": "ENTITY_PREFERENCE",
                                    "entityType": "MOVIE_GENRE_OR_CATEGORY",
                                }
                            ]
                        }
                    ],
                },
            ],
        }
    ]
    ids = by_id(run("ccpem", {"data.json": j(data)}))
    assert ids["ccpem-ccpe-1-1"].slices == ["ccpem", "user", "spoken", "movies"]
    assert ids["ccpem-ccpe-1-0"].slices == ["ccpem", "agent", "written", "movies"]
    assert ids["ccpem-ccpe-1-1"].native == {
        "scheme": "ccpem-annotation-types",
        "label": ["ENTITY_PREFERENCE"],
    }


def test_harpervalley():
    segs = [
        {
            "index": 2,
            "start_ms": 900,
            "speaker_role": "agent",
            "human_transcript": "hello this is the bank",
            "transcript": "hello this is the bang",
            "dialog_acts": ["gridspace_greeting"],
        },
        {
            "index": 1,
            "start_ms": 100,
            "speaker_role": "caller",
            "human_transcript": "[noise]",
            "transcript": "[noise]",
            "dialog_acts": [],
        },
        {
            "index": 3,
            "start_ms": 2000,
            "speaker_role": "caller",
            "human_transcript": "hi i lost my card",
            "transcript": "hi i lost my card",
            "dialog_acts": [],
        },
    ]
    meta = {"tasks": [{"task_type": "replace card"}]}
    b = run("harpervalley", {"data/transcript/abc.json": j(segs), "data/metadata/abc.json": j(meta)})
    ids = by_id(b)
    assert sorted(ids) == [
        "harpervalley-abc-1",
        "harpervalley-abc-2",
        "harpervalley-abc-asr-1",
        "harpervalley-abc-asr-2",
    ]
    assert ids["harpervalley-abc-1"].input.previous is None  # the noise-only segment is not a turn
    assert ids["harpervalley-abc-2"].input.previous == "Agent: hello this is the bank"
    assert ids["harpervalley-abc-asr-2"].input.previous == "Agent: hello this is the bang"
    assert ids["harpervalley-abc-asr-1"].slices == ["harpervalley", "agent", "asr", "replace_card"]
    assert ids["harpervalley-abc-1"].native == {
        "scheme": "harpervalley-model-dialogue-act",
        "label": ["gridspace_greeting"],
    }
    assert {ln.group_key for ln in b.lines} == {"harpervalley:abc"} and b.conversations == {
        open_sources.split_of("harpervalley", "abc"): 1
    }
    assert b.dropped["empty"] == 2 and b.notes["tag removed [noise]"] == 2


AMI_WORDS = {
    "A": '<?xml version="1.0" encoding="ISO-8859-1"?><nite:root xmlns:nite="http://nite.sourceforge.net/">'
    '<w nite:id="M.A.words0" starttime="1.0" endtime="1.2">Okay</w>'
    '<w nite:id="M.A.words1" starttime="1.2" endtime="1.2" punc="true">,</w>'
    '<vocalsound nite:id="M.A.words2" starttime="1.3" endtime="1.4" type="laugh"/>'
    '<w nite:id="M.A.words3" starttime="1.5" endtime="1.7">let&#39;s</w>'
    '<w nite:id="M.A.words4" starttime="1.7" endtime="1.9">start</w>'
    '<w nite:id="M.A.words5" starttime="1.9" endtime="1.9" punc="true">.</w>'
    '<w nite:id="M.A.words6" starttime="5.0" endtime="5.2">Right</w></nite:root>',
    "B": '<?xml version="1.0" encoding="ISO-8859-1"?><nite:root xmlns:nite="http://nite.sourceforge.net/">'
    '<w nite:id="M.B.words0" starttime="3.0" endtime="3.2">Mm-hmm</w>'
    '<w nite:id="M.B.words1" starttime="3.2" endtime="3.2" punc="true">.</w></nite:root>',
}


def ami_acts(agent: str, acts: list[tuple[str | None, str]]) -> bytes:
    body = "".join(
        f'<dact nite:id="M.{agent}.dialog-act.{k}">'
        + (f'<nite:pointer role="da-aspect" href="da-types.xml#id({t})"/>' if t else "")
        + f'<nite:child href="M.{agent}.words.xml#{ref}"/></dact>'
        for k, (t, ref) in enumerate(acts)
    )
    return f'<?xml version="1.0"?><nite:root xmlns:nite="http://nite.sourceforge.net/">{body}</nite:root>'.encode()


def test_ami():
    members = {
        "ontologies/da-types.xml": b'<da-type xmlns:nite="http://nite.sourceforge.net/" nite:id="cmrda" name="da-type">'
        b'<da-type nite:id="ami_daclass_0" name="minor"><da-type nite:id="ami_da_1" name="bck"/></da-type>'
        b'<da-type nite:id="ami_daclass_1" name="task"><da-type nite:id="ami_da_4" name="inf"/></da-type></da-type>',
        "corpusResources/meetings.xml": b'<nite:root xmlns:nite="http://nite.sourceforge.net/"><meeting observation="M" type="scenario"/></nite:root>',
        "words/M.A.words.xml": AMI_WORDS["A"].encode(),
        "words/M.B.words.xml": AMI_WORDS["B"].encode(),
        "dialogueActs/M.A.dialog-act.xml": ami_acts(
            "A", [("ami_da_4", "id(M.A.words0)..id(M.A.words5)"), (None, "id(M.A.words6)")]
        ),
        "dialogueActs/M.B.dialog-act.xml": ami_acts("B", [("ami_da_1", "id(M.B.words0)..id(M.B.words1)")]),
    }
    b = run("ami", members)
    assert [(ln.input.speaker, ln.input.text) for ln in b.lines] == [
        ("Alex", "Okay, let's start."),
        ("Blair", "Mm-hmm."),
        ("Alex", "Right"),
    ]
    okay, mm, right = b.lines
    assert (
        okay.native == {"scheme": "ami-dialogue-act", "label": "inf"}
        and mm.native["label"] == "bck"
        and right.native is None
    )
    assert mm.input.previous == "Alex: Okay, let's start." and right.input.previous == "Blair: Mm-hmm."
    assert (
        okay.input.to is None
        and okay.input.targets == ["Blair"]
        and okay.slices == ["ami", "participant", "spoken", "scenario_meeting"]
    )
    assert b.notes["dialogue acts without a type (kept, native null)"] == 1


def test_dipco():
    def u(spk, start, words):
        return {
            "start_time": {"close-talk": start},
            "end_time": {"close-talk": start},
            "words": words,
            "session_id": "S09",
            "speaker_id": spk,
        }

    utts = [
        u("P34", "00:00:05.00", "so what did you cook"),
        u("P34", "00:00:09.00", "[noise]"),
        u("P33", "00:00:07.50", "pasta [laugh] again"),
        u("P35", "00:00:01.00", "hi everyone"),
    ]
    b = run(
        "dipco",
        {"Dipco/transcriptions/dev/S09.json": j(utts), "Dipco/transcriptions/dev/._S09.json": b"\x00\x05"},
    )
    assert [(ln.input.speaker, ln.input.text) for ln in b.lines] == [
        ("Casey", "hi everyone"),
        ("Blair", "so what did you cook"),
        ("Alex", "pasta again"),
    ]
    assert b.lines[2].input.previous == "Blair: so what did you cook" and b.lines[2].input.targets == [
        "Blair",
        "Casey",
    ]
    assert b.lines[0].group_key == "dipco:S09" and b.dropped["empty"] == 1


def test_dynasent_r2():
    def rows(ids):
        return "".join(
            json.dumps(
                {
                    "text_id": i,
                    "sentence": f"sentence {i}",
                    "gold_label": "mixed" if i.endswith("1") else None,
                    "prompt_data": {"prompt_sentence": "PROMPT TEXT NEVER USED"},
                }
            )
            + "\n"
            for i in ids
        ).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("dynasent-v1.1/dynasent-v1.1-round02-dynabench-train.jsonl", rows(["r2-01", "r2-02"]))
        z.writestr("dynasent-v1.1/dynasent-v1.1-round01-yelp-train.jsonl", rows(["r1-01"]))
        z.writestr("__MACOSX/dynasent-v1.1/._dynasent-v1.1-round02-dynabench-dev.jsonl", b"\x00\x05\x16")
    b = run("dynasent_r2", {"dynasent-v1.1.zip": buf.getvalue()})
    assert [ln.line_id for ln in b.lines] == ["dynasent_r2-r2-01-0", "dynasent_r2-r2-02-0"]
    first = b.lines[0]
    assert first.native == {"scheme": "dynasent_r2-sentiment", "label": "mixed"} and b.lines[1].native is None
    assert (
        first.input.to == "Agent"
        and first.input.previous is None
        and first.slices == ["dynasent_r2", "user", "written"]
    )
    assert all("PROMPT" not in ln.model_dump_json() for ln in b.lines)


def test_isarcasmeval():
    train = csv_bytes(
        ["", "tweet", "sarcastic", "rephrase"],
        [[0, "oh great, more rain &amp; wind", 1, "I am sad it is raining"], [1, "lunch was nice", 0, ""]],
    )
    a = csv_bytes(
        ["text", "sarcastic"], [["love waiting in line", 1], ["I hate waiting in line", 0], ["nice day", 0]]
    )
    c = csv_bytes(
        ["text_0", "text_1", "sarcastic_id"],
        [
            ["I hate waiting in line", "love waiting in line", 1],
            ["so fun, a flat tyre", "I got a flat tyre", 0],
        ],
    )
    b = run(
        "isarcasmeval",
        {"train/train.En.csv": train, "test/task_A_En_test.csv": a, "test/task_C_En_test.csv": c},
    )
    ids = by_id(b)
    sarc, reph = ids["isarcasmeval-train-0-0"], ids["isarcasmeval-train-0-1"]
    assert (
        sarc.native["label"] == 1
        and reph.native["label"] == 0
        and reph.group_key == sarc.group_key == "isarcasmeval:train-0"
    )
    assert sarc.input.text == "oh great, more rain & wind"
    assert reph.input.previous is None and sarc.split == reph.split
    # the first pair joins both of its task A rows; the second has none; task A's third row stands alone
    assert (
        ids["isarcasmeval-test-a0-0"].input.text == "love waiting in line"
        and ids["isarcasmeval-test-a0-1"].native["label"] == 0
    )
    assert (
        "isarcasmeval-test-c1-0" in ids
        and "isarcasmeval-test-a2-0" in ids
        and "isarcasmeval-test-a1-0" not in ids
    )
    # Train 3, the pairs 4, task A's own 1.
    assert len(b.lines) == 8 and b.notes["task A rows joined to a task C pair"] == 2


def test_topicalchat_test():
    conv = {
        "t_1": {
            "content": [
                {
                    "agent": "agent_1",
                    "message": "Did you know cats sleep a lot?",
                    "sentiment": "Curious to dive deeper",
                },
                {"agent": "agent_2", "message": "Wow, I did not!", "sentiment": "Surprised"},
            ]
        }
    }
    b = run(
        "topicalchat_test", {"conversations/test_freq.json": j(conv), "conversations/test_rare.json": j({})}
    )
    first, second = b.lines
    assert (
        {ln.split for ln in b.lines} == {"test"}
        and first.input.speaker == "player"
        and second.input.speaker == "Agent"
    )
    assert second.native == {"scheme": "topicalchat_test-sentiment", "label": "Surprised"}
    assert second.slices == ["topicalchat_test", "participant", "written"] and "test only" in second.licence


def test_every_line_validates_as_written():
    b = run(
        "topicalchat_test",
        {
            "conversations/test_freq.json": j({"t_9": {"content": [{"agent": "agent_1", "message": "hi"}]}}),
            "conversations/test_rare.json": j({}),
        },
    )
    row = json.loads(b.lines[0].model_dump_json())
    assert set(row) == {
        "line_id",
        "source",
        "source_ref",
        "licence",
        "split",
        "group_key",
        "input",
        "slices",
        "twin_of",
        "generator",
        "native",
    }
    assert open_sources.OpenLine.model_validate(row) == b.lines[0]
    with pytest.raises(Exception):
        open_sources.OpenLine.model_validate({**row, "line_id": "Bad ID"})


def test_repeated_ids_share_a_group_and_escapes_go():
    first = open_sources.Conversation(
        "dlg-1", two((open_sources.PLAYER, "book a table\\r"), (open_sources.AGENT, "for when?"))
    )
    # Another version.
    again = open_sources.Conversation(
        "dlg-1", two((open_sources.PLAYER, "book a table"), (open_sources.AGENT, "what time?"))
    )
    b = open_sources.build("taskmaster2", [first, again], "CC-BY-4.0")
    assert [ln.line_id for ln in b.lines] == [
        "taskmaster2-dlg-1-0",
        "taskmaster2-dlg-1-1",
        "taskmaster2-dlg-1-r2-1",
    ]
    assert {ln.group_key for ln in b.lines} == {"taskmaster2:dlg-1"} and len(
        {ln.split for ln in b.lines}
    ) == 1
    assert b.lines[0].input.text == "book a table" and b.notes["literal escape sequences removed"] == 1
    assert b.dropped["duplicate_text_and_previous"] == 1 and b.notes["repeated conversation id"] == 1


def test_natcs_tags_and_cutoffs():
    dlg = {
        "dialogue_id": "banking_0001",
        "turns": [
            {
                "turn_id": "banking_0001_000",
                "speaker_role": "Agent",
                "utterance": "This is ~tellibank Intellibank [Laughter] hi",
                "dialogue_acts": [],
                "intents": [],
            },
            {
                "turn_id": "banking_0001_001",
                "speaker_role": "Customer",
                "utterance": "[sounds of typing]",
                "dialogue_acts": [],
                "intents": [],
            },
        ],
    }
    b = run(
        "natcs",
        {
            "dstc11/development/dialogues.jsonl": b"",
            "dstc11/test-banking/dialogues.jsonl": j(dlg),
            "dstc11/test-finance/dialogues.jsonl": b"",
        },
    )
    assert [ln.input.text for ln in b.lines] == ["This is tellibank Intellibank hi"]
    assert b.dropped["empty"] == 1 and b.notes["cut-off marks (~) removed"] == 1


def test_escaped_apostrophes():
    b = open_sources.build(
        "dynasent_r2",
        [
            open_sources.Conversation(
                "r2-1",
                [open_sources.Turn(open_sources.PLAYER, "I\\'ve never paid", "user", "written", "r")],
                no_previous=True,
            )
        ],
        "CC-BY-4.0",
    )
    assert b.lines[0].input.text == "I've never paid"


def test_ami_spelled_letters():
    assert open_sources.unspell("an L_C_D_ screen on the T_V_s", Counter()) == "an LCD screen on the TVs"
