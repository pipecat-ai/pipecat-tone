"""The committee's three questions on a set of lines: asking them, checkpointing the answers, and writing the label
sets with their votes and summaries, which hold no line text."""

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from lab import llm
from lab.reader import scoring, sources, storage
from lab.reader.app import save
from lab.reader.committee import MEMBERS, MODEL_ENV, act_emotion, clients, move, sarcasm
from lab.reader.schemas import ACTS, EMOTIONS, LineTags

CONCURRENCY = 12
ROLES_VOICE = ("user", "agent")
QUESTIONS = {"committee": act_emotion, "exchange": move, "sarcasm": sarcasm}
# The questions A and B may answer by batch.
BATCHED = ("committee", "exchange")


def prepare(step: str, questions: dict | None = None, **keys) -> tuple[dict, dict]:
    """For a labelling entrypoint: the system prompts of `questions` (QUESTIONS by default), built here and passed to
    the jobs, and the run's meta: the labellers' variable names, the prompts' sha256, the inputs' keys and the step."""
    systems = {q: question.system_prompt() for q, question in (questions or QUESTIONS).items()}
    sha = {q: hashlib.sha256(s.encode()).hexdigest() for q, s in systems.items()}
    return systems, {"labellers": dict(MODEL_ENV), "system_prompt_sha256": sha, **keys, "step": step}


def models() -> dict:
    """The labellers' model ids, read inside a job from the tone-tagger-keys secret."""
    missing = [v for v in MODEL_ENV.values() if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"set {', '.join(missing)} in the tone-tagger-keys secret: the labellers' model ids"
        )
    return {m: os.environ[v] for m, v in MODEL_ENV.items()}


def load_lines(key: str, model) -> tuple[list, list]:
    """Every line of a set as `model`, and (custom id, input) for each: ids follow the file's order, so a pilot's lines
    keep the ids they have in the full set."""
    lines = [model.model_validate(r) for r in storage.jsonl(key)]
    return lines, [(f"l{i:05d}", ln.input) for i, ln in enumerate(lines)]


# --- asking (paid) ---------------------------------------------------------------------------------------------------


def ask_member(
    q: str,
    m: str,
    items: list,
    system: str,
    model: str,
    concurrency: int = CONCURRENCY,
    questions: dict | None = None,
) -> dict:
    """One member's answers to question `q` of `questions`, synchronously: {cid: (vote, error, usage, ...)}."""
    question = (questions or QUESTIONS)[q]
    if m == "d":
        return clients.ask_d(items, partial(question.d_request, model=model), question.parse_d, concurrency)
    ask_one = clients.ask_a if m == "a" else clients.ask_b
    return ask_one(question.FORM, items, system, model, concurrency)


def runners(
    systems: dict, models: dict, b_concurrency: int = CONCURRENCY, questions: dict | None = None
) -> dict:
    """Each (question, member)'s synchronous asking, items -> {cid: answer}."""
    questions = questions or QUESTIONS
    return {
        (q, m): partial(
            ask_member,
            q,
            m,
            system=systems[q],
            model=models[m],
            concurrency=b_concurrency if m == "b" else CONCURRENCY,
            questions=questions,
        )
        for q in questions
        for m in MEMBERS
    }


def ask(
    q: str,
    items: list,
    system: str,
    models: dict,
    batches: dict | None = None,
    wait_s: float = 0.0,
    questions: dict | None = None,
) -> tuple:
    """Every member's answers to question `q`: D synchronously; A and B from `batches` ({member: batch id}), each
    waited for `wait_s`, or synchronously; what a batch left unanswered asked synchronously once. Returns (answers, the
    lines filled synchronously, `batches`)."""
    batches = batches or {}
    kw = {"questions": questions} if questions else {}
    with ThreadPoolExecutor(max_workers=len(MEMBERS)) as ex:
        futs = {}
        for m in MEMBERS:
            if m not in batches:
                futs[m] = ex.submit(ask_member, q, m, items, system, models[m], **kw)
            elif isinstance(batches[m], str):
                futs[m] = ex.submit(_collect, q, m, batches[m], items, wait_s, **kw)
        res = {m: futs[m].result() if m in futs else {} for m in MEMBERS}
    filled = {}
    for m in MEMBERS:
        if m in batches:
            missing = [(cid, inp) for cid, inp in items if res[m].get(cid, (None,))[0] is None]
            filled[m] = ask_member(q, m, missing, system, models[m], **kw) if missing else {}
            res[m].update(filled[m])
    return res, filled, batches


def ask_all(
    items: list,
    systems: dict,
    models: dict,
    batches: dict | None = None,
    wait_s: float = 0.0,
    questions: dict | None = None,
) -> dict:
    """Every question of `questions` (the three by default) at once: {question: (answers, the lines filled
    synchronously, batch ids)}. `batches` holds A and B's batches per question."""
    batches, questions = batches or {}, questions or QUESTIONS
    with ThreadPoolExecutor(max_workers=len(questions)) as ex:
        kw = {"questions": questions} if questions is not QUESTIONS else {}
        futs = {
            q: ex.submit(ask, q, items, systems[q], models, batches.get(q), wait_s, **kw) for q in questions
        }
        return {q: f.result() for q, f in futs.items()}


def answers(done: dict, from_batch: dict, batches: dict, questions: dict | None = None) -> dict:
    """{question: (answers, the lines filled synchronously, batch ids)} from the answers by (question, member) and
    those that came from batches."""
    return {
        q: (
            {m: done[(q, m)] for m in MEMBERS},
            {
                m: {cid: r for cid, r in done[(q, m)].items() if cid not in from_batch[(q, m)]}
                for m in MEMBERS
                if (q, m) in from_batch
            },
            batches.get(q, {}),
        )
        for q in (questions or QUESTIONS)
    }


# --- rows in the label sets' formats ---------------------------------------------------------------------------------


def head_json(h) -> dict:
    return {"label": h.label, "intensity": h.intensity, "agree": h.agree, "votes": h.votes}


def _errors(errors: Counter, res: dict, cid: str) -> None:
    for m in res:
        err = res[m].get(cid, (None, clients.NO_RESULT))[1]
        if err:
            errors[f"{m}: {err}"] += 1


def _extra(answer: tuple | None):
    """What D's answer carries beside its vote: the move's seven probabilities, or sarcasm's probability."""
    return answer[3] if answer and len(answer) > 3 else None


def act_emotion_rows(lines: list, items: list, res: dict) -> tuple[list, list, dict]:
    """Votes (with each member's full tags) and labelled lines."""
    votes, labelled, errors = [], [], Counter()
    for (cid, inp), ln in zip(items, lines):
        tags = {m: res[m].get(cid, (None,))[0] for m in res}
        _errors(errors, res, cid)
        v = act_emotion.vote(inp, tags)
        rec = {
            "line_id": ln.line_id,
            "complete": v.complete,
            "emotion": head_json(v.emotion),
            "acts": {t: head_json(h) for t, h in v.acts.items()},
            "backchannel": head_json(v.backchannel),
            "members": {m: (t.model_dump() if t else None) for m, t in tags.items()},
        }
        votes.append(rec)
        labelled.append(
            {
                "line": ln.model_dump(),
                **{k: rec[k] for k in ("complete", "emotion", "acts", "backchannel")},
                "label_source": "committee",
            }
        )
    return votes, labelled, dict(errors.most_common(20))


def exchange_rows(lines: list, items: list, res: dict) -> tuple[list, list, dict]:
    """Votes ({"line_id", label, agree, votes, d_probabilities}) and labelled lines ({"line", "exchange",
    "label_source"}); D's seven probabilities are kept in the votes file only."""
    votes, labelled, errors = [], [], Counter()
    for (cid, _), ln in zip(items, lines):
        v = move.vote({m: res[m].get(cid, (None,))[0] for m in MEMBERS})
        _errors(errors, res, cid)
        votes.append({"line_id": ln.line_id, **v, "d_probabilities": _extra(res["d"].get(cid))})
        labelled.append(
            {
                "line": ln.model_dump(),
                "exchange": {k: v[k] for k in ("label", "agree", "votes")},
                "label_source": "committee",
            }
        )
    return votes, labelled, dict(errors.most_common(20))


def sarcasm_rows(lines: list, items: list, res: dict) -> tuple[list, dict]:
    """Labelled lines ({"line", "sarcasm": {label, votes, agree, d_probability}, "label_source"})."""
    out, errors = [], Counter()
    for (cid, _), ln in zip(items, lines):
        v = sarcasm.vote({m: res[m].get(cid, (None,))[0] for m in MEMBERS})
        _errors(errors, res, cid)
        out.append(
            {
                "line": ln.model_dump(),
                "sarcasm": {**v, "d_probability": _extra(res["d"].get(cid))},
                "label_source": "committee",
            }
        )
    return out, dict(errors.most_common(20))


# --- what the summaries read -----------------------------------------------------------------------------------------


def speaker_role(row: dict) -> str:
    """The voice-agent set's reading of a row: the person (`player`) is the user, anyone else the agent."""
    return "user" if row["line"]["input"]["speaker_kind"] == "player" else "agent"


def act_emotion_summary(
    labelled: list, role: Callable[[dict], str] = speaker_role, roles: tuple[str, ...] = ROLES_VOICE
) -> dict:
    """Coverage, the majority's spread, and A against B: emotion macro-F1 over the nine emotions both directions
    averaged, on every line both answered and on the lines with a majority; by `role` for each of `roles`."""

    def pair(rows, head, classes):
        both = [r for r in rows if r[head]["votes"].get("a") and r[head]["votes"].get("b")]
        a = [r[head]["votes"]["a"] for r in both]
        b = [r[head]["votes"]["b"] for r in both]
        return {
            "lines": len(both),
            "macro_f1": round(
                (scoring.macro_f1(a, b, classes, None) + scoring.macro_f1(b, a, classes, None)) / 2, 4
            ),
            "agree": round(sum(x == y for x, y in zip(a, b)) / max(1, len(both)), 4),
            "kappa": round(scoring.kappa(a, b), 4),
        }

    act_rows = [{"act": h, "line": r["line"]} for r in labelled for h in r["acts"].values()]
    major = [r for r in labelled if r["emotion"]["label"]]

    def spread(rows):
        return dict(Counter(r["emotion"]["label"] or "(no majority)" for r in rows).most_common())

    return {
        "lines": len(labelled),
        "coverage": {
            "emotion_majority": len(major),
            "emotion_unanimous": sum(r["emotion"]["agree"] == 3 for r in labelled),
            "complete_lines": sum(r["complete"] for r in labelled),
            "act_majority": sum(r["act"]["label"] is not None for r in act_rows),
            "backchannel_majority": sum(r["backchannel"]["label"] is not None for r in labelled),
        },
        "emotion": spread(labelled),
        "emotion_by_role": {x: spread([r for r in labelled if role(r) == x]) for x in roles},
        "intensity": dict(Counter(r["emotion"]["intensity"] or "(none)" for r in major).most_common()),
        "backchannel_true": sum(r["backchannel"]["label"] == "true" for r in labelled),
        "acts": dict(Counter(r["act"]["label"] or "(no majority)" for r in act_rows).most_common()),
        "a_b": {
            "emotion_all_lines": pair(labelled, "emotion", EMOTIONS),
            "emotion_majority_lines": pair(major, "emotion", EMOTIONS),
            **{
                f"emotion_{x}_lines": pair([r for r in labelled if role(r) == x], "emotion", EMOTIONS)
                for x in roles
            },
            "act_all_lines": pair(act_rows, "act", ACTS),
            "definition": "labellers A and B macro-F1 against each other, both directions averaged",
        },
        "member_answered": {m: sum(1 for r in labelled if r["emotion"]["votes"].get(m)) for m in MEMBERS},
    }


def exchange_summary(
    labelled: list,
    votes: list,
    role: Callable[[dict], str] = speaker_role,
    roles: tuple[str, ...] = ROLES_VOICE,
    split: str | None = "test",
) -> dict:
    """The move's majority, spread and A against B on the lines of `split`, with `split=None` on every line, and the
    committee against itself."""

    def level(rs):
        if split is None:
            rs = [{**r, "line": {**r["line"], "split": "every"}} for r in rs]
        return move.committee_level(rs, split=split or "every")

    def spread(rows):
        return dict(Counter(r["exchange"]["label"] or "(no majority)" for r in rows).most_common())

    return {
        "lines": len(labelled),
        "majority": sum(1 for r in labelled if r["exchange"]["label"]),
        "move": spread(labelled),
        "move_by_role": {x: spread([r for r in labelled if role(r) == x]) for x in roles},
        "a_b": {
            "committee_level": level(labelled),
            **{f"{x}_lines": level([r for r in labelled if role(r) == x]) for x in roles},
            "definition": "labellers A and B macro-F1 against each other, both directions averaged, on the lines with a "
            "majority; all_lines adds the rest",
        },
        **move.agreement(votes),
    }


def sarcasm_summary(
    rows: list,
    role: Callable[[dict], str] = speaker_role,
    roles: tuple[str, ...] = ROLES_VOICE,
    seeds: tuple[str, ...] = ("sarcasm_seed", "lookalike_seed"),
    ab_roles: tuple[str, ...] = ("user",),
) -> dict:
    """The sarcasm majority, by role, by the seed slices a set was built with (`seeds`), and A against B on every
    line and on each of `ab_roles`."""

    def all_lines(rs):
        """Every line both answered, with or without a majority."""
        both = [
            (r["sarcasm"]["votes"]["a"], r["sarcasm"]["votes"]["b"])
            for r in rs
            if "a" in r["sarcasm"]["votes"] and "b" in r["sarcasm"]["votes"]
        ]
        a, b = [int(x) for x, _ in both], [int(y) for _, y in both]
        return {
            "lines": len(both),
            "f1": round((scoring.f1_sarcastic(a, b) + scoring.f1_sarcastic(b, a)) / 2, 4),
        }

    def seeded(tag):
        rs = [r for r in rows if tag in r["line"]["slices"]]
        return {"lines": len(rs), **dict(Counter(str(r["sarcasm"]["label"]) for r in rs))}

    out = {
        "lines": len(rows),
        "majority": dict(Counter(str(r["sarcasm"]["label"]) for r in rows)),
        "sarcastic_by_role": {
            x: sum(r["sarcasm"]["label"] is True for r in rows if role(r) == x) for x in roles
        },
    }
    if seeds:
        out.update({f"{t}_lines": seeded(t) for t in seeds})
        out["other_lines_sarcastic"] = sum(
            r["sarcasm"]["label"] is True for r in rows if not set(seeds) & set(r["line"]["slices"])
        )
    out["a_b"] = {
        "committee_level": sarcasm.committee_level(rows),
        "all_lines": all_lines(rows),
        **{f"{x}_lines": sarcasm.committee_level([r for r in rows if role(r) == x]) for x in ab_roles},
        "definition": "labellers A and B, F1 of the sarcastic class with each as gold, averaged, on the lines with a "
        "majority",
    }
    out["member_answered"] = {m: sum(1 for r in rows if m in r["sarcasm"]["votes"]) for m in MEMBERS}
    return out


def costs(answers: dict, batch: dict, batched: tuple[str, ...] = BATCHED) -> dict:
    """Dollars from usage per question and member, and their total: `batch` flags the members whose answers to the
    `batched` questions came by batch, at half price but on the lines filled synchronously."""
    out = {}
    for q, (res, filled, _) in answers.items():
        out[q] = clients.cost_usd(res, batch if q in batched else {}, filled)
        if q in batched:
            out[q]["filled_synchronously"] = {m: len(filled.get(m, {})) for m in ("a", "b")}
    total = sum(v for c in out.values() for m, v in c.items() if m in MEMBERS)
    return {**out, "total": round(total, 2)}


def guarded(fn, *args) -> dict:
    """fn(*args), or the error as a summary: the label sets are already written and a failed summary must not lose
    them."""
    try:
        return fn(*args)
    except Exception as exc:
        return {"summary_error": f"{type(exc).__name__}: {str(exc)[:300]}"}


SUMMARIES = {"committee": act_emotion_summary, "exchange": exchange_summary, "sarcasm": sarcasm_summary}


def write_sets(
    set_ids: dict,
    lines: list,
    items: list,
    answers: dict,
    batch: dict,
    meta: dict,
    samples_key: str,
    samples_fn: Callable,
    summaries: dict | None = None,
    batched: tuple[str, ...] = BATCHED,
) -> dict:
    """The label sets `set_ids` names (committee and sarcasm, and the move's when it names one), their summaries and a
    sample of labelled lines (`samples_fn(lines, act rows, move rows or None, sarcasm rows)`, at `samples_key`), from
    `answers` as `ask_all` gives them; `batch` and `batched` as `costs` reads them. `summaries` maps each question to its
    summary function (committee: rows; exchange: rows, votes; sarcasm: rows), the voice-agent set's by default. Returns
    the combined summary, which holds no line text."""
    fns = summaries or SUMMARIES
    key = sources.tonetext_labels_key
    act_votes, act_l, act_err = act_emotion_rows(lines, items, answers["committee"][0])
    sar_l, sar_err = sarcasm_rows(lines, items, answers["sarcasm"][0])
    storage.put_jsonl(key(set_ids["committee"], "votes.jsonl"), act_votes)
    storage.put_jsonl(key(set_ids["committee"]), act_l)
    found = {"committee": (act_err, guarded(fns["committee"], act_l))}
    xch_l = None
    if "exchange" in set_ids:
        xch_votes, xch_l, xch_err = exchange_rows(lines, items, answers["exchange"][0])
        storage.put_jsonl(key(set_ids["exchange"], "votes.jsonl"), xch_votes)
        storage.put_jsonl(key(set_ids["exchange"]), xch_l)
        found["exchange"] = (xch_err, guarded(fns["exchange"], xch_l, xch_votes))
    storage.put_jsonl(key(set_ids["sarcasm"]), sar_l)
    found["sarcasm"] = (sar_err, guarded(fns["sarcasm"], sar_l))
    cost = costs(answers, batch, batched)
    summ = {}
    for q, (errors, summary) in found.items():
        batches = {"batches": answers[q][2]} if q in batched else {}
        summ[q] = {
            "label_id": set_ids[q],
            "key": key(set_ids[q]),
            "errors": errors,
            **batches,
            "cost_usd": cost[q],
            **summary,
        }
        storage.put_json(key(set_ids[q], "summary.json"), {**summ[q], **meta})
    combined = {
        **meta,
        "lines": len(lines),
        "cost_usd_total": cost["total"],
        **summ,
        "samples_key": samples_key,
    }
    storage.put_json(key(set_ids["committee"], "combined_summary.json"), combined)
    storage.put_json(samples_key, samples_fn(lines, act_l, xch_l, sar_l))
    return combined


# --- checkpoints and batches already sent ----------------------------------------------------------------------------


def _chunks(items: list, size: int | None) -> list[list]:
    size = size or max(len(items), 1)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _submit(
    q: str, m: str, items: list, system: str, model: str, label: str, questions: dict | None = None
) -> str:
    form = (questions or QUESTIONS)[q].FORM
    if m == "a":
        return clients.submit_a(form, items, system, model, label)
    return clients.submit_b(form, items, system, model)


def _collect(
    q: str, m: str, batch_id: str, items: list, wait_s: float, questions: dict | None = None
) -> dict:
    collect_one = clients.collect_a if m == "a" else clients.collect_b
    return collect_one((questions or QUESTIONS)[q].FORM, batch_id, items, wait_s)


def retry(fn, *args, tries: int = 5):
    """fn(*args), tried again on a network error from the batch clients."""
    for k in range(tries):
        try:
            return fn(*args)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(60, 5 * 2**k))


class Work:
    """Checkpoints under one work set id, so a restarted job keeps what it asked and sent: each (question, member)'s
    answers as <question>-<member>.json, {cid: [vote, error, usage, ...]}, and its batches as batches.json. The
    questions are QUESTIONS and those sent by batch BATCHED, unless given."""

    def __init__(self, set_id: str, questions: dict | None = None, batched: tuple[str, ...] = BATCHED):
        self.set_id = set_id
        self.questions = questions or QUESTIONS
        self.batched = batched

    def _key(self, name: str) -> str:
        return sources.tonetext_labels_key(self.set_id, f"{name}.json")

    def load(self, q: str, m: str) -> dict:
        key = self._key(f"{q}-{m}")
        if not storage.exists(key):
            return {}
        out = {}
        for cid, row in json.loads(storage.get(key)).items():
            val = LineTags.model_validate(row[0]) if (q == "committee" and row[0] is not None) else row[0]
            out[cid] = (val, *row[1:])
        return out

    def save(self, q: str, m: str, res: dict) -> None:
        def dump(r):
            val = r[0].model_dump() if (q == "committee" and r[0] is not None) else r[0]
            return [val, *r[1:]]

        storage.put_json(self._key(f"{q}-{m}"), {cid: dump(r) for cid, r in res.items()})

    def ask_chunked(
        self, q: str, m: str, items: list, runner, have: dict | None = None, chunk: int = 150
    ) -> tuple[dict, dict]:
        """Every item answered by `runner` (items -> {cid: answer}), keeping answers already saved; the checkpoint is
        written after each chunk, so a restart loses at most one chunk. Returns (all answers, the ones asked now)."""
        have = dict(self.load(q, m) if have is None else have)
        todo = [(cid, inp) for cid, inp in items if have.get(cid, (None,))[0] is None]
        asked: dict = {}
        for i in range(0, len(todo), chunk):
            got = runner(todo[i : i + chunk])
            asked.update(got)
            have.update(got)
            self.save(q, m, have)
        return have, asked

    def send_once(
        self,
        items: list,
        systems: dict,
        models: dict,
        members: tuple[str, ...],
        wait_s: float,
        set_ids: dict,
        chunk: int | None = None,
    ) -> dict:
        """The batches for the batched questions (act and emotion and the move by default) from each of `members`,
        `chunk` lines each (every line when None), each id saved the moment the batch exists, so a restarted call reads
        them and never pays twice. A batch that couldn't be made is saved as its error kind and never retried blind: a
        retry could make a second batch. Returns {member: {question: [id or {"error": kind}]}, "deadline": first sent
        + `wait_s`, "chunk": `chunk`}."""
        key = self._key("batches")
        if storage.exists(key):
            sent = json.loads(storage.get(key))
        else:
            sent = {**{m: {q: [] for q in self.batched} for m in members}, "deadline": None, "chunk": chunk}
        chunks = _chunks(items, sent["chunk"])
        for m in members:
            for q in self.batched:
                for k in range(len(sent[m][q]), len(chunks)):
                    label = set_ids[q] if sent["chunk"] is None else f"{set_ids[q]}-{k}"
                    try:
                        bid = _submit(q, m, chunks[k], systems[q], models[m], label, self.questions)
                    except Exception as exc:
                        bid = {"error": clients.error_kind(exc)}
                    sent[m][q].append(bid)
                    if sent["deadline"] is None:
                        sent["deadline"] = time.time() + wait_s
                    storage.put_json(key, sent)
        return sent

    def collect_batches(self, sent: dict, items: list) -> dict:
        """What the batches `send_once` sent finished, each waited for until the deadline, then cancelled and kept as
        it is: {(question, member): the lines answered}, each saved as <question>-<member>-batch.json to read later."""
        chunks = _chunks(items, sent["chunk"])
        keys = [(q, m) for m in MEMBERS if m in sent for q in self.batched]

        def left() -> float:
            return max(0.0, sent["deadline"] - time.time())

        def one(k):
            q, m = k
            out = {}
            for bid, xs in zip(sent[m][q], chunks):
                if isinstance(bid, str):
                    got = retry(lambda: _collect(q, m, bid, xs, left(), self.questions))
                    out.update({cid: r for cid, r in got.items() if r[0] is not None})
            self.save(q, f"{m}-batch", out)
            return out

        with ThreadPoolExecutor(max_workers=len(keys)) as ex:
            return dict(zip(keys, ex.map(one, keys)))

    def answer_all(
        self, items: list, runners: dict, batched: list, collect: Callable[[], dict]
    ) -> tuple[dict, dict]:
        """Every (question, member)'s answers, checkpointed: those in `batched` from their batches (`collect()`, their
        answers by (question, member)) and the lines those left asked synchronously, the rest synchronously at once.
        Returns (the answers, the answers that came from batches), both by (question, member)."""
        now = [k for k in runners if k not in batched]
        with ThreadPoolExecutor(max_workers=len(now) + 1) as ex:
            futs = {k: ex.submit(self.ask_chunked, *k, items, runners[k]) for k in now}
            sent = ex.submit(collect)
            done = {k: f.result()[0] for k, f in futs.items()}
            got = sent.result()
        from_batch = {k: {cid: r for cid, r in got.get(k, {}).items() if r[0] is not None} for k in batched}

        def fill(k):
            saved = {cid: r for cid, r in self.load(*k).items() if r[0] is not None}
            return self.ask_chunked(*k, items, runners[k], have={**saved, **from_batch[k]})[0]

        with ThreadPoolExecutor(max_workers=len(batched)) as ex:
            done.update(zip(batched, ex.map(fill, batched)))
        return done, from_batch


def b_question(rows: list) -> str | None:
    """Which question a Message Batch asked, from the keys of its first answer that reads as JSON."""
    for _, message in rows:
        if not message:
            continue
        try:
            keys = set(llm.read_message(message)[0])
        except (ValueError, TypeError):
            continue
        return "committee" if "emotion" in keys else ("exchange" if "exchange" in keys else None)
    return None


def collect_sent(batches: dict, items: list, deadline: float) -> dict:
    """{(question, member): answers} from batches sent by hand ({"a_committee", "a_exchange", "b": [ids]}): A's by
    question, B's known by their answers. Each is waited for until `deadline`, then cancelled and kept as it is."""

    def left() -> float:
        return max(0.0, deadline - time.time())

    out = {}
    for q in BATCHED:
        out[(q, "a")] = retry(lambda: _collect(q, "a", batches[f"a_{q}"], items, left()))
    for bid in batches["b"]:
        retry(lambda: clients.wait_or_cancel_b(bid, left()))
        rows = retry(clients.b_results, bid)
        q = b_question(rows)
        if q is not None:
            out[(q, "b")] = clients.b_answers(QUESTIONS[q].FORM, rows, items)
    return out


# --- the stages' local entrypoints -----------------------------------------------------------------------------------


def pilot(stage: str, job, keys: dict, show: Callable, digest: Callable) -> None:
    """A stage's pilot, run now: its result saved as results/reader/<stage>/pilot.json, `digest(result)` printed and the
    pilot's lines shown."""
    systems, meta = prepare("pilot", **keys)
    res = job.remote(systems, meta)
    save(stage, "pilot.json", res)
    print(json.dumps(digest(res), indent=1, default=str))
    show(json.loads(storage.get(res["samples_key"])))


def spawn(stage: str, step: str, job, keys: dict, **options) -> None:
    """A stage's step spawned with `options`, to outlive this client under `modal run --detach`: its call id saved
    with the run's meta and the options as results/reader/<stage>/<step>-spawned.json."""
    systems, meta = prepare(step, **keys)
    call = job.spawn(systems, meta, **options)
    save(stage, f"{step}-spawned.json", {"call": call.object_id, **meta, **options})
    print("spawned", call.object_id)


def collect(stage: str, set_ids: dict, samples_key: str, show: Callable, digest: Callable) -> None:
    """A stage's combined summary, once its label job wrote it: saved as results/reader/<stage>/summary.json,
    `digest(summary)` printed and the samples shown."""
    key = sources.tonetext_labels_key(set_ids["committee"], "combined_summary.json")
    if not storage.exists(key):
        print("not yet")
        return
    res = json.loads(storage.get(key))
    save(stage, "summary.json", res)
    print(json.dumps(digest(res), indent=1))
    show(json.loads(storage.get(samples_key)))
