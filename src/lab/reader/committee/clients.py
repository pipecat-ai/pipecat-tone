"""The committee's three APIs: A through OpenAI's Responses API and B through Anthropic's Messages API, each by batch
or synchronously, and D through OpenAI's Decisions API. Keys are read from the environment by name, never printed."""

import email.utils
import json
import os
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from lab import llm
from lab.reader.committee import user_message
from lab.reader.schemas import LineInput

OPENAI = "https://api.openai.com/v1"
BATCH_ENDPOINT = "/v1/responses"
DECISIONS = f"{OPENAI}/decisions"
BATCH_DONE = ("completed", "failed", "expired", "cancelled")
# How long a cancelled batch is waited for, to keep what it finished.
GRACE_S = 1200.0

A_EFFORT = B_EFFORT = "low"
# Standard prices, $ per million tokens of each usage field; batches halve them. B is priced as Claude.
USD = {
    "a": {"input_tokens": 2.0, "output_tokens": 10.0},
    "b": {name: llm.CLAUDE_PRICES[k] for k, name in llm.CLAUDE_USAGE.items()},
    # D's output and cache are free.
    "d": {"input_tokens": 0.10},
}

# Why a member gave a line no vote: the only error texts results keep, beside an HTTP status.
REFUSAL, MAX_TOKENS, INVALID, NO_RESULT = "refusal", "max_tokens", "invalid answer", "no result"
# Why A left an answer incomplete.
INCOMPLETE = {"max_output_tokens": MAX_TOKENS, "content_filter": REFUSAL}


def http_error(status) -> str:
    return f"HTTP {status}"


def error_kind(exc: Exception) -> str:
    """A provider's exception by kind: its HTTP status, else no result."""
    status = getattr(exc, "status_code", None)
    return http_error(status) if status else NO_RESULT


class HTTPError(RuntimeError):
    """A provider's HTTP error: its status, and its body for the log."""

    def __init__(self, status: int, text: str):
        super().__init__(f"{http_error(status)}: {text[:500]}")
        self.status_code = status


@dataclass(frozen=True)
class Form:
    """How A and B are asked a question: the line as they see it, the answer's strict schema and its name, the token
    cap, and the reader that makes an answer a vote, raising when it can't."""

    view: Callable[[LineInput], dict]
    schema: dict
    name: str
    cap: int
    read: Callable[[dict, LineInput], object]


# --- retries ---------------------------------------------------------------------------------------------------------
# Retries follow TypeSafe's SDK policy: backoff from 0.5 s doubling to 30 s, up to 25% jitter taken off, on 408, 429,
# 5xx, connection errors and timeouts, honouring Retry-After.

RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
RETRIES = 6


def _retry_after(headers) -> float | None:
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, when.timestamp() - time.time())


def _delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, 120.0)
    return min(30.0, 0.5 * 2**attempt) * (1 - 0.25 * random.random())


def post(http, url: str, body: dict, ok: Callable[[object], bool]) -> tuple[int, object]:
    """One POST with retries: the last response's status and JSON body (None when it isn't JSON). A 200 whose body `ok`
    rejects is retried too; a connection that never succeeds raises httpx.TransportError."""
    import httpx

    for attempt in range(RETRIES + 1):
        retry_after = None
        try:
            r = http.post(url, json=body)
        except httpx.TransportError:
            if attempt == RETRIES:
                raise
        else:
            try:
                data = r.json()
            except ValueError:
                data = None
            done = ok(data) if r.status_code == 200 else r.status_code not in RETRY_STATUSES
            if done or attempt == RETRIES:
                return r.status_code, data
            retry_after = _retry_after(r.headers)
        time.sleep(_delay(attempt, retry_after))


def _is_object(data) -> bool:
    return isinstance(data, dict)


def _wait_or_cancel(ended: Callable[[], bool], cancel: Callable[[], None], wait_s: float) -> None:
    """Until a batch ends, or `wait_s` passes: then it is cancelled, keeping what it finished, and given GRACE_S to
    stop."""
    start = time.time()
    while not ended():
        if time.time() - start > wait_s:
            cancel()
            while not ended() and time.time() - start < wait_s + GRACE_S:
                time.sleep(30)
            return
        time.sleep(60)


# --- A: Responses API --------------------------------------------------------------------------------------------------


def a_request(form: Form, custom_id: str, inp: LineInput, system: str, model: str) -> dict:
    """A's request for one line, in the Batch format whether sent by batch or synchronously."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message(form.view(inp))},
            ],
            "text": {
                "format": {"type": "json_schema", "name": form.name, "schema": form.schema, "strict": True}
            },
            "reasoning": {"effort": A_EFFORT},
            "max_output_tokens": form.cap,
        },
    }


def parse_output_line(line: dict) -> tuple[str | None, str | None, dict]:
    """A Batch-format output line as (its text, or None and why there is none; usage)."""
    if line.get("error"):
        return None, NO_RESULT, {}
    resp = line.get("response") or {}
    body = resp.get("body") or {}
    usage = body.get("usage") or {}
    if resp.get("status_code") != 200:
        return None, http_error(resp.get("status_code")), usage
    if body.get("status") != "completed":
        reason = (body.get("incomplete_details") or {}).get("reason")
        return None, INCOMPLETE.get(reason, NO_RESULT), usage
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "refusal":
                return None, REFUSAL, usage
            if content.get("type") == "output_text":
                return content.get("text", ""), None, usage
    return None, NO_RESULT, usage


def parse_a(form: Form, line: dict, inp: LineInput) -> tuple:
    """A's Batch-format output line as (vote, error, usage)."""
    text, error, usage = parse_output_line(line)
    if error:
        return None, error, usage
    try:
        return form.read(json.loads(text), inp), None, usage
    except (ValueError, KeyError, TypeError):
        return None, INVALID, usage


def answer_sync(lines: list[dict], concurrency: int = 8) -> list[dict]:
    """Each Batch-format line's body sent to /v1/responses, the answers as Batch-format output lines."""
    import httpx

    http = httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})

    def one(line: dict) -> dict:
        try:
            status, body = post(http, f"{OPENAI}/responses", line["body"], _is_object)
        except httpx.TransportError:
            return {"custom_id": line["custom_id"], "error": {"message": NO_RESULT}}
        body = body if _is_object(body) else {}
        return {"custom_id": line["custom_id"], "response": {"status_code": status, "body": body}}

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, lines))
    finally:
        http.close()


def ask_a(form: Form, items: list, system: str, model: str, concurrency: int = 8) -> dict:
    """A's answers synchronously; one request first, so the shared system prompt is in the provider's cache."""
    if not items:
        return {}
    by_id = dict(items)
    lines = [a_request(form, cid, inp, system, model) for cid, inp in items]
    first = answer_sync(lines[:1], concurrency=1)
    rest = answer_sync(lines[1:], concurrency)
    return {ln["custom_id"]: parse_a(form, ln, by_id[ln["custom_id"]]) for ln in first + rest}


class BatchClient:
    """OpenAI's Files and Batches endpoints."""

    def __init__(self, api_key: str, timeout: float = 120.0):
        import httpx

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._http = httpx.Client(
            base_url=OPENAI, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    def close(self) -> None:
        self._http.close()

    def _check(self, r):
        if r.status_code >= 400:
            raise HTTPError(r.status_code, r.text)
        return r

    def upload_batch_file(self, jsonl: bytes, filename: str) -> str:
        r = self._check(
            self._http.post(
                "/files", data={"purpose": "batch"}, files={"file": (filename, jsonl, "application/jsonl")}
            )
        )
        return r.json()["id"]

    def create_batch(self, input_file_id: str, metadata: dict | None = None) -> dict:
        body = {"input_file_id": input_file_id, "endpoint": BATCH_ENDPOINT, "completion_window": "24h"}
        if metadata:
            body["metadata"] = metadata
        return self._check(self._http.post("/batches", json=body)).json()

    def retrieve_batch(self, batch_id: str) -> dict:
        return self._check(self._http.get(f"/batches/{batch_id}")).json()

    def cancel_batch(self, batch_id: str) -> None:
        self._check(self._http.post(f"/batches/{batch_id}/cancel"))

    def file_lines(self, file_id: str) -> list[dict]:
        text = self._check(self._http.get(f"/files/{file_id}/content")).text
        return [json.loads(x) for x in text.splitlines() if x.strip()]


def submit_a(form: Form, items: list, system: str, model: str, label: str) -> str:
    """An A batch of these lines, its file and campaign named `label`: the batch's id."""
    client = BatchClient(os.environ["OPENAI_API_KEY"])
    try:
        body = "\n".join(json.dumps(a_request(form, cid, inp, system, model)) for cid, inp in items).encode()
        file_id = client.upload_batch_file(body, f"{label}.jsonl")
        return client.create_batch(file_id, metadata={"campaign": label})["id"]
    finally:
        client.close()


def wait_or_cancel_a(batch_id: str, wait_s: float) -> None:
    """Until an A batch ends, or `wait_s` passes: then it is cancelled, keeping what it finished."""
    client = BatchClient(os.environ["OPENAI_API_KEY"])
    try:
        _wait_or_cancel(
            lambda: client.retrieve_batch(batch_id)["status"] in BATCH_DONE,
            lambda: client.cancel_batch(batch_id),
            wait_s,
        )
    finally:
        client.close()


def collect_a(form: Form, batch_id: str, items: list, wait_s: float) -> dict:
    """What an A batch finished within `wait_s`, each line's (vote, error, usage); a line without a result has none."""
    wait_or_cancel_a(batch_id, wait_s)
    client = BatchClient(os.environ["OPENAI_API_KEY"])
    by_id = dict(items)
    out = {}
    try:
        batch = client.retrieve_batch(batch_id)
        for file_id in (batch.get("output_file_id"), batch.get("error_file_id")):
            for line in client.file_lines(file_id) if file_id else []:
                if line["custom_id"] in by_id:
                    out[line["custom_id"]] = parse_a(form, line, by_id[line["custom_id"]])
    finally:
        client.close()
    return {cid: out.get(cid, (None, NO_RESULT, {})) for cid in by_id}


# --- B: Messages API ---------------------------------------------------------------------------------------------------


def b_request(form: Form, inp: LineInput, system: str, model: str) -> dict:
    """B's request params for one line; the system prompt, the same for every line, is cached."""
    return {
        "model": model,
        "max_tokens": form.cap,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(form.view(inp))}],
        "output_config": {"effort": B_EFFORT, "format": {"type": "json_schema", "schema": form.schema}},
    }


def parse_b(form: Form, message: dict, inp: LineInput) -> tuple:
    """B's Messages API response, as a dict, as (vote, error, usage)."""
    usage = message.get("usage") or {}
    try:
        answer, stop = llm.read_message(message)
        return (None, stop, usage) if stop else (form.read(answer, inp), None, usage)
    except (ValueError, KeyError, TypeError):
        return None, INVALID, usage


def ask_b(form: Form, items: list, system: str, model: str, concurrency: int = 8) -> dict:
    """B's answers synchronously; one request first, which writes the prompt cache."""
    import anthropic

    client = anthropic.Anthropic(max_retries=4)

    def one(item):
        cid, inp = item
        try:
            message = client.messages.create(**b_request(form, inp, system, model))
        except anthropic.APIStatusError as exc:
            return cid, (None, http_error(exc.status_code), {})
        except anthropic.APIConnectionError:
            return cid, (None, NO_RESULT, {})
        return cid, parse_b(form, message.to_dict(), inp)

    if not items:
        return {}
    out = dict([one(items[0])])
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out.update(pool.map(one, items[1:]))
    return out


def submit_b(form: Form, items: list, system: str, model: str) -> str:
    """A Message Batch of these lines: its id."""
    import anthropic

    requests = [{"custom_id": cid, "params": b_request(form, inp, system, model)} for cid, inp in items]
    return anthropic.Anthropic().messages.batches.create(requests=requests).id


def wait_or_cancel_b(batch_id: str, wait_s: float) -> None:
    """Until a Message Batch ends, or `wait_s` passes: then it is cancelled, keeping what it finished."""
    import anthropic

    batches = anthropic.Anthropic().messages.batches
    _wait_or_cancel(
        lambda: batches.retrieve(batch_id).processing_status == "ended",
        lambda: batches.cancel(batch_id),
        wait_s,
    )


def b_results(batch_id: str) -> list:
    """Every result of a Message Batch as (custom id, message dict or None when it failed). The client is held while
    its results stream: one collected mid-stream fails the read."""
    import anthropic

    client = anthropic.Anthropic()
    return [
        (r.custom_id, r.result.message.to_dict() if r.result.type == "succeeded" else None)
        for r in client.messages.batches.results(batch_id)
    ]


def b_answers(form: Form, rows: list, items: list) -> dict:
    """A Message Batch's results as each line's (vote, error, usage)."""
    by_id = dict(items)
    return {
        cid: parse_b(form, message, by_id[cid]) if message else (None, NO_RESULT, {})
        for cid, message in rows
        if cid in by_id
    }


def collect_b(form: Form, batch_id: str, items: list, wait_s: float) -> dict:
    """What a Message Batch finished within `wait_s`, each line's (vote, error, usage)."""
    wait_or_cancel_b(batch_id, wait_s)
    return b_answers(form, b_results(batch_id), items)


# --- D: Decisions API --------------------------------------------------------------------------------------------------


class DecisionsError(Exception):
    """A Decisions response that gives no vote, by kind."""


def _has_answers(data) -> bool:
    return isinstance(data, dict) and isinstance(data.get("answers"), list)


def ask_d(items: list, request: Callable, parse: Callable, concurrency: int = 8) -> dict:
    """D's answers: `parse(response, input)` per line, or (None, error, {}) when the request fails or `parse` raises."""
    import httpx

    http = httpx.Client(
        timeout=60.0,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )

    def one(item):
        cid, inp = item
        try:
            status, data = post(http, DECISIONS, request(inp), _has_answers)
            if status != 200:
                return cid, (None, http_error(status), {})
            if not _has_answers(data):
                return cid, (None, NO_RESULT, {})
            return cid, parse(data, inp)
        except httpx.TransportError:
            return cid, (None, NO_RESULT, {})
        except DecisionsError as exc:
            return cid, (None, str(exc), {})
        except ValueError:
            return cid, (None, INVALID, {})

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return dict(pool.map(one, items))
    finally:
        http.close()


# --- cost --------------------------------------------------------------------------------------------------------------


def cost_usd(results: dict, batch: dict, filled: dict) -> dict:
    """Dollars per member from usage: the members flagged in `batch` at half price on the lines their batches answered
    (all but those in `filled`), everything else at Standard."""
    out = {}
    for m, res in results.items():
        usd = 0.0
        for cid, (_, _, usage, *_) in res.items():
            standard = sum(((usage or {}).get(name) or 0) * price for name, price in USD[m].items()) / 1e6
            usd += standard / 2 if batch.get(m) and cid not in filled.get(m, {}) else standard
        out[m] = round(usd, 4)
    return out
