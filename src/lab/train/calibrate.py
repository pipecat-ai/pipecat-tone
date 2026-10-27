"""The breath bias: how far to shift a model's breath logit for it to breathe as often as a target on dev, written into
its config as ``sound_bias`` with its cap on a reply's breaths. The target is the teacher's breaths, or with
``--speech`` the teacher's held, for each length of sentence, to how often ICSI's speakers breathe before or inside one.

    uv run python -m lab.train.calibrate models/<folder> <dev.jsonl> ... [--speech] [--per-reply 1] [--kinds]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lab.corpus.rows import Row, load_rows
from lab.corpus.speech_breaths import (
    SENTENCE_LENGTHS,
    band,
    band_name,
    breath_kinds,
    icsi_turns,
    sentence_rate,
    summary,
)
from pipecat_tone.tagger import ToneTaggerModel
from pipecat_tone.tagger.model import words

BIASES = np.round(np.arange(-6.0, 6.0001, 0.01), 2) + 0.0  # adding 0.0 turns -0.0 into 0.0


def breath_bias(
    model: ToneTaggerModel, rows: list[Row], stats: dict | None = None, cap: int | None = None
) -> dict:
    """The breath bias for the rows offering breath, at most ``cap`` breaths to a reply. Without ``stats``, one bias
    fitted to the teacher's count. With a speech summary, one per sentence length fitted to the teacher's breaths or
    people's, whichever are fewer: breaths the teacher never takes aren't forced, as where they'd go the model can only
    guess."""
    offered = sorted((r for r in rows if "breath" in r.space.sounds), key=lambda r: (r.call, r.turn, r.n))
    margins = np.array([breath_margin(model, r) for r in offered])
    lengths = SENTENCE_LENGTHS if stats else ((1, None),)
    counts = [words(r.sentence) for r in offered]
    bands = np.array([band(n, lengths) for n in counts])
    gold = np.array([r.labels.sound_before == "breath" for r in offered])
    teachers = [int(gold[bands == k].sum()) for k in range(len(lengths))]
    people = None
    if stats:
        rate = sentence_rate(stats)
        people = [
            round(sum(rate(n, not r.earlier) for n, r, b in zip(counts, offered, bands) if b == k))
            for k in range(len(lengths))
        ]
    aims = [min(t, p) for t, p in zip(teachers, people)] if people else teachers
    replies = [(r.call, r.turn) for r in offered]
    starts = np.array([i == 0 or replies[i] != replies[i - 1] for i in range(len(replies))])
    biases = fit(margins, bands, aims, starts, cap)
    chosen = breathes(margins, bands, biases, starts, cap)
    found = {
        "target": "speech" if stats else "teacher",
        "bias": [[high, float(b)] for (_, high), b in zip(lengths, biases)],
        "rows": len(offered),
        "teacher": int(gold.sum()),
        "aim": sum(aims),
        "model": int(chosen.sum()),
        "unbiased": int(breathes(margins, bands, np.zeros(len(aims)), starts, cap).sum()),
    }
    if people:
        found["lengths"] = {
            band_name(*length): {
                "teacher": teachers[k],
                "people": people[k],
                "aim": aims[k],
                "model": int(chosen[bands == k].sum()),
            }
            for k, length in enumerate(lengths)
        }
    return found


def breath_margin(model: ToneTaggerModel, row: Row) -> float:
    """How far a sentence's breath logit is from winning its sound before: the breath head's logit where no other
    sound goes."""
    heads, _ = model.heads(row.turn_context(), row.earlier, row.sentence)
    sounds, before = model.config["sounds"], heads[1]
    others = [i for i, c in enumerate(sounds) if c == "none" or (c != "breath" and c in row.space.sounds)]
    best = max(others, key=lambda i: before[i])
    return float(heads[-1][0]) if sounds[best] == "none" else -np.inf


def fit(
    margins: np.ndarray, bands: np.ndarray, aims: list[int], starts: np.ndarray, cap: int | None
) -> np.ndarray:
    """One bias per band of rows, the smallest whose breath count is closest to the band's aim. Each band is fitted
    again given the others, as the cap ties a reply's bands together."""
    biases = np.zeros(len(aims))
    for _ in range(3):
        for k in range(len(aims)):

            def count(b: float, k: int = k) -> int:
                trial = biases.copy()
                trial[k] = b
                return int(breathes(margins, bands, trial, starts, cap)[bands == k].sum())

            biases[k] = min(BIASES, key=lambda b: (abs(count(b) - aims[k]), abs(b)))
    return biases


def breathes(
    margins: np.ndarray, bands: np.ndarray, biases: np.ndarray, starts: np.ndarray, cap: int | None
) -> np.ndarray:
    """Which rows breathe with their band's bias added to their breath margin, under the cap."""
    return capped(margins + biases[bands] > 0, starts, cap)


def capped(chosen: np.ndarray, starts: np.ndarray, cap: int | None) -> np.ndarray:
    """``chosen`` without each reply's breaths past ``cap``; ``starts`` marks each reply's first sentence."""
    if cap is None:
        return chosen
    total = np.cumsum(chosen)
    return chosen & (total - np.maximum.accumulate(np.where(starts, total - chosen, 0)) <= cap)


def calibrate(
    folder: Path,
    rows: list[Row],
    stats: dict | None = None,
    per_reply: int | None = None,
    kinds: dict[str, float] | None = None,
) -> dict:
    """Fit a model folder's breath bias with at most ``per_reply`` breaths to a reply, or no cap, and write both into
    its config, with the kinds a breath is drawn as and their shares, if given."""
    found = breath_bias(ToneTaggerModel(folder), rows, stats, per_reply)
    path = folder / "config.json"
    config = json.loads(path.read_text())
    for key, value in (("breaths_per_reply", per_reply), ("breath_kinds", kinds)):
        config.pop(key, None)
        if value is not None:
            config[key] = value
    config.update(sound_bias={"breath": found["bias"]}, breath_calibration=found)
    path.write_text(json.dumps(config, indent=2))
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("dev", type=Path, nargs="+")
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--per-reply", type=int)
    parser.add_argument(
        "--kinds", action="store_true", help="draw breaths as light, deep and exhaled, as people's"
    )
    args = parser.parse_args()
    rows = [r for path in args.dev for r in load_rows(path)]
    stats = summary(icsi_turns()) if args.speech else None
    print(calibrate(args.folder, rows, stats, args.per_reply, breath_kinds() if args.kinds else None))
