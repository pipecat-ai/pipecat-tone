"""The reader's calibration, fitted on dev by the export and applied by the runtime: a temperature for emotion and one
for the move, and a Platt curve for sarcasm. Every map rises, so no decision changes, only the numbers reported."""

import math

import numpy as np

from pipecat_tone.reader.model import platt, softmax  # noqa: F401

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def fit_platt(p, y, l2: float = 1e-3, iters: int = 100) -> dict:
    """Two parameters by Newton's method on the log loss of y (0 or 1) given logit(p); a little L2 for stability."""
    x = np.array([logit(v) for v in p], dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = q * (1 - q)
        g = np.array([np.sum((q - y) * x) + l2 * (a - 1.0), np.sum(q - y) + l2 * b])
        h = np.array([[np.sum(w * x * x) + l2, np.sum(w * x)], [np.sum(w * x), np.sum(w) + l2]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"a": round(float(a), 6), "b": round(float(b), 6), "items": int(len(x))}


def fit_temperature(logits, y) -> float:
    """The temperature minimising the log loss of the gold classes: a coarse search over 0.05 to 10, then a finer
    one."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    def nll(t: float) -> float:
        p = softmax(z, t)[np.arange(len(y)), y]
        return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

    best = min(np.exp(np.linspace(np.log(0.05), np.log(10.0), 200)), key=nll)
    best = min(np.linspace(best / 1.05, best * 1.05, 101), key=nll)
    return round(float(best), 4)


def ece(conf, correct, bins: int = 10) -> dict:
    """Is a confidence of 0.8 right about 80% of the time? Expected calibration error over equal-width bins, with the
    bins; `correct` is 0 or 1 per item."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if not len(conf):
        return {"ece": None, "items": 0, "bins": []}
    edges = np.linspace(0.0, 1.0, bins + 1)
    out, err = [], 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        if m.any():
            c, a = float(conf[m].mean()), float(correct[m].mean())
            out.append(
                {
                    "from": round(float(lo), 2),
                    "to": round(float(hi), 2),
                    "items": int(m.sum()),
                    "confidence": round(c, 4),
                    "right": round(a, 4),
                }
            )
            err += m.sum() / len(conf) * abs(a - c)
    return {
        "ece": round(float(err), 4),
        "items": int(len(conf)),
        "brier": round(float(np.mean((conf - correct) ** 2)), 4),
        "bins": out,
    }


def ece_top(probs, y) -> float:
    """The calibration error of the top probability as the chance the top class is right."""
    p = np.asarray(probs)
    return ece(p.max(1).tolist(), (p.argmax(1) == np.asarray(y)).astype(int).tolist())["ece"]
