"""How close a generated scenario comes to any of tone-bench's, by shared word 5-grams.

tone-bench's scenarios are read only here, never shown to the writer. A score is the share of the
generated scenario's 5-grams found in a tone-bench scenario: near 0 for unrelated calls, high for a copy.
"""

import re
from functools import cache

import yaml

N = 5


def closest(raw: dict) -> tuple[float, str]:
    """The highest overlap with any tone-bench scenario, and that scenario's name."""
    grams = _grams(_text(raw))
    if not grams:
        return 0.0, ""
    return max((len(grams & theirs) / len(grams), name) for name, theirs in _tone_bench().items())


@cache
def _tone_bench() -> dict[str, frozenset]:
    from lab.bench import SCENARIOS

    return {
        p.stem: frozenset(_grams(_text(yaml.safe_load(p.read_text()))))
        for p in sorted(SCENARIOS.glob("*.yaml"))
    }


def _text(raw: dict) -> str:
    return " ".join([raw.get("system_prompt", ""), *(t.get("user", "") for t in raw.get("turns", []))])


def _grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i : i + N]) for i in range(len(words) - N + 1)}
