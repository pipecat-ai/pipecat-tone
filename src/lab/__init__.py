"""Tooling for the tone models: corpus, training, tone-bench runs and Modal jobs. Not shipped.

tone-bench, Pipecat's private benchmark, is imported read-only from its checkout, ``$TONE_BENCH`` or
``../tone-bench``, as ``bench``. Only building the tagger's corpus and benchmark runs need it.
"""

import os
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TONE_BENCH = Path(os.environ.get("TONE_BENCH", ROOT.parent / "tone-bench")).resolve()
if str(TONE_BENCH) not in sys.path:
    sys.path.insert(0, str(TONE_BENCH))


def quiet_logs():
    """Errors from everything; warnings from this repo's code only, not Pipecat's."""
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    logger.add(
        sys.stderr,
        level="WARNING",
        filter=lambda r: r["name"].startswith(("bench.", "pipecat_tone.", "lab.")) and r["level"].no < 40,
    )
