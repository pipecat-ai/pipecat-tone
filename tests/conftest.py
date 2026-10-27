import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def rc1() -> dict:
    """Pipecat Tone rc1's readings from Pipecat's own runtime, frozen before Pipecat dropped it."""
    return json.loads((Path(__file__).parent / "data" / "tone_rc1.json").read_text())
