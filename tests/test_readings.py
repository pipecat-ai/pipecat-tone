from pipecat.expressive.perception import PerceptionSignal

from pipecat_tone.tagger import readings


def test_parse_reading():
    heard, words = readings.parse("[heard: tone frustrated 0.86, neutral 0.10; sarcasm 0.04] What the hell.")
    assert words == "What the hell."
    assert heard[0].labels == [("frustrated", 0.86), ("neutral", 0.10)]
    assert heard[1].value == 0.04
    assert readings.parse("No reading.") == (None, "No reading.")


def test_pipecat_tone_readings_fold_into_tone_labels():
    heard, _ = readings.parse("[heard: emotion angry 0.62, contemptuous 0.21; sarcasm 0.7] Great.")
    assert heard[0].name == "tone" and heard[0].labels == [("frustrated", 0.83)]
    assert heard[1].name == "sarcasm" and heard[1].value == 0.7


def test_reading_in_bands():
    heard = [
        PerceptionSignal("tone", labels=[("frustrated", 0.86), ("neutral", 0.55), ("sad", 0.1)]),
        PerceptionSignal("sarcasm", value=0.04),
    ]
    assert readings.describe(heard) == "tone frustrated clear, neutral likely, sad guess; sarcasm low"
    assert readings.describe(None) == "no reading"
