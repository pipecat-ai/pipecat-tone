import pytest
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.services.minimax.tts import MiniMaxHttpTTSService

from pipecat_tone.tagger.pace import NORMAL, SLOW, requested_pace, speed_frame

# Each TTS's service, and where its settings keep the speed.
SERVICES = [
    ("cartesia", CartesiaTTSService, lambda d: d.generation_config.speed),
    ("elevenlabs", ElevenLabsTTSService, lambda d: d.speed),
    ("fish", FishAudioTTSService, lambda d: d.prosody_speed),
    ("minimax", MiniMaxHttpTTSService, lambda d: d.speed),
]


@pytest.mark.parametrize("tts,service,speed", SERVICES)
def test_speed_is_the_services_own_setting(tts, service, speed):
    frame = speed_frame(tts, SLOW, f"{service.__name__}#0")
    assert isinstance(frame.delta, service.Settings)
    assert speed(frame.delta) == SLOW


def test_a_service_without_speed_gets_no_frame():
    assert speed_frame("elevenlabs", SLOW, "ElevenLabsDialogueTTSService#0") is None


def test_requested_pace():
    assert requested_pace("Whoa, slow down a sec.") == SLOW
    assert requested_pace("You can go faster now.") == NORMAL
    assert requested_pace("Say that again?") is None
