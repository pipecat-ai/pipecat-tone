"""Standing pace: "slow down" sets the TTS's speed for the rest of the call, "you can go faster" resets it.

Each TTS names speed in its own settings type, imported only when needed.
"""

from loguru import logger
from pipecat.frames.frames import TTSUpdateSettingsFrame

from .patterns import FASTER, SLOWER

SLOW = 0.85
NORMAL = 1.0

# Services whose settings have no speed: a speed update would be dropped silently.
_NO_SPEED = ("dialogue",)


def requested_pace(text: str) -> float | None:
    """The speed a caller's words ask for, or None."""
    if FASTER.search(text):
        return NORMAL
    if SLOWER.search(text):
        return SLOW
    return None


def speed_frame(tts: str, speed: float, service: str) -> TTSUpdateSettingsFrame | None:
    """A settings update setting the TTS's speed, or None when the service can't take one."""
    if any(name in service.lower() for name in _NO_SPEED):
        logger.warning(f"tone tagger: {service} has no speed setting; pace requests are ignored")
        return None
    try:
        if tts == "cartesia":
            from pipecat.services.cartesia.tts import CartesiaTTSSettings, GenerationConfig

            delta = CartesiaTTSSettings(generation_config=GenerationConfig(speed=speed))
        elif tts == "elevenlabs":
            from pipecat.services.elevenlabs.tts import ElevenLabsTTSSettings

            delta = ElevenLabsTTSSettings(speed=speed)
        elif tts == "minimax":
            from pipecat.services.minimax.tts import MiniMaxTTSSettings

            delta = MiniMaxTTSSettings(speed=speed)
        elif tts == "fish":
            from pipecat.services.fish.tts import FishAudioTTSSettings

            delta = FishAudioTTSSettings(prosody_speed=speed)
        else:
            return None
    except ImportError as e:
        logger.warning(f"tone tagger: can't set {tts}'s speed ({e})")
        return None
    return TTSUpdateSettingsFrame(delta=delta)
