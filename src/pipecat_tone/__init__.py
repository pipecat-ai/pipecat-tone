"""Pipecat processors for how a voice agent hears and sounds: the tone reader and the tone tagger.

They build on Pipecat's expressive conversation, not yet released: pipecat-ai/pipecat#6319.
"""

try:
    import pipecat.expressive  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError(
        "pipecat_tone needs Pipecat with expressive conversation (pipecat-ai/pipecat#6319); "
        "install Pipecat from that pull request's branch"
    ) from e
