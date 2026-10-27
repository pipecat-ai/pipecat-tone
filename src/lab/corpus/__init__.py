"""The training corpus: synthetic calls with plain replies, labeled by the teacher.

Per call, Claude writes a scenario, a deployment LLM holds the call through a real Pipecat pipeline
(its replies come back plain), the teacher labels every sentence, and another vendor's model reviews
each reply. Against tuning to tone-bench: its domain families are left out, the writer never sees its
files, the reviewer isn't its judge's family, and splits are by domain.
"""
