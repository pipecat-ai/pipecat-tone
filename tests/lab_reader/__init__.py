"""Made-up lines in the shapes the volume holds."""


def open_line(
    line_id: str,
    text: str,
    source: str = "natcs",
    role: str = "user",
    modality: str = "spoken",
    split: str = "train",
    previous: str | None = "Agent: How can I help?",
    group: str = "c1",
    domain: str = "banking",
    native: dict | None = {"scheme": "x", "label": "y"},  # noqa: B006
    ref: str = "f.json#1",
    licence: str = "CC BY 4.0",
    participants: list[str] | None = None,
) -> dict:
    """An open set's line: the user speaks as `player` to `Agent`, or among `participants`, the first speaking."""
    if participants:
        speaker, to, targets = participants[0], None, participants[1:]
    else:
        speaker = "player" if role == "user" else "Agent"
        to = "Agent" if speaker == "player" else "player"
        targets = [to]
    return {
        "line_id": line_id,
        "source": source,
        "source_ref": ref,
        "licence": licence,
        "split": split,
        "group_key": f"{source}:{group}",
        "input": {
            "speaker": speaker,
            "speaker_kind": "player" if speaker == "player" else "npc",
            "to": to,
            "targets": targets,
            "text": text,
            "previous": previous,
        },
        "slices": [source, role, modality, domain],
        "twin_of": None,
        "generator": None,
        "native": native,
    }
