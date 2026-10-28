"""Where the reader's lines, labels and runs are on the data volume: the line classifier's label sets, pool and teachers'
labels, read as they are, and what this pipeline adds."""

# The line classifier's act and emotion label sets: emotion, intensity and backchannel per line. Their train splits train,
# their dev splits choose and calibrate, their test splits score.
ACT_EMOTION_SETS = (
    "linecls-corpus-0.1-thin-committee",
    "linecls-corpus-0.1-full-committee",
    "linecls-corpus-0.1-hardneg-committee",
    "linecls-corpus-0.1-heldout-committee",
    "linecls-corpus-0.2-speech-committee",
    "linecls-corpus-0.4-targeted-committee",
    "linecls-corpus-0.4-speechdev-committee",
    "linecls-corpus-0.9-targeted-committee",
    "linecls-sarcasm-0.2-dev-committee",
    "linecls-corpus-0.9b-targeted-committee",
    "linecls-chain-0.1-dev-committee",
)
# The act and emotion sets the line classifier was scored on (3,999 test lines).
ACT_TEST_SETS = ACT_EMOTION_SETS[:7]
POOL_ID = "linecls-pool-0.6"  # unlabelled lines, noisy twins included
SOFT_ID = (
    "linecls-soft-0.6"  # the line classifier's act and emotion teachers' averaged labels on its clean lines
)

# The move: the committee's labels and three Ettin-1B teachers' labels on a pool.
EXCHANGE_SET = "linecls-exchange-0.1-committee"
EXCHANGE_SOFT_ID = "linecls-exchange-soft-0.1"
EXCHANGE_TEACHERS = ("teacher-ettin-1b-seed0", "teacher-ettin-1b-seed1", "teacher-ettin-1b-seed2")

# Sarcasm: the committee's labels; each set's id carries "-sarcasm" on the volume.
SARCASM_SETS = {
    "train": ("linecls-sarcasm-train-0.1",),
    "dev": ("linecls-sarcasm-natural-dev-0.1", "linecls-sarcasm-0.2-dev", "linecls-chain-0.1-dev"),
    "test": ("linecls-sarcasm-natural-test-0.1", "linecls-sarcasm-0.2-test", "linecls-chain-0.1-test"),
}
NATURAL_SARCASM = (
    "linecls-sarcasm-natural-dev-0.1",
    "linecls-sarcasm-natural-test-0.1",
)  # false alarms are read here
HARD_SARCASM_TEST = ("linecls-sarcasm-0.2-test", "linecls-chain-0.1-test")  # F1 is read here
CHAIN_TEST = "linecls-chain-0.1-test"  # two lines in a row
SARCASM_SOFT_ID = "tonetext-sarcasm-soft-0.1"  # r1's sarcasm teachers
SARCASM_TEACHERS = tuple(f"teacher-ettin-1b-seed{k}" for k in range(3))
# Test slices scored apart from the main test set.
HELD_SLICES = ("unseen_genre", "player_speech")


def labels_key(label_id: str) -> str:
    return f"runs/linecls/labels/{label_id}/labelled.jsonl"


def sarcasm_key(set_id: str) -> str:
    return labels_key(f"{set_id}-sarcasm")


def pool_key(pool_id: str = POOL_ID) -> str:
    return f"runs/linecls/pool/{pool_id}/lines.jsonl"


def soft_key(soft_id: str = SOFT_ID) -> str:
    return f"runs/linecls/soft/{soft_id}/soft.jsonl"


def exchange_soft_key(name: str) -> str:
    return f"runs/linecls/soft/{EXCHANGE_SOFT_ID}/{name}.jsonl"


def sarcasm_soft_key(name: str) -> str:
    return f"runs/tonetext/soft/{SARCASM_SOFT_ID}/{name}.jsonl"


# --- r2 --------------------------------------------------------------------------------------------------------------
# The open conversation sets, each split by conversation; only their train splits reach the pool.
OPEN_TRAIN_SOURCES = (
    "natcs",
    "taskmaster1",
    "taskmaster2",
    "taskmaster3",
    "abcd",
    "multiwoz22",
    "star",
    "ccpem",
    "harpervalley",
    "ami",
    "dipco",
    "dynasent_r2",
    "isarcasmeval",
)
# Lines per source in the pool at most (a fixed draw), so one large source cannot swamp the rest.
OPEN_CAPS = {
    "natcs": 120_000,
    "taskmaster1": 80_000,
    "taskmaster2": 80_000,
    "taskmaster3": 60_000,
    "abcd": 80_000,
    "multiwoz22": 50_000,
    "star": 40_000,
    "ccpem": 15_000,
    "harpervalley": 25_000,
    "ami": 60_000,
    "dipco": 10_000,
    "dynasent_r2": 20_000,
    "isarcasmeval": 10_000,
}
# r2's committee labels: train, the voice-agent dev set and the real voice-agent test set, one id per question.
R2_SETS = {
    "committee": "tonetext-r2-committee",
    "exchange": "tonetext-r2-exchange",
    "sarcasm": "tonetext-r2-sarcasm",
}
SELECTION = "tonetext-r2-selection"  # the turns r2's committee labels
SPEECH_ID = "tonetext-speech-0.1"  # noisy twins of pool turns through text to speech and back
R2_SOFT_ID = "tonetext-r2-soft-0.1"  # the text-only teachers' labels on every training line
R2_TEACHERS = tuple(f"teacher-ettin-1b-seed{k}" for k in range(5))
# r1's generated voice-agent set, test only.
VOICE_ID = "tonetext-voice-0.1"
VOICE_GENERATED = {q: f"{VOICE_ID}-{q}" for q in ("committee", "exchange", "sarcasm")}
# The same sets by what the scorecard reads from each.
VOICE_SETS = {
    "emotion": VOICE_GENERATED["committee"],
    "move": VOICE_GENERATED["exchange"],
    "sarcasm": VOICE_GENERATED["sarcasm"],
}


# r3: r2 with the support-call set (``lab.reader.support``), written for rc1's misread of a turn sarcastic on its own
# words after a plain line and labelled under the Realtime Context Standard's perception; its own teachers.
SUPPORT_ID = "tonetext-support-0.1"
SUPPORT_SETS = {q: f"{SUPPORT_ID}-{q}" for q in ("committee", "sarcasm")}
R3_SOFT_ID = "tonetext-r3-soft-0.1"
# r3c: r3 with the contrast pairs (``support.select_contrast``), one turn of praise or thanks after two lines of the
# agent's, one it is meant after and one it is sarcastic after; read with r2's teachers.
CONTRAST_ID = "tonetext-support-contrast-0.1"
CONTRAST_SETS = {q: f"{CONTRAST_ID}-{q}" for q in ("committee", "sarcasm")}


def open_raw_key(source: str, name: str) -> str:
    return f"runs/tonetext/open-raw/{source}/{name}"


def open_key(source: str, name: str = "lines.jsonl") -> str:
    return f"runs/tonetext/open/{source}-0.1/{name}"


def tonetext_labels_key(set_id: str, name: str = "labelled.jsonl") -> str:
    return f"runs/tonetext/labels/{set_id}/{name}"


def voice_key(name: str) -> str:
    return f"runs/tonetext/generated/{VOICE_ID}/{name}"


def support_key(name: str, set_id: str = SUPPORT_ID) -> str:
    return f"runs/tonetext/generated/{set_id}/{name}"


def contrast_key(name: str) -> str:
    return support_key(name, CONTRAST_ID)


def speech_key(name: str = "lines.jsonl") -> str:
    return f"runs/tonetext/speech/{SPEECH_ID}/{name}"


def r2_soft_key(name: str, soft_id: str = R2_SOFT_ID) -> str:
    return f"runs/tonetext/soft/{soft_id}/{name}.jsonl"


def run_id(recipe: str, role: str, encoder: str, seed: int, tag: str = "") -> str:
    """A training run's id, as tonetext-r2-student-minilm-l6-seed1."""
    return "-".join(("tonetext", recipe, role, encoder, *([tag] if tag else []), f"seed{seed}"))


def run_key(run_id: str, *parts: str) -> str:
    """A training run's folder, or a file in it: its weights and results at the top, an export under its prefix."""
    return "/".join(("models/tonetext", run_id, *parts))


# --- as published ------------------------------------------------------------------------------------------------------
# Hugging Face holds the files that are ours, and stand-ins for those that hold others' text, which `fetch` rebuilds
# from the sources (``lab.reader.rebuild``).
MANIFEST_KEY = "rebuild/manifest.json"  # every file's sha256, and how each rebuilt one is made


def standin_key(key: str) -> str:
    return f"rebuild/standins/{key}"


UPSTREAM = "upstream"  # the sources of the rebuilt files, as downloaded


def upstream_key(rel: str) -> str:
    """A source file of the rebuilt files, at hf/<repo>@<revision>/<path> in the upstream folder."""
    return f"{UPSTREAM}/{rel}"
