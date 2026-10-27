"""Patterns shared by the rules, the model's state hints and pace."""

import re

# Details someone might write down: a code with letters and digits, five or more digits
# (with dashes or spaces), characters spelled out one by one, an email.
DETAILS = re.compile(
    r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b"
    r"|(?<![\d$.,])\d(?:[\d\- ]*\d){4,}(?![\d.,]\d)"
    r"|\b(?:[A-Za-z0-9][ -]){3,}[A-Za-z0-9]\b"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
)

# The model's state hints were trained on these exact patterns.
APOLOGY = re.compile(r"\b(sorry|apologi[sz]e)", re.IGNORECASE)
SLOW_OR_REPEAT = re.compile(
    r"\b(slow(er)? down|slower|say (that|it) again|repeat|didn't (catch|get) (that|it)|one more time|come again)\b",
    re.IGNORECASE,
)

# The rules.
SORRY = re.compile(r"\b(sorry|apologi[sz]e|unfortunately|i'm afraid|bad news)\b", re.IGNORECASE)
GOOD_NEWS = re.compile(
    r"\b(great news|good news|congratulations|wonderful|you're all set|that's great)\b", re.IGNORECASE
)
JOKE = re.compile(r"\b(ha(ha)+|ha ha|lol|just kidding|kidding|joking)\b", re.IGNORECASE)

# Pace.
SLOWER = re.compile(
    r"\b(slow(er)? down|(speak|talk|go) (a (little|bit) )?(more )?slow(ly|er)|too fast|so fast|slower please)\b",
    re.IGNORECASE,
)
FASTER = re.compile(
    r"\b(speed up|(speak|talk|go) (a (little|bit) )?faster|you can go faster|normal speed)\b", re.IGNORECASE
)

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip


def normalize(text: str) -> str:
    """Letters and digits only, spoken digits as digits: "seven Q two" is "7q2"."""
    return "".join(_DIGIT_WORDS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower()))
