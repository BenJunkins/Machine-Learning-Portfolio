"""

Checks that the fast text cleanup gives the same output as the cleanup the model was
trained with, and that input built to slow it down no longer can.


"""

import itertools
import random
import time

import original_cleanup as original
import pytest

from spam_filter import DATA_DIR, emails
from spam_filter.api import MAX_EMAIL_CHARACTERS

# Fragments the cleanup patterns react to, mixed in with ordinary characters
PIECES = list("aZ9_ \n\t\r<>/@.-:&#;=\"'") + [
    "  ", "\n\n", "\n \n", "> ", "--", "<>", "<!--", "-->",
    "<style", "<STYLE>", "<sTyle x>", "</style>", "</STYLE>", "<script>", "</script>",
    "wrote:", " wrote:", "http://", "HTTPS://", "www.", "a@b.c", ".x",
    "&amp;", "&#65;", "&lt;", " ", " ", "\x0b", "é", "ſ", "İ",
]  # fmt: skip

SECONDS_ALLOWED = 1.0


def random_text(rng: random.Random):
    longest = rng.choice([4, 12, 40])

    return "".join(rng.choice(PIECES) for _ in range(rng.randint(0, longest)))


def every_string(alphabet: str, longest: int):
    for length in range(longest + 1):
        for characters in itertools.product(alphabet, repeat=length):
            yield "".join(characters)


def fill(piece: str):
    """Repeats piece until it is exactly as long as the biggest email the API accepts"""
    return (piece * (MAX_EMAIL_CHARACTERS // len(piece) + 1))[:MAX_EMAIL_CHARACTERS]


def check_same_as_original(text: str):
    assert emails.html_to_text(text) == original.html_to_text(
        text
    ), f"html_to_text differs for {text!r}"
    assert emails.clean_text(text) == original.clean_text(
        text
    ), f"clean_text differs for {text!r}"


@pytest.mark.parametrize(
    "alphabet", ["a> \n\t", "a.@ -", "a<> /"], ids=["quoted lines", "addresses", "tags"]
)
def test_same_as_original_on_every_short_string(alphabet):
    # Every possible string up to 7 characters long, so a mistake shows up as a tiny example
    for text in every_string(alphabet, 7):
        check_same_as_original(text)


def test_same_as_original_on_random_text():
    rng = random.Random(2026)

    for _ in range(50_000):
        check_same_as_original(random_text(rng))


def result_or_error(path):
    try:
        return emails.text_from_file(path)
    except Exception as error:
        return type(error)


needs_dataset = pytest.mark.skipif(
    not (DATA_DIR / "ham").is_dir(),
    reason="needs the dataset: python -m spam_filter.download",
)


@needs_dataset
def test_same_as_original_on_the_training_emails(monkeypatch):
    paths = sorted(
        path for folder in ("ham", "spam") for path in (DATA_DIR / folder).iterdir()
    )
    fast = [result_or_error(path) for path in paths]

    monkeypatch.setattr(emails, "html_to_text", original.html_to_text)
    monkeypatch.setattr(emails, "clean_text", original.clean_text)
    slow = [result_or_error(path) for path in paths]

    assert len(paths) > 6000
    assert [
        path.name
        for path, new, old in zip(paths, fast, slow, strict=True)
        if new != old
    ] == []


STYLE_SPELLINGS = [
    "".join(letters)
    for letters in itertools.product(*[(c, c.upper()) for c in "style"])
]

CRAFTED = {
    "ordinary sentences": fill("the quick brown fox jumps over the lazy dog. "),
    "one long word": fill("a"),
    "only < characters": fill("<"),
    "<style> never closed": fill("<style>"),
    "<style> and <script> never closed": fill("<style><script>"),
    "32 spellings of <style>": fill(
        "".join(f"<{spelling}>" for spelling in STYLE_SPELLINGS)
    ),
    "blank lines": "a" + fill("\n")[2:] + "b",
    "lines of spaces": "a\n" + fill(" \n")[3:] + "b",
    "wrote: then blank lines": "a wrote:" + fill("\n")[9:] + "b",
    "wrote: repeated": fill("wrote: "),
    "a@a@a@": fill("a@"),
    "a@ words": fill("a@ "),
    "a@ then dots": "a@" + fill(".")[2:],
    "a@ then .a.a.a": "a@" + fill(".a")[2:],
    "boundary lines": fill("--x" + " " * 50 + "y\n"),
    "http:// repeated": fill("http://"),
    "& characters": fill("&"),
    "&# and digits": "&#" + fill("1")[3:] + ";",
    "many MIME headers": fill("To: b\n"),
}


@pytest.mark.parametrize("text", CRAFTED.values(), ids=CRAFTED.keys())
def test_crafted_input_is_fast(text):
    assert len(text) <= MAX_EMAIL_CHARACTERS

    started = time.perf_counter()
    emails.text_from_paste(text)
    seconds = time.perf_counter() - started

    assert seconds < SECONDS_ALLOWED


def test_deeply_nested_email_is_read_as_plain_text():
    # More nesting than Python's email parser can follow
    nested = "".join(
        f"Content-Type: multipart/mixed; boundary=b{i}\n\n--b{i}\n" for i in range(1500)
    )

    assert isinstance(emails.text_from_paste(nested + "hello"), str)
