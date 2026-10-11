"""Reading emails and pulling out the text the model is trained on."""

import email
import email.policy
import html
import re
from email.message import EmailMessage
from pathlib import Path

from sklearn.model_selection import train_test_split

from spam_filter import DATA_DIR

# If a pasted email has any of these, treat it as a full raw email with headers
COMMON_HEADERS = {
    "from",
    "to",
    "subject",
    "date",
    "received",
    "return-path",
    "delivered-to",
    "message-id",
    "mime-version",
    "content-type",
}


# The cleanup the model was trained with is kept in tests/original_cleanup.py, and the
# tests check that this file gives the same output
STYLE_OR_SCRIPT_BLOCK = re.compile(
    r"<(style|script)[^>]*>.*?</\1>", flags=re.IGNORECASE | re.DOTALL
)
STYLE_OR_SCRIPT_OPENING = re.compile(r"<(style|script)", flags=re.IGNORECASE)
TAG = re.compile(r"<[^>]+>")
BOUNDARY_LINE = re.compile(r"^--\S+\s*$", flags=re.MULTILINE)
REPLY_HEADER = re.compile(r"^.*\bwrote:\s*$", flags=re.MULTILINE)
LINK = re.compile(r"(https?://|www\.)\S+", flags=re.IGNORECASE)

# A run of non-space characters with an @ in it. The lookbehind only lets a match
# start at the beginning of a run, so each run is read once
WORD_WITH_AT = re.compile(r"(?<!\S)[^\s@]*@\S*")
LAST_DOT_BEFORE_A_WORD = re.compile(r"\S*\.(?=\w)")
WORD_CHARACTERS = re.compile(r"\w+")


def remove_style_and_script(content: str):
    """Removes <style> and <script> blocks, contents included.

    Gives the same result as STYLE_OR_SCRIPT_BLOCK.sub(" ", content). The difference is what
    happens when an opening tag is never closed. sub() searches to the end of the text again
    for every one of those, and this gives up on a tag name the first time that search fails.
    """
    kept = []
    position = 0
    never_closed = set()

    for opening in STYLE_OR_SCRIPT_OPENING.finditer(content):
        name = opening.group(1).lower()
        if opening.start() < position or name in never_closed:
            continue

        block = STYLE_OR_SCRIPT_BLOCK.match(content, opening.start())
        if block is None:
            never_closed.add(name)
            continue

        kept.append(content[position : block.start()])
        kept.append(" ")
        position = block.end()

    kept.append(content[position:])

    return "".join(kept)


def html_to_text(content: str):
    content = remove_style_and_script(content)

    # A tag has to end with ">", so nothing after the last one can be a tag. Stopping there
    # keeps the pattern from reading to the end of the text for every "<" that is never closed
    end_of_tags = content.rfind(">") + 1
    content = TAG.sub(" ", content[:end_of_tags]) + content[end_of_tags:]

    try:
        return html.unescape(content)
    except ValueError:
        # A numeric reference such as &#65; with thousands of digits, which Python won't convert
        return content


def drop_quoted_lines(text: str):
    """Replaces each quoted line, one whose first non-space character is >, with a single space.

    Gives the same result as the pattern ^\\s*>.*$ did. In that pattern \\s also matches line
    breaks, so blank lines directly above a quoted line were replaced along with it. The pattern
    found them by reading ahead from every blank line, and this reads each line once.
    """
    kept = []
    blank_lines = []

    for line in text.split("\n"):
        start = line.lstrip()

        if not start:
            # Not known yet whether a quoted line follows, so hold it back
            blank_lines.append(line)
        elif start[0] == ">":
            blank_lines.clear()
            kept.append(" ")
        else:
            kept.extend(blank_lines)
            blank_lines.clear()
            kept.append(line)

    kept.extend(blank_lines)

    return "\n".join(kept)


def replace_address(match: re.Match):
    """Swaps the email address in a run of non-space characters for a placeholder word.

    Gives the same result as the pattern \\S+@\\S+\\.\\w+ did. That pattern needs a character
    before the @, at least one between the @ and a later dot, and a letter, digit or underscore
    after that dot. It reaches to the last dot that qualifies, and anything after the word that
    follows it, such as a closing bracket, is kept.
    """
    word = match.group()
    last_dot = LAST_DOT_BEFORE_A_WORD.match(word)
    if last_dot is None:
        return word

    dot = last_dot.end() - 1
    if dot < 3 or word.find("@", 1, dot - 1) == -1:
        return word

    end_of_address = WORD_CHARACTERS.match(word, dot + 1).end()

    return " emailaddress " + word[end_of_address:]


def clean_text(text: str):
    """Removes quoted replies and leftover MIME boundaries, and swaps
    links/addresses for placeholder words.

    The ham in the dataset is mostly mailing list posts, so without this the
    model learns list footers and URLs instead of what the email actually says.
    Some spam also has boundary lines like --DeathToSpamDeathToSpam-- left in
    the body, which the model would otherwise pick up as a spam word.
    """
    text = drop_quoted_lines(text)
    text = BOUNDARY_LINE.sub(" ", text)
    text = REPLY_HEADER.sub(" ", text)
    text = LINK.sub(" urllink ", text)
    text = WORD_WITH_AT.sub(replace_address, text)

    return text


def extract_text(message: EmailMessage):
    """Returns the plain text and HTML parts of an email body as one string"""
    body_parts = []

    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue

        content_type = part.get_content_type()
        if content_type not in ["text/plain", "text/html"]:
            continue

        try:
            content = part.get_content()
        except (LookupError, UnicodeDecodeError, ValueError):
            raw_payload = part.get_payload(decode=True)
            if not isinstance(raw_payload, bytes):
                continue
            content = raw_payload.decode("utf-8", errors="ignore")
        except Exception:
            continue

        if content_type == "text/html":
            content = html_to_text(content)

        body_parts.append(content)

    return " ".join(body_parts)


def text_from_file(path: Path):
    with open(path, "rb") as file:
        message = email.message_from_binary_file(file, policy=email.policy.default)

    return clean_text(extract_text(message))


def text_from_paste(raw: str):
    """Accepts either a full raw email (with headers) or just the body text"""
    raw = raw.strip()

    try:
        message = email.message_from_string(raw, policy=email.policy.default)
        header_names = {key.lower() for key in message.keys()}
        text = extract_text(message) if header_names & COMMON_HEADERS else ""
    except RecursionError:
        # About a thousand MIME parts nested inside each other, which is more than Python's
        # email parser can follow. No real email looks like that, so read it as plain text
        text = ""

    if text.strip():
        return clean_text(text)

    return clean_text(html_to_text(raw))


def get_emails(data_dir: Path = DATA_DIR):
    """Returns email texts and labels (0 = ham, 1 = spam)"""
    ham_dir = data_dir / "ham"
    spam_dir = data_dir / "spam"
    if not ham_dir.is_dir() or not spam_dir.is_dir():
        raise SystemExit(
            f"No emails found in {data_dir}. Run: python -m spam_filter.download"
        )

    ham_files = sorted(ham_dir.iterdir())
    spam_files = sorted(spam_dir.iterdir())

    X = []
    y = []
    seen = set()
    number_failed = number_empty = number_duplicates = 0
    labeled_files = [(path, 0) for path in ham_files] + [
        (path, 1) for path in spam_files
    ]

    for path, label in labeled_files:
        try:
            text = text_from_file(path)
        except Exception:
            number_failed += 1
            continue

        # Copies of the same email could land in both the train and test sets
        # and make the test scores look better than they are
        key = " ".join(text.split()).lower()
        if not key:
            number_empty += 1
            continue
        if key in seen:
            number_duplicates += 1
            continue
        seen.add(key)

        X.append(text)
        y.append(label)

    print(f"Loaded {y.count(0)} ham and {y.count(1)} spam emails")
    print(
        f"Skipped {number_duplicates} duplicates, {number_empty} emails with no text "
        f"and {number_failed} files that could not be read"
    )

    return X, y


def split_emails(X: list, y: list):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
