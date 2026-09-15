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


def html_to_text(content: str):
    content = re.sub(
        r"<(style|script)[^>]*>.*?</\1>", " ", content, flags=re.IGNORECASE | re.DOTALL
    )
    content = re.sub(r"<[^>]+>", " ", content)

    return html.unescape(content)


def clean_text(text: str):
    """Removes quoted replies and leftover MIME boundaries, and swaps
    links/addresses for placeholder words.

    The ham in the dataset is mostly mailing list posts, so without this the
    model learns list footers and URLs instead of what the email actually says.
    Some spam also has boundary lines like --DeathToSpamDeathToSpam-- left in
    the body, which the model would otherwise pick up as a spam word.
    """
    text = re.sub(r"^\s*>.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^--\S+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^.*\bwrote:\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"(https?://|www\.)\S+", " urllink ", text, flags=re.IGNORECASE)
    text = re.sub(r"\S+@\S+\.\w+", " emailaddress ", text)

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
    message = email.message_from_string(raw, policy=email.policy.default)
    header_names = {key.lower() for key in message.keys()}

    if header_names & COMMON_HEADERS:
        text = extract_text(message)
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
    labeled_files = [(path, 0) for path in ham_files] + [(path, 1) for path in spam_files]

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
