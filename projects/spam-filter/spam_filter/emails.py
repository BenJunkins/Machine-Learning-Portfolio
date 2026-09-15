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
    """Removes quoted replies and swaps links/addresses for placeholder words.

    The ham in the dataset is mostly mailing list posts, so without this the
    model learns list footers and URLs instead of what the email actually says.
    """
    text = re.sub(r"^\s*>.*$", " ", text, flags=re.MULTILINE)
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
    number_failed = 0
    for path in ham_files + spam_files:
        try:
            X.append(text_from_file(path))
        except Exception:
            X.append("")
            number_failed += 1

    y = [0] * len(ham_files) + [1] * len(spam_files)
    print(f"Loaded {len(ham_files)} ham and {len(spam_files)} spam emails")
    if number_failed:
        print(f"{number_failed} files could not be read")

    return X, y


def split_emails(X: list, y: list):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
