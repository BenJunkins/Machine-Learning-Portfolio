"""

The text cleanup exactly as it was when the model was trained.

Nothing in the app uses this file. It is kept so the tests can check that the faster cleanup
in spam_filter/emails.py still gives the same output, which is what keeps the saved model valid.
Don't edit it.

"""

import html
import re


def html_to_text(content: str):
    content = re.sub(
        r"<(style|script)[^>]*>.*?</\1>", " ", content, flags=re.IGNORECASE | re.DOTALL
    )
    content = re.sub(r"<[^>]+>", " ", content)

    return html.unescape(content)


def clean_text(text: str):
    text = re.sub(r"^\s*>.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^--\S+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^.*\bwrote:\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"(https?://|www\.)\S+", " urllink ", text, flags=re.IGNORECASE)
    text = re.sub(r"\S+@\S+\.\w+", " emailaddress ", text)

    return text
