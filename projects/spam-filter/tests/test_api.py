from spam_filter.api import MAX_EMAIL_CHARACTERS
from spam_filter.predict import label_for

SPAM = """Dear Friend,

Congratulations! You have been selected to receive a FREE $1000 gift card. This is a limited
time offer and you must act now to claim your prize.

Click here to claim your free gift: http://www.example-prizes.com/claim

No purchase necessary. 100% guaranteed. To be removed from our mailing list, reply with REMOVE
in the subject line.
"""

HAM = """Hi all,

I upgraded the kernel on my laptop last night and now the wireless card isn't detected at boot.
I checked dmesg and the module loads without errors, but ifconfig doesn't show the interface.

Has anyone else run into this? I'm running Red Hat 8 with the stock drivers. I can post the
full log if that helps.

Thanks,
Mark
"""

# The training emails are from 2002-2005, so a modern automated message lands in the middle
UNSURE = (
    "Your password was changed. If you did not make this change, visit your account settings "
    "to secure your account. This is an automated message, please do not reply."
)

RAW_SPAM = """From: "Prize Dept" <winner@example.com>
To: you@example.org
Subject: You are a winner
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html><body><h1>Congratulations!</h1>
<p>You have been selected to receive a <b>FREE</b> $1000 gift card. This is a limited time
offer and you must act now to claim your prize.</p>
<p><a href="http://www.example-prizes.com/claim">Click here to claim your free gift</a></p>
<p>No purchase necessary. 100% guaranteed. To be removed from our mailing list, reply with
REMOVE in the subject line.</p>
</body></html>
"""


def score(client, email):
    response = client.post("/predict", json={"email": email})
    assert response.status_code == 200

    return response.json()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_spam_is_labeled_spam(client):
    result = score(client, SPAM)

    assert result["label"] == "spam"
    assert result["spam_probability"] >= 0.6


def test_ham_is_labeled_ham(client):
    result = score(client, HAM)

    assert result["label"] == "ham"
    assert result["spam_probability"] < 0.4


def test_modern_email_is_labeled_unsure(client):
    result = score(client, UNSURE)

    assert result["label"] == "unsure"
    assert 0.4 <= result["spam_probability"] < 0.6


def test_raw_email_with_headers_is_accepted(client):
    result = score(client, RAW_SPAM)

    assert result["label"] == "spam"


def test_response_has_only_the_documented_fields(client):
    result = score(client, HAM)

    assert set(result) == {"spam_probability", "label"}
    assert 0 <= result["spam_probability"] <= 1
    assert result["label"] == label_for(result["spam_probability"])


def test_blank_email_is_rejected(client):
    response = client.post("/predict", json={"email": "   \n\t  "})

    assert response.status_code == 422


def test_missing_email_is_rejected(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_oversized_email_is_rejected(client):
    response = client.post("/predict", json={"email": "a" * (MAX_EMAIL_CHARACTERS + 1)})

    assert response.status_code == 422


def test_email_at_the_size_limit_is_accepted(client):
    # Ordinary words, repeated and cut to exactly the limit
    email = ("see you at the meeting on thursday " * MAX_EMAIL_CHARACTERS)[
        :MAX_EMAIL_CHARACTERS
    ]

    response = client.post("/predict", json={"email": email})

    assert response.status_code == 200
