"""Downloads the SpamAssassin public corpus into data/ham and data/spam."""

import shutil
import tarfile
import urllib.request
from pathlib import Path

from spam_filter import DATA_DIR

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "ham": [
        "20030228_easy_ham.tar.bz2",
        "20030228_easy_ham_2.tar.bz2",
        "20030228_hard_ham.tar.bz2",
    ],
    "spam": [
        "20030228_spam.tar.bz2",
        "20050311_spam_2.tar.bz2",
    ],
}


def download(data_dir: Path = DATA_DIR):
    for label, archives in ARCHIVES.items():
        out_dir = data_dir / label
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"{out_dir} already has emails, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        for name in archives:
            print(f"Downloading {name}...")
            with urllib.request.urlopen(BASE_URL + name) as response:
                with tarfile.open(fileobj=response, mode="r|bz2") as tar:
                    for member in tar:
                        filename = Path(member.name).name
                        # Each archive has a "cmds" file that isn't an email
                        if not member.isfile() or filename == "cmds":
                            continue
                        with tar.extractfile(member) as src:
                            with open(out_dir / filename, "wb") as dst:
                                shutil.copyfileobj(src, dst)

    for label in ARCHIVES:
        count = len(list((data_dir / label).iterdir()))
        print(f"{label}: {count} emails")


if __name__ == "__main__":
    download()
