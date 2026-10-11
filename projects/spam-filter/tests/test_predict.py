import pytest

from spam_filter.predict import UNSURE_HIGH, UNSURE_LOW, label_for


@pytest.mark.parametrize(
    ("probability", "label"),
    [
        (0.0, "ham"),
        (0.39, "ham"),
        (UNSURE_LOW, "unsure"),
        (0.59, "unsure"),
        (UNSURE_HIGH, "spam"),
        (1.0, "spam"),
    ],
)
def test_label_for(probability, label):
    assert label_for(probability) == label
