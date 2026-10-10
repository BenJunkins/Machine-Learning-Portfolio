"""

Loading the saved model and scoring an email.
Shared by the popup and the web API.

"""

import joblib

from spam_filter import MODEL_PATH
from spam_filter.emails import text_from_paste

# Newer emails often score in this range because the
# training data is from 2002-2005, so it's reported as unsure instead of a verdict
UNSURE_LOW = 0.4
UNSURE_HIGH = 0.6


def load_model():
    model = joblib.load(MODEL_PATH)
    model.set_params(
        randomforestclassifier__n_jobs=1
    )  # Only one email is scored at a time.

    return model


def spam_probability(model, raw_email: str):
    spam_class_index = list(model.classes_).index(1)
    text = text_from_paste(raw_email)

    return model.predict_proba([text])[0][spam_class_index]


def label_for(probability: float):
    if probability >= UNSURE_HIGH:
        return "spam"
    if probability >= UNSURE_LOW:
        return "unsure"

    return "ham"
