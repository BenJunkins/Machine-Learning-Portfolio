"""Trains the spam classifier and saves it to models/spam_classifier.pkl."""

import joblib
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

from spam_filter import MODEL_PATH
from spam_filter.emails import get_emails, split_emails

# Below 1 weights precision over recall. Flagging a real email as spam is worse
# than letting a spam email through.
BETA = 0.5


def model():
    return make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(n_jobs=-1, random_state=42),
    )


def model_optimization(model, X: list, y: list):
    print("Searching for the best parameters...")
    param_dist = {
        "tfidfvectorizer__max_df": stats.uniform(0.7, 0.3),
        "tfidfvectorizer__min_df": stats.randint(2, 7),
        "tfidfvectorizer__sublinear_tf": [True, False],
        "randomforestclassifier__n_estimators": stats.randint(200, 800),
        "randomforestclassifier__max_depth": [None, 50, 100],
        "randomforestclassifier__min_samples_split": stats.randint(2, 10),
        "randomforestclassifier__min_samples_leaf": stats.randint(1, 4),
        "randomforestclassifier__max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring=make_scorer(fbeta_score, beta=BETA),
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X, y)

    print(f"Best cross-validation F{BETA}: {search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  - {param}: {value}")

    return search.best_estimator_


def create_model():
    X, y = get_emails()
    X_train, X_test, y_train, y_test = split_emails(X, y)

    best_model = model_optimization(model(), X_train, y_train)

    print("\nTest set results:")
    print(
        classification_report(
            y_test, best_model.predict(X_test), target_names=["Ham", "Spam"], digits=3
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    print(f"Model saved to {MODEL_PATH}")

    return best_model


if __name__ == "__main__":
    create_model()
