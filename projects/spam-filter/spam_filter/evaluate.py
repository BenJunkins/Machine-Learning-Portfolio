"""Scores the saved model on the test set and plots the results."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
)

from spam_filter import MODEL_PATH, ROOT_DIR
from spam_filter.emails import get_emails, split_emails

PLOT_PATH = ROOT_DIR / "images" / "evaluation.png"
beta = 0.5


def plot_model_evaluation(model, X_test, y_test):
    """Prints a precision/recall table and shows a 2x2 grid of evaluation plots"""
    y_pred = model.predict(X_test)

    spam_class_index = np.where(model.classes_ == 1)[0][0]
    spam_prob = model.predict_proba(X_test)[:, spam_class_index]

    precision, recall, fbeta, support = precision_recall_fscore_support(
        y_test, y_pred, beta=beta
    )

    report_df = pd.DataFrame(
        {
            "Precision": precision,
            "Recall": recall,
            f"F{beta}-Score": fbeta,
            "Support": support,
        },
        index=pd.Index(["Ham (0)", "Spam (1)"]),
    )
    print(report_df.round(3))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax[0, 0],
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"],
        cbar=False,
    )
    ax[0, 0].set_title("Confusion Matrix", fontsize=14)
    ax[0, 0].set_xlabel("Predicted Label", fontsize=12)
    ax[0, 0].set_ylabel("True Label", fontsize=12)

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test, spam_prob, ax=ax[0, 1], name="Random Forest"
    )
    ax[0, 1].set_title("Precision-Recall Curve", fontsize=14)

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test, spam_prob, ax=ax[1, 0], name="Random Forest"
    )
    ax[1, 0].set_title("ROC Curve", fontsize=14)
    ax[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)

    # Top Feature Bar Chart
    vectorizer = model.named_steps["tfidfvectorizer"]
    rf_classifier = model.named_steps["randomforestclassifier"]
    feat_df = pd.DataFrame(
        {
            "Feature": vectorizer.get_feature_names_out(),
            "Importance": rf_classifier.feature_importances_,
        }
    )
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(20)

    sns.barplot(
        data=feat_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax[1, 1],
    )
    ax[1, 1].set_title("Top 20 Most Important Words", fontsize=14)
    ax[1, 1].set_xlabel("Mean Decrease in Impurity", fontsize=12)
    ax[1, 1].set_ylabel("")

    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=100)
    print(f"Plots saved to {PLOT_PATH}")
    plt.show()


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run: python -m spam_filter.train")

    model = joblib.load(MODEL_PATH)
    X, y = get_emails()
    _, X_test, _, y_test = split_emails(X, y)
    plot_model_evaluation(model, X_test, y_test)


if __name__ == "__main__":
    main()
