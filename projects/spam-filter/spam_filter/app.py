"""A small window to paste an email into and see how likely it is to be spam."""

import ctypes
import tkinter as tk
from tkinter import messagebox, ttk

import joblib

from spam_filter import MODEL_PATH
from spam_filter.emails import text_from_paste

SPAM_COLOR = "#c62828"
UNSURE_COLOR = "#b26a00"
HAM_COLOR = "#2e7d32"

# Newer emails, real and spam alike, often score in this range because the
# training data is from 2002-2005, so it's shown as unsure instead of a verdict
UNSURE_LOW = 0.4
UNSURE_HIGH = 0.6


def load_model():
    model = joblib.load(MODEL_PATH)
    # Only one email is scored at a time, so extra threads just add overhead
    model.set_params(randomforestclassifier__n_jobs=1)

    return model


def spam_probability(model, raw_email: str):
    spam_class_index = list(model.classes_).index(1)
    text = text_from_paste(raw_email)

    return model.predict_proba([text])[0][spam_class_index]


class SpamFilterApp:
    def __init__(self, root: tk.Tk, model):
        self.model = model

        root.title("Spam Filter")
        root.minsize(420, 320)

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Paste an email below:").pack(anchor="w")

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, pady=(4, 8))
        self.text = tk.Text(
            text_frame, width=70, height=18, wrap="word", undo=True, font=("Segoe UI", 10)
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        bottom = ttk.Frame(frame)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Check", command=self.check).pack(side="left")
        ttk.Button(bottom, text="Clear", command=self.clear).pack(
            side="left", padx=(6, 0)
        )
        self.result = tk.Label(bottom, text="", font=("Segoe UI", 13, "bold"))
        self.result.pack(side="right")

        self.text.bind("<Control-Return>", self.check)
        self.text.focus_set()

    def check(self, event=None):
        raw_email = self.text.get("1.0", "end")
        if not raw_email.strip():
            self.result.configure(text="Paste an email first", fg="gray")
            return "break"

        probability = spam_probability(self.model, raw_email)
        result = f"{probability:.0%} likely spam"
        if probability >= UNSURE_HIGH:
            color = SPAM_COLOR
        elif probability >= UNSURE_LOW:
            result += " (unsure)"
            color = UNSURE_COLOR
        else:
            color = HAM_COLOR
        self.result.configure(text=result, fg=color)

        return "break"

    def clear(self):
        self.text.delete("1.0", "end")
        self.result.configure(text="")
        self.text.focus_set()


def main():
    # Keeps the window from looking blurry on high DPI Windows displays
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

    root = tk.Tk()

    if not MODEL_PATH.exists():
        root.withdraw()
        messagebox.showerror(
            "Spam Filter", "No saved model found.\nRun: python -m spam_filter.train"
        )
        root.destroy()
        return

    SpamFilterApp(root, load_model())
    root.mainloop()


if __name__ == "__main__":
    main()
