from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "spam_classifier.pkl"
