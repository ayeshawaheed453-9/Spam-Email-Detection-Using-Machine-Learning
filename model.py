"""Core machine-learning utilities for spam email detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


REQUIRED_COLUMNS = {"label", "text"}


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV dataset containing label and text columns."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame[["label", "text"]].dropna().copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["label"].isin(["ham", "spam"]) & frame["text"].ne("")]
    if frame["label"].nunique() < 2:
        raise ValueError("Dataset must contain both ham and spam examples.")
    return frame.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """Create the TF-IDF plus logistic-regression classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def train_and_evaluate(
    data_path: str | Path,
    model_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train the model, evaluate it on a holdout set, and save artifacts."""
    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics: dict[str, Any] = {
        "dataset_size": int(len(data)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0),
    }
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: str | Path) -> Pipeline:
    """Load a trained pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return joblib.load(path)


def predict_email(model_path: str | Path, text: str) -> dict[str, Any]:
    """Predict whether an email is spam and return a confidence score."""
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("Email text cannot be empty.")
    model = load_model(model_path)
    label = str(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(label)])
    return {"label": label, "confidence": round(confidence, 4), "text": cleaned}
