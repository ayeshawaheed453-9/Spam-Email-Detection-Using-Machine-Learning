from pathlib import Path

import pytest

from src.model import load_dataset, predict_email, train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "emails.csv"


def test_dataset_has_both_classes():
    frame = load_dataset(DATA_PATH)
    assert set(frame["label"]) == {"ham", "spam"}
    assert len(frame) >= 20


def test_train_and_predict(tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "metrics.json"
    metrics = train_and_evaluate(DATA_PATH, model_path, report_path)
    assert model_path.exists()
    assert report_path.exists()
    assert 0 <= metrics["accuracy"] <= 1
    result = predict_email(model_path, "Congratulations, claim your free cash prize now!")
    assert result["label"] in {"ham", "spam"}
    assert 0 <= result["confidence"] <= 1


def test_empty_text_is_rejected(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_and_evaluate(DATA_PATH, model_path)
    with pytest.raises(ValueError):
        predict_email(model_path, "   ")
