"""Train and evaluate the spam email classifier."""

from pathlib import Path

from src.model import train_and_evaluate


ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    metrics = train_and_evaluate(
        data_path=ROOT / "data" / "emails.csv",
        model_path=ROOT / "models" / "spam_classifier.joblib",
        report_path=ROOT / "reports" / "metrics.json",
    )
    print("Training complete.")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1_score']:.2%}")
    print("Model saved to models/spam_classifier.joblib")
