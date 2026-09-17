"""Classify one email from the command line."""

import argparse
from pathlib import Path

from src.model import predict_email


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect whether an email is spam.")
    parser.add_argument("text", nargs="?", help="Email text to classify.")
    args = parser.parse_args()
    text = args.text or input("Paste email text: ")
    result = predict_email(ROOT / "models" / "spam_classifier.joblib", text)
    print(f"Prediction: {result['label'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
