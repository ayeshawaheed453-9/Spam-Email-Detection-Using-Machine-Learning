# Spam Email Detection

This project is a complete Python machine-learning application that classifies email text as **spam** or **ham** (not spam). It uses a TF-IDF text representation and a logistic-regression classifier. The included dataset is small and intended for demonstration and testing; for production use, replace it with a larger, representative, privacy-safe dataset.

## Features

The project includes a validated CSV dataset, model training, holdout evaluation, command-line prediction, a Flask web interface, saved model and metric artifacts, and automated tests. The application does not require an external API or internet connection after dependencies are installed.

## Project structure

```text
spam-email-detection/
├── app.py                         # Flask web app
├── predict.py                     # Command-line prediction
├── train.py                       # Train and evaluate the model
├── requirements.txt               # Python dependencies
├── data/emails.csv                # Labeled sample dataset
├── models/                        # Saved model is created here
├── reports/                       # Evaluation metrics are created here
├── src/model.py                   # Reusable ML functions
├── templates/index.html            # Web page
└── tests/test_model.py             # Automated tests
```

## Installation

Use Python 3.10 or newer. From the project directory, create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train.py
```

This creates `models/spam_classifier.joblib` and `reports/metrics.json`. The training script uses a stratified 75/25 train/test split and reports accuracy, precision, recall, F1 score, and a confusion matrix.

## Predict from the command line

Train the model first, then run:

```bash
python predict.py "Congratulations! Claim your free prize now."
```

You can also run `python predict.py` and paste the email when prompted.

## Run the web app

```bash
python train.py
python app.py
```

Open <http://127.0.0.1:5000> in a browser and paste an email message into the form.

## Run tests

```bash
python -m pytest -q
```

## Dataset format

To use your own data, replace `data/emails.csv` with a CSV containing exactly these required columns:

```csv
label,text
ham,"Your appointment is confirmed for Friday."
spam,"You won a cash prize. Click now!"
```

Labels must be `ham` or `spam`. Do not train on sensitive personal email content without appropriate consent, privacy controls, and data governance.

## Limitations

The included dataset is intentionally small, so its evaluation score is not a reliable estimate of real-world performance. A production classifier should use a larger dataset, a separate validation set, threshold tuning, monitoring for concept drift, and a review path for uncertain messages. Never rely on this demo alone for high-impact decisions.
