# 📧 Spam Email Detection Using Machine Learning

## 📌 Project Description

Spam Email Detection is a machine learning classification project that identifies whether an email is **Spam** or **Ham (Not Spam)**.

The project uses Natural Language Processing (NLP), TF-IDF Vectorization, and Logistic Regression to analyze email text and predict its category.

## 🎯 Objective

To build a machine learning model that automatically detects unwanted emails and classifies them as Spam or Ham.

## 🧠 Algorithm and Model

* **Algorithm:** Logistic Regression
* **Text Processing:** TF-IDF Vectorization
* **Type:** Supervised Machine Learning – Classification

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP
* Flask
* Matplotlib

## 📝 Main Tasks

1. Load and understand the email dataset.
2. Clean and preprocess email text.
3. Convert text into numerical features using TF-IDF.
4. Train the Logistic Regression model.
5. Evaluate model performance.
6. Predict Spam or Ham emails.
7. Run the model using a Flask web application.

## 📂 Project Structure

```text
spam-email-detection/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── data/
│   └── emails.csv
├── models/
├── reports/
├── src/
│   └── model.py
├── templates/
│   └── index.html
└── tests/
    └── test_model.py
```

## ⚙️ Installation

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### 1. Train the Model

```bash
python train.py
```

### 2. Predict an Email

```bash
python predict.py "Congratulations! You won a free prize."
```

Example output:

```text
Prediction: spam
```

### 3. Run the Web App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 📊 Results

The model generates the following evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

**Actual results are generated after running `python train.py`.**

## 📧 Example Prediction

| Email                                       | Prediction |
| ------------------------------------------- | ---------- |
| Congratulations! Claim your free prize now! | Spam       |
| Your appointment is confirmed for Friday.   | Ham        |

## ⚠️ Limitations

The included dataset is small and intended for demonstration and testing. The model may not perform reliably on all real-world emails.

## 🔮 Future Improvements

* Use a larger dataset.
* Compare different machine learning algorithms.
* Improve text preprocessing.
* Enhance the Flask web interface.

## 👩‍💻 Author

**Ayesha**

Machine Learning | Data Analytics | Python

