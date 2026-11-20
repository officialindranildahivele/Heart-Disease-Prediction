# Heart Disease Prediction

A Flask web application that uses a TensorFlow/Keras artificial neural network to estimate heart-disease risk from 13 clinical inputs. It includes a dashboard, prediction history, PDF-assisted form filling, and downloadable PDF reports.

> **Medical disclaimer:** This project is for educational purposes only. Its output is not a diagnosis and must not be used to make healthcare decisions. Consult a qualified healthcare professional for medical advice.

## Features

- Register and sign in to a local account.
- Enter clinical values and receive a `Risk` or `Safe` model result.
- View a dashboard and a history of saved predictions.
- Upload a PDF to extract values for the prediction form.
- Download a PDF report for a saved prediction.
- Run inference using the included trained model and feature scaler.

## Tech stack

- Python and Flask
- TensorFlow/Keras
- NumPy, pandas, scikit-learn
- SQLite
- pdfplumber and ReportLab

## Getting started

### Requirements

- Python installed with a version supported by the TensorFlow release available for your operating system.
- pip

### Install

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/officialindranildahivele/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start the app from the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The app creates `patients.db` and its tables on startup. The trained model and scaler are stored in `models/`.

## Model and input data

The app expects 13 values, in this order:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

Use the form’s labels and choices when entering encoded categorical values. The model applies the included `StandardScaler` before prediction. The result threshold is defined in `app.py`; the displayed confidence is a model score, not a clinically validated probability.

The training script reads:

```text
dataset/Dataset--Heart-Disease-Prediction-using-ANN (1).csv
```

To retrain the model from the repository root:

```bash
python train_model.py
```

This writes updated model artifacts under `models/`. Retraining is optional; the committed model and scaler are used for normal app startup.

## Project structure

```text
.
├── app.py              # Flask routes and prediction/report logic
├── database.py         # SQLite initialization and schema updates
├── dataset/            # Training data
├── models/             # Trained Keras model and fitted scaler
├── templates/          # Flask HTML templates
├── train_model.py      # Model training script
└── requirements.txt    # Python dependencies
```

## Security and deployment

This repository is an educational prototype, not a production-ready medical service. Before deploying or handling real or sensitive data:

- Replace the hard-coded Flask session secret in `app.py` with a strong secret loaded from a secure environment variable.
- Add proper password hashing, HTTPS, secure cookie settings, input validation, and production server configuration.
- Review consent, privacy, retention, and access controls for uploaded PDFs and stored health information.
- Do not run Flask's development server with debug mode enabled in production.

## License

No license is currently specified. Unless a license is added, standard copyright restrictions apply.
