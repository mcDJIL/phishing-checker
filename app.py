from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np

# Load model (pastikan ini RandomForestClassifier trained dengan 11 fitur)
model = joblib.load("./phishing_detector_model.joblib")

# Inisialisasi FastAPI
app = FastAPI(title="Phishing Detection API")

# Skema input user
class URLRequest(BaseModel):
    url: str

# Fungsi extract features sesuai urutan kolom model
def extract_features(url: str):
    return [
        url.count("."),                                  # num_dots
        len(url),                                       # url_length
        1 if "@" in url else 0,                         # at_symbol
        url.count("-"),                                 # num_dash
        url.count("%"),                                 # num_percent
        url.count("?"),                                 # num_query_components
        1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0,  # ip_address
        1 if len(url.split("/")) > 2 and "https" in url.split("/")[2] else 0,  # https_in_hostname
        url.count("/"),                                 # path_level
        len(url.split("/")[-1]),                        # path_length
        sum(c.isdigit() for c in url)                  # num_numeric_chars
    ]

@app.post("/predict")
def predict(data: URLRequest):
    # Ekstraksi fitur
    X = np.array([extract_features(data.url)])

    # Probabilitas phishing (kelas 1)
    phishing_score = model.predict_proba(X)[0][1] * 100

    # Tentukan kategori berdasarkan score
    if phishing_score <= 30:
        label = "aman"
    elif phishing_score <= 70:
        label = "mencurigakan"
    else:
        label = "phishing"

    return {
        "url": data.url,
        "phishing_score": round(phishing_score, 2),
        "prediction": label,
    }
