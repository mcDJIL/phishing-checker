from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np

# Load model
model = joblib.load("./phishing_detector_model.joblib")

# Inisialisasi FastAPI
app = FastAPI(title="Phishing Detection API")

# Skema input user
class URLRequest(BaseModel):
    url: str

# Fungsi extract features
def extract_features(url: str):
    return {
        "unnamed:_0": 0,  # Placeholder, tidak digunakan dalam prediksi
        "num_dots": url.count("."),
        "url_length": len(url),
        "at_symbol": 1 if "@" in url else 0,
        "num_dash": url.count("-"),
        "num_percent": url.count("%"),
        "num_query_components": url.count("?"),
        "ip_address": 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0,
        "https_in_hostname": (
            1 if len(url.split("/")) > 2 and "https" in url.split("/")[2] else 0
        ),
        "path_level": url.count("/"),
        "path_length": len(url.split("/")[-1]),
        "num_numeric_chars": sum(c.isdigit() for c in url),
    }

@app.post("/predict")
def predict(data: URLRequest):
    # Ekstraksi fitur
    features = extract_features(data.url)

    # Pastikan urutan fitur sesuai dengan training (exclude 'unnamed:_0' dan 'phising')
    feature_order = [
        "unnamed:_0",
        "num_dots",
        "url_length",
        "at_symbol",
        "num_dash",
        "num_percent",
        "num_query_components",
        "ip_address",
        "https_in_hostname",
        "path_level",
        "path_length",
        "num_numeric_chars",
    ]
    X = np.array([[features[f] for f in feature_order]])

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
