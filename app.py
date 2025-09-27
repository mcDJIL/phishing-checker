from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re

# Load model & scaler
model = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Phishing URL Detection API")

# ================== Feature Extraction ==================
def extract_features(url: str):
    return {
        "num_dots": url.count("."),
        "url_length": len(url),
        "at_symbol": 1 if "@" in url else 0,
        "num_dash": url.count("-"),
        "num_percent": url.count("%"),
        "num_query_components": url.count("?"),
        "ip_address": 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0,
        "https_in_hostname": 1 if len(url.split("/")) > 2 and "https" in url.split("/")[2] else 0,
        "path_level": url.count("/"),
        "path_length": len(url.split("/")[-1]),
        "num_numeric_chars": sum(c.isdigit() for c in url),
    }

# ================== Request Body ==================
class URLInput(BaseModel):
    url: str

# ================== API Endpoint ==================
@app.post("/predict")
def predict(input: URLInput):
    features = extract_features(input.url)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)

    prob = model.predict_proba(X_new_scaled)[0][1]
    score = int(prob * 100)

    status = "Phishing"

    if score >= 50:
        status = "Phishing"
    elif score >= 30:
        status = "Mencurigakan"
    else:
        status = "Aman"

    return {
        "url": input.url,
        "phishing_score": score,
        "prediction": status
    }
