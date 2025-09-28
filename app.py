from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np
from urllib.parse import urlparse

# Load pipeline model
model = joblib.load("./phishing_detector_model.joblib")

app = FastAPI(title="Phishing Detection API")

# Skema input
class URLRequest(BaseModel):
    url: str

# Whitelist domain terpercaya
TRUSTED_DOMAINS = {
    # Educational institutions Indonesia
    'pens.ac.id', 'its.ac.id', 'ui.ac.id', 'ugm.ac.id', 'unair.ac.id',
    'itb.ac.id', 'unpad.ac.id', 'undip.ac.id', 'unhas.ac.id', 'uny.ac.id',
    'ub.ac.id', 'um.ac.id', 'unesa.ac.id', 'upi.edu', 'unsri.ac.id',
    
    # Government Indonesia
    'go.id', 'kemendikbud.go.id', 'kemenkeu.go.id', 'polri.go.id',
    'bps.go.id', 'pajak.go.id', 'bpkp.go.id', 'bkpm.go.id',
    
    # Major international companies
    'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com',
    'youtube.com', 'gmail.com', 'yahoo.com', 'outlook.com',
    
    # Indonesian companies
    'tokopedia.com', 'bukalapak.com', 'shopee.co.id', 'blibli.com',
    'bca.co.id', 'mandiri.co.id', 'bni.co.id', 'bri.co.id',
    'detik.com', 'kompas.com', 'tribunnews.com', 'liputan6.com'
}

def is_trusted_domain(url: str) -> bool:
    """Check if URL belongs to trusted domain"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Check exact match
        if domain in TRUSTED_DOMAINS:
            return True
            
        # Check if it's a subdomain of trusted domain
        for trusted in TRUSTED_DOMAINS:
            if domain.endswith('.' + trusted):
                return True
                
        return False
    except:
        return False

# Fungsi ekstraksi fitur yang diperbaiki
def extract_features(url: str):
    """
    Fixed feature extraction function with better logic
    """
    try:
        # Parse URL properly
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query
        
        # Remove www prefix for analysis
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return [
            url.count("."),                                     # num_dots
            len(url),                                          # url_length
            1 if "@" in url else 0,                           # at_symbol
            url.count("-"),                                    # num_dash
            url.count("%"),                                    # num_percent
            url.count("?"),                                    # num_query_components
            1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", domain) else 0,  # ip_address (check domain only)
            1 if parsed.scheme == 'https' else 0,             # https_protocol (fixed logic)
            max(0, url.count("/") - 2),                       # path_level (subtract protocol //)
            len(path) if path else 0,                         # path_length (actual path length)
            sum(c.isdigit() for c in url)                     # num_numeric_chars
        ]
    except Exception as e:
        # Fallback to safer method if URL parsing fails
        return [
            url.count("."),                             
            len(url),                                   
            1 if "@" in url else 0,                     
            url.count("-"),                             
            url.count("%"),                             
            url.count("?"),                             
            1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0,  
            1 if url.lower().startswith('https') else 0,  # simplified but safer check
            max(0, url.count("/") - 2),                             
            len(url.split("/")[-1]) if "/" in url else 0,                    
            sum(c.isdigit() for c in url)              
        ]

@app.post("/predict")
def predict(data: URLRequest):
    url = data.url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Check whitelist first
    if is_trusted_domain(url):
        return {
            "url": data.url,
            "phishing_score": 5.0,
            "prediction": "aman",
            "reason": "Domain terpercaya dalam whitelist",
            "confidence": "tinggi"
        }
    
    # Extract features
    features = extract_features(url)
    X = np.array([features])

    # Prediksi probabilitas phishing
    try:
        phishing_prob = model.predict_proba(X)[0][1]
        phishing_score = phishing_prob * 100
    except Exception as e:
        return {
            "url": data.url,
            "error": f"Model prediction failed: {str(e)}",
            "prediction": "error"
        }

    # Tentukan kategori dengan threshold yang lebih realistis
    if phishing_score <= 20:
        label = "aman"
        confidence = "tinggi" if phishing_score <= 10 else "sedang"
    elif phishing_score <= 75:
        label = "mencurigakan"
        confidence = "sedang"
    else:
        label = "phishing"
        confidence = "tinggi" if phishing_score >= 90 else "sedang"

    return {
        "url": data.url,
        "phishing_score": round(phishing_score, 2),
        "prediction": label,
        "confidence": confidence,
        "recommendation": get_recommendation(label, phishing_score)
    }

def get_recommendation(label: str, score: float) -> str:
    """Get recommendation based on prediction"""
    if label == "aman":
        return "URL terlihat aman untuk diakses"
    elif label == "mencurigakan":
        if score < 50:
            return "URL mungkin aman, tapi tetap waspada. Periksa domain dan konten dengan teliti"
        else:
            return "URL mencurigakan. Hindari memasukkan data pribadi. Verifikasi keaslian website"
    else:
        return "URL sangat berbahaya! Jangan akses atau masukkan data apapun. Laporkan sebagai phishing"

@app.get("/")
def root():
    return {
        "message": "Phishing Detection API", 
        "status": "running",
        "version": "1.1.0",
        "trusted_domains_count": len(TRUSTED_DOMAINS)
    }

@app.post("/batch_predict")
def batch_predict(urls: list[str]):
    """Predict multiple URLs at once"""
    results = []
    for url in urls:
        try:
            result = predict(URLRequest(url=url))
            results.append(result)
        except Exception as e:
            results.append({
                "url": url, 
                "error": str(e),
                "prediction": "error"
            })
    return {"results": results, "total": len(urls)}

@app.get("/trusted_domains")
def get_trusted_domains():
    """Get list of trusted domains"""
    return {
        "trusted_domains": sorted(list(TRUSTED_DOMAINS)),
        "count": len(TRUSTED_DOMAINS)
    }

@app.post("/add_trusted_domain")
def add_trusted_domain(domain: str):
    """Add domain to trusted list (for admin use)"""
    domain = domain.lower().strip()
    if domain and domain not in TRUSTED_DOMAINS:
        TRUSTED_DOMAINS.add(domain)
        return {"message": f"Domain {domain} added to trusted list"}
    return {"message": f"Domain {domain} already in trusted list or invalid"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}