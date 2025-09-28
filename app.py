from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re
from urllib.parse import urlparse
from typing import List

# Load model yang sudah ditraining dengan dataset Kaggle
try:
    model_data = joblib.load("./kaggle_phishing_model.joblib")
    print("Kaggle phishing model loaded successfully")
    FEATURE_COLUMNS = model_data['feature_columns']
    MODEL_PIPELINE = model_data['pipeline']
    MODEL_CLASSES = model_data['classes']
except:
    model_data = None
    print("Warning: Kaggle model not found. Please train the model first.")
    FEATURE_COLUMNS = []
    MODEL_PIPELINE = None
    MODEL_CLASSES = []

app = FastAPI(title="Kaggle Phishing Detection API")

class URLRequest(BaseModel):
    url: str

class BatchURLRequest(BaseModel):
    urls: List[str]

def extract_features(url: str):
    """
    Extract features sesuai dengan dataset Kaggle phishing
    Function ini akan disesuaikan dengan kolom yang ada di dataset
    """
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Dictionary untuk semua fitur yang mungkin ada di dataset Kaggle
        all_features = {
            # Common features dari berbagai dataset Kaggle phishing
            'having_ip_address': 1 if re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url) else -1,
            'url_length': 1 if len(url) < 54 else (-1 if len(url) >= 75 else 0),
            'shortining_service': 1 if any(short in url for short in ['bit.ly', 'tinyurl', 't.co', 'goo.gl']) else -1,
            'having_at_symbol': 1 if '@' in url else -1,
            'double_slash_redirecting': 1 if '//' in url[7:] else -1,
            'prefix_suffix': 1 if '-' in domain else -1,
            'having_sub_domain': 1 if len(domain.split('.')) > 3 else (-1 if len(domain.split('.')) < 3 else 0),
            'ssl_final_state': 1 if url.startswith('https') else (-1 if 'https' in domain else 0),
            'domain_registeration_length': 0,  # Simplified
            'favicon': 0,  # Simplified
            'port': 1 if ':' in parsed.netloc and not any(x in parsed.netloc for x in [':80', ':443']) else -1,
            'https_token': 1 if 'https' in domain else -1,
            
            # Request URL related (simplified)
            'request_url': 0,
            'url_of_anchor': 0,
            'links_in_tags': 0,
            'sfh': 0,
            'submitting_to_email': 1 if 'mailto:' in url else -1,
            'abnormal_url': 0,
            
            # Additional features
            'redirect': 0,
            'on_mouseover': 0,
            'right_click': 0,
            'popup_window': 0,
            'iframe': 0,
            'age_of_domain': 0,
            'dns_record': 0,
            'web_traffic': 0,
            'page_rank': 0,
            'google_index': 0,
            'links_pointing_to_page': 0,
            'statistical_report': 0,
            
            # Numeric features
            'length_url': len(url),
            'length_hostname': len(domain),
            'ip': 1 if re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', domain) else 0,
            'nb_dots': url.count('.'),
            'nb_hyphens': url.count('-'),
            'nb_at': url.count('@'),
            'nb_qm': url.count('?'),
            'nb_and': url.count('&'),
            'nb_or': url.count('|'),
            'nb_eq': url.count('='),
            'nb_underscore': url.count('_'),
            'nb_tilde': url.count('~'),
            'nb_percent': url.count('%'),
            'nb_slash': url.count('/'),
            'nb_star': url.count('*'),
            'nb_colon': url.count(':'),
            'nb_comma': url.count(','),
            'nb_semicolon': url.count(';'),
            'nb_dollar': url.count('$'),
            'nb_space': url.count(' '),
            'nb_www': url.count('www'),
            'nb_com': url.count('com'),
            'nb_dslash': url.count('//'),
            'http_in_path': 1 if 'http' in path else 0,
            'https_token_in_path': 1 if 'https' in path else 0,
            'ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if url else 0,
            'ratio_digits_host': sum(c.isdigit() for c in domain) / len(domain) if domain else 0,
            'punycode': 1 if 'xn--' in url else 0,
            'port_in_url': 1 if ':' in parsed.netloc else 0,
            'tld_in_path': 1 if any(tld in path for tld in ['.com', '.org', '.net']) else 0,
            'tld_in_subdomain': 1 if len([x for x in domain.split('.')[:-2] if x in ['com', 'org', 'net']]) > 0 else 0,
            'abnormal_subdomain_number': max(0, len(domain.split('.')) - 3),
            'nb_subdomains': max(0, len(domain.split('.')) - 2),
            'prefix_suffix_sep_ratio': domain.count('-') / len(domain) if domain else 0,
            'path_extension': 1 if '.' in path.split('/')[-1] else 0,
            'query_length': len(query),
            'suspicious_tld': 1 if any(tld in url for tld in ['.tk', '.ml', '.ga', '.cf']) else 0,
            
            # Original features dari kode sebelumnya
            'num_dots': url.count('.'),
            'url_length_category': 1 if len(url) < 30 else (2 if len(url) < 75 else 3),
            'at_symbol': 1 if '@' in url else 0,
            'num_dash': url.count('-'),
            'num_percent': url.count('%'),
            'num_query_components': url.count('?'),
            'ip_address': 1 if re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', domain) else 0,
            'https_in_hostname': 1 if 'https' in domain else 0,
            'path_level': len([x for x in path.split('/') if x]),
            'path_length': len(path),
            'num_numeric_chars': sum(c.isdigit() for c in url)
        }
        
        # Return features sesuai dengan yang digunakan model
        if FEATURE_COLUMNS:
            result = []
            for feature_name in FEATURE_COLUMNS:
                if feature_name in all_features:
                    result.append(all_features[feature_name])
                else:
                    result.append(0)  # Default value
            return result
        else:
            # Fallback jika model belum loaded
            return [
                all_features['num_dots'],
                all_features['url_length_category'], 
                all_features['at_symbol'],
                all_features['num_dash'],
                all_features['num_percent'],
                all_features['num_query_components'],
                all_features['ip_address'],
                all_features['https_in_hostname'],
                all_features['path_level'],
                all_features['path_length'],
                all_features['num_numeric_chars']
            ]
            
    except Exception as e:
        # Return zeros if extraction fails
        feature_count = len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 11
        return [0] * feature_count

def get_prediction_label(prediction_result):
    """Convert model prediction to readable label"""
    if isinstance(prediction_result, (int, float)):
        # Binary model (0/1 or -1/1)
        if prediction_result == 1:
            return "phishing"
        else:
            return "aman"
    else:
        # String prediction
        return str(prediction_result).lower()

def get_confidence_score(probabilities):
    """Calculate confidence score from probabilities"""
    if len(probabilities) > 0:
        return float(max(probabilities)) * 100
    return 50.0

@app.post("/predict")
def predict(data: URLRequest):
    """Predict single URL using trained Kaggle model"""
    
    if not MODEL_PIPELINE:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first using the dataset.")
    
    url = data.url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Extract features
        features = extract_features(url)
        X = np.array([features])
        
        # Predict
        prediction = MODEL_PIPELINE.predict(X)[0]
        probabilities = MODEL_PIPELINE.predict_proba(X)[0]
        
        # Convert prediction to readable format
        prediction_label = get_prediction_label(prediction)
        
        # Calculate confidence
        confidence_score = get_confidence_score(probabilities)
        
        # Determine confidence level
        if confidence_score >= 80:
            confidence_level = "tinggi"
        elif confidence_score >= 60:
            confidence_level = "sedang"
        else:
            confidence_level = "rendah"
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_label in enumerate(MODEL_CLASSES):
            readable_label = get_prediction_label(class_label)
            prob_dict[readable_label] = round(probabilities[i] * 100, 2)
        
        return {
            "url": data.url,
            "prediction": prediction_label,
            "confidence_score": round(confidence_score, 2),
            "confidence_level": confidence_level,
            "probabilities": prob_dict,
            "model_features_used": len(features),
            "recommendation": get_recommendation(prediction_label, confidence_score)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def get_recommendation(prediction: str, confidence_score: float) -> str:
    """Get recommendation based on prediction"""
    
    if prediction == "aman":
        if confidence_score >= 80:
            return "URL terlihat sangat aman untuk diakses."
        elif confidence_score >= 60:
            return "URL kemungkinan aman, tapi tetap gunakan kehati-hatian umum."
        else:
            return "URL mungkin aman tapi model kurang yakin. Periksa dengan teliti."
    else:  # phishing
        if confidence_score >= 80:
            return "BAHAYA! URL ini sangat mungkin phishing. Jangan akses dan hindari memasukkan data pribadi!"
        elif confidence_score >= 60:
            return "URL berpotensi berbahaya. Sangat disarankan untuk tidak mengakses."
        else:
            return "URL mungkin berbahaya tapi model kurang yakin. Gunakan kehati-hatian ekstra."

@app.post("/batch_predict")
def batch_predict(data: BatchURLRequest):
    """Predict multiple URLs"""
    if not MODEL_PIPELINE:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    results = []
    for url in data.urls:
        try:
            result = predict(URLRequest(url=url))
            results.append(result)
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e),
                "prediction": "error"
            })
    
    # Summary statistics
    summary = {
        "aman": len([r for r in results if r.get('prediction') == 'aman']),
        "phishing": len([r for r in results if r.get('prediction') == 'phishing']),
        "error": len([r for r in results if r.get('prediction') == 'error'])
    }
    
    return {
        "results": results,
        "total": len(data.urls),
        "summary": summary
    }

@app.get("/")
def root():
    return {
        "message": "Kaggle Phishing Detection API",
        "status": "running",
        "model_type": "Trained on Kaggle Phishing Dataset",
        "features_count": len(FEATURE_COLUMNS),
        "classes": MODEL_CLASSES.tolist() if hasattr(MODEL_CLASSES, 'tolist') else list(MODEL_CLASSES),
        "accuracy": model_data.get('accuracy', 0) if model_data else 0
    }

@app.get("/model_info")
def model_info():
    """Get detailed information about the loaded model"""
    if not model_data:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": str(type(MODEL_PIPELINE.named_steps['classifier'])),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS[:20],  # First 20 features
        "classes": MODEL_CLASSES.tolist() if hasattr(MODEL_CLASSES, 'tolist') else list(MODEL_CLASSES),
        "accuracy": model_data.get('accuracy', 0),
        "cv_scores_mean": float(np.mean(model_data.get('cv_scores', [0]))),
        "pipeline_steps": list(MODEL_PIPELINE.named_steps.keys())
    }

@app.get("/test_features/{url:path}")
def test_features(url: str):
    """Test feature extraction for debugging"""
    features = extract_features(url)
    
    if FEATURE_COLUMNS:
        feature_dict = dict(zip(FEATURE_COLUMNS, features))
        return {
            "url": url,
            "features": feature_dict,
            "feature_count": len(features),
            "non_zero_features": {k: v for k, v in feature_dict.items() if v != 0}
        }
    else:
        return {
            "url": url,
            "features": features,
            "feature_count": len(features),
            "note": "Model not loaded, showing basic features"
        }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PIPELINE is not None,
        "features_available": len(FEATURE_COLUMNS) > 0
    }