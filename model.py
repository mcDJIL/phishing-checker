import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_dataset(file_path):
    """
    Load dataset dan analisis struktur data
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Analisis target column
    possible_target_cols = ['Result', 'class', 'label', 'target', 'phishing', 'Phishing', 'Phising']
    target_col = None
    
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        print(f"\nTarget column found: '{target_col}'")
        print(f"Target distribution:")
        print(df[target_col].value_counts())
        
        # Jika binary (0,1 atau -1,1)
        unique_values = df[target_col].unique()
        print(f"Unique values in target: {unique_values}")
        
    return df, target_col

def preprocess_kaggle_data(df, target_col):
    """
    Preprocess dataset berdasarkan format yang terdeteksi
    """
    # Copy dataframe
    processed_df = df.copy()
    
    # Handle target column
    if target_col:
        # Convert binary labels to readable format
        unique_values = processed_df[target_col].unique()
        
        if len(unique_values) == 2:
            # Binary classification
            if set(unique_values) == {0, 1}:
                processed_df['label'] = processed_df[target_col].map({0: 'aman', 1: 'phishing'})
            elif set(unique_values) == {-1, 1}:
                processed_df['label'] = processed_df[target_col].map({-1: 'aman', 1: 'phishing'})
            else:
                # Keep original if already string
                processed_df['label'] = processed_df[target_col]
                
        else:
            # Multi-class - keep as is
            processed_df['label'] = processed_df[target_col]
    
    # Remove original target column if different name
    if target_col and target_col != 'label':
        processed_df = processed_df.drop(columns=[target_col])
    
    # Remove non-feature columns
    non_feature_cols = ['id', 'ID', 'url', 'URL', 'Unnamed: 0', 'index']
    for col in non_feature_cols:
        if col in processed_df.columns:
            processed_df = processed_df.drop(columns=[col])
    
    # Handle missing values
    processed_df = processed_df.fillna(0)
    
    return processed_df

def train_kaggle_model(file_path, model_type='rf'):
    """
    Train model menggunakan dataset Kaggle
    """
    # Load and analyze
    df, target_col = load_and_analyze_dataset(file_path)
    
    # Preprocess
    processed_df = preprocess_kaggle_data(df, target_col)
    
    # Separate features and target
    if 'label' not in processed_df.columns:
        raise ValueError("Target column tidak ditemukan. Pastikan dataset memiliki kolom target.")
    
    X = processed_df.drop(columns=['label'])
    y = processed_df['label']
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Choose model
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        model_name = "Random Forest"
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model_name = "Gradient Boosting"
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train model
    print(f"\nTraining {model_name} model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Cross validation
    print(f"\nCross-validation scores:")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (if Random Forest)
    if model_type == 'rf':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_columns': X.columns.tolist(),
        'classes': pipeline.classes_,
        'accuracy': accuracy,
        'cv_scores': cv_scores
    }
    
    joblib.dump(model_data, 'kaggle_phishing_model.joblib')
    print(f"\nModel saved as 'kaggle_phishing_model.joblib'")
    
    return model_data

def create_extract_features_function(feature_columns):
    """
    Create extract_features function yang sesuai dengan kolom dataset
    """
    
    function_template = f"""
def extract_features(url: str):
    \"\"\"
    Extract features sesuai dengan dataset Kaggle yang digunakan
    \"\"\"
    import re
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Dictionary untuk menyimpan semua fitur yang mungkin
        all_features = {{
            # URL structure features
            'having_ip_address': 1 if re.search(r'(\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}})', url) else -1,
            'url_length': 1 if len(url) < 54 else (-1 if len(url) >= 75 else 0),
            'shortining_service': 1 if any(short in url for short in ['bit.ly', 'tinyurl', 't.co', 'goo.gl']) else -1,
            'having_at_symbol': 1 if '@' in url else -1,
            'double_slash_redirecting': 1 if '//' in url[7:] else -1,
            'prefix_suffix': 1 if '-' in domain else -1,
            'having_sub_domain': len(domain.split('.')) - 2 if domain else 0,
            'ssl_final_state': 1 if url.startswith('https') else -1,
            'domain_registeration_length': 0,  # Would need whois lookup
            'favicon': 0,  # Would need to check favicon
            'port': 1 if ':' in parsed.netloc and not parsed.netloc.endswith(':80') and not parsed.netloc.endswith(':443') else -1,
            'https_token': 1 if 'https' in domain else -1,
            
            # Request URL features
            'request_url': 0,  # Would need to analyze page content
            'url_of_anchor': 0,  # Would need to analyze page content
            'links_in_tags': 0,  # Would need to analyze page content
            'sfh': 0,  # Would need to analyze page content
            'submitting_to_email': 0,  # Would need to analyze page content
            'abnormal_url': 0,  # Would need whois lookup
            
            # HTML and JavaScript features
            'redirect': 0,  # Would need to check redirects
            'on_mouseover': 0,  # Would need to analyze page content
            'right_click': 0,  # Would need to analyze page content
            'popup_window': 0,  # Would need to analyze page content
            'iframe': 0,  # Would need to analyze page content
            'age_of_domain': 0,  # Would need whois lookup
            'dns_record': 0,  # Would need DNS lookup
            'web_traffic': 0,  # Would need traffic data
            'page_rank': 0,  # Would need PageRank data
            'google_index': 0,  # Would need to check Google index
            'links_pointing_to_page': 0,  # Would need backlink data
            'statistical_report': 0,  # Would need to check blacklists
            
            # Additional common features
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_percent': url.count('%'),
            'num_query_components': url.count('?'),
            'num_ampersand': url.count('&'),
            'num_hash': url.count('#'),
            'num_numeric_chars': sum(c.isdigit() for c in url),
            'no_of_dir': len([x for x in path.split('/') if x]),
            'no_of_embed_domain': 0,  # Would need content analysis
            'shortening_service': 1 if any(short in url for short in ['bit.ly', 'tinyurl', 't.co']) else 0,
            'abnormal_subdomain': 1 if len(domain.split('.')) > 3 else 0,
            'length_url': len(url),
            'length_hostname': len(domain),
            'ip': 1 if re.search(r'(\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}})', domain) else 0,
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
            'random_string_ratio': 0,  # Complex calculation
            'path_extension': 1 if '.' in path.split('/')[-1] else 0,
            'query_length': len(query),
            'domain_in_brand': 0,  # Would need brand database
            'brand_in_subdomain': 0,  # Would need brand database
            'brand_in_path': 0,  # Would need brand database
            'suspicious_tld': 1 if any(tld in url for tld in ['.tk', '.ml', '.ga', '.cf']) else 0,
            'statistical_report_based': 0  # Would need statistical analysis
        }}
        
        # Return only features that exist in the model
        feature_names = {feature_columns}
        result = []
        
        for feature_name in feature_names:
            if feature_name in all_features:
                result.append(all_features[feature_name])
            else:
                result.append(0)  # Default value for unknown features
                
        return result
        
    except Exception as e:
        # Return zeros if extraction fails
        return [0] * len({feature_columns})
"""
    
    return function_template

# Example usage
if __name__ == "__main__":
    print("Kaggle Phishing Dataset Trainer")
    print("================================")
    
    # You would replace this with your actual dataset path
    dataset_path = "./Phising_dataset_predict.csv"  # Change this to your downloaded dataset path
    
    try:
        # Train model
        model_data = train_kaggle_model(dataset_path, model_type='rf')
        
        # Generate extract_features function
        function_code = create_extract_features_function(model_data['feature_columns'])
        
        # Save function to file
        with open('extract_features_generated.py', 'w') as f:
            f.write(function_code)
        
        print("\\nGenerated extract_features function saved to 'extract_features_generated.py'")
        print("Copy this function to your app.py file")
        
    except FileNotFoundError:
        print(f"Dataset file '{dataset_path}' not found.")
        print("Please download the Kaggle dataset and update the dataset_path variable.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your dataset format and try again.")