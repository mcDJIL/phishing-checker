
def extract_features(url: str):
    """
    Extract features sesuai dengan dataset Kaggle yang digunakan
    """
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
        all_features = {
            # URL structure features
            'having_ip_address': 1 if re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url) else -1,
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
            'random_string_ratio': 0,  # Complex calculation
            'path_extension': 1 if '.' in path.split('/')[-1] else 0,
            'query_length': len(query),
            'domain_in_brand': 0,  # Would need brand database
            'brand_in_subdomain': 0,  # Would need brand database
            'brand_in_path': 0,  # Would need brand database
            'suspicious_tld': 1 if any(tld in url for tld in ['.tk', '.ml', '.ga', '.cf']) else 0,
            'statistical_report_based': 0  # Would need statistical analysis
        }
        
        # Return only features that exist in the model
        feature_names = ['NumDots', 'UrlLength', 'AtSymbol', 'NumDash', 'NumPercent', 'NumQueryComponents', 'IpAddress', 'HttpsInHostname', 'PathLevel', 'PathLength', 'NumNumericChars']
        result = []
        
        for feature_name in feature_names:
            if feature_name in all_features:
                result.append(all_features[feature_name])
            else:
                result.append(0)  # Default value for unknown features
                
        return result
        
    except Exception as e:
        # Return zeros if extraction fails
        return [0] * len(['NumDots', 'UrlLength', 'AtSymbol', 'NumDash', 'NumPercent', 'NumQueryComponents', 'IpAddress', 'HttpsInHostname', 'PathLevel', 'PathLength', 'NumNumericChars'])
