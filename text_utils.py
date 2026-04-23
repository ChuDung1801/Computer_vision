import re

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_volume(text):
    if not text:
        return None
    pattern = r'(\d+(?:\.\d+)?)\s?(ml|l)'
    match = re.search(pattern, text.lower())
    
    if match:
        return match.group()
    return None