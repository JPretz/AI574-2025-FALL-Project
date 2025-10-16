import re
import pandas as pd

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\$\.\,\-\_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_sentiment(label):
    """Map raw sentiment labels to numerical values."""
    label = str(label).lower()
    if label in ['negative', 'neg', '0']:
        return 0
    elif label in ['neutral', 'neu', '1']:
        return 1
    elif label in ['positive', 'pos', '2']:
        return 2
    else:
        return 1
