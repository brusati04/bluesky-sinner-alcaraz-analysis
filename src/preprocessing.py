import sys
import os
import re
import html
import subprocess
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from utils import parse_list_col, build_did_to_handle

# Ensure NLTK resources are downloaded
for _resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{_resource}' if _resource == 'stopwords' else f'tokenizers/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)

# Load the spaCy English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("spaCy model 'en_core_web_sm' not found, downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({'tennis', 'match', 'play', 'player', 'set'})


def clean_text(text):
    if pd.isna(text):
        return ""
    # 1. Decode HTML entities
    text = html.unescape(text)
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 3. Remove handles (e.g. @carlosalcaraz)
    text = re.sub(r'@\w+', '', text)
    # 4. Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    cleaned = clean_text(text)
    tokens = tokenizer.tokenize(cleaned.lower())
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    # Run lemmatization components
    for name in ["tagger", "attribute_ruler", "lemmatizer"]:
        if name in nlp.pipe_names:
            doc = nlp.get_pipe(name)(doc)
            
    filtered_tokens = []
    for token in doc:
        t = token.text
        lemma = token.lemma_.lower()
        
        # Filter stopwords
        if lemma not in custom_stopwords and t not in custom_stopwords:
            # Keep alphanumeric, hashtags, and characters/emojis
            if t.isalnum() or t.startswith('#') or not t.isascii():
                # Avoid single characters unless they are emojis
                if len(lemma) > 1 or not t.isascii():
                    filtered_tokens.append(lemma)
                    
    return " ".join(filtered_tokens)

def load_and_preprocess_data(filepath="data/sinner_alcaraz_posts.csv"):
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found! Run the crawler first.")
        return None, None

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    df = df.dropna(subset=['created_at'])

    for col in ['hashtags', 'mentions', 'links']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_col)

    did_to_handle = build_did_to_handle(df)
    return df, did_to_handle
