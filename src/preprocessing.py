import os
import re
import sys
import html
import subprocess
from typing import Optional

import pandas as pd
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from utils import parse_list_col, build_did_to_handle

# Ensure the NLTK resources required for tokenisation/stopwords are available.
for _resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{_resource}' if _resource == 'stopwords' else f'tokenizers/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)

# Load the spaCy English model, downloading it on first run if needed.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found, downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

tokenizer = TweetTokenizer()
custom_stopwords = set(stopwords.words('english')).union({'tennis', 'match', 'play', 'player', 'set'})


def clean_text(text: Optional[str]) -> str:
    """Normalise raw post text: unescape HTML entities, strip URLs and @handles, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess(text: Optional[str]) -> str:
    """Clean, tokenise and lemmatise post text into a space-joined string of meaningful tokens.

    Stopwords and single-character ASCII tokens are dropped, while hashtags and
    non-ASCII tokens (e.g. emojis) are preserved.
    """
    cleaned = clean_text(text)
    tokens = tokenizer.tokenize(cleaned.lower())

    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    for name in ["tagger", "attribute_ruler", "lemmatizer"]:
        if name in nlp.pipe_names:
            doc = nlp.get_pipe(name)(doc)

    filtered_tokens = []
    for token in doc:
        t = token.text
        lemma = token.lemma_.lower()
        if lemma in custom_stopwords or t in custom_stopwords:
            continue
        if t.isalnum() or t.startswith('#') or not t.isascii():
            # Keep emojis/non-ASCII even when a single character; drop lone ASCII letters.
            if len(lemma) > 1 or not t.isascii():
                filtered_tokens.append(lemma)

    return " ".join(filtered_tokens)


def load_and_preprocess_data(filepath: str = "data/sinner_alcaraz_posts.csv") -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load raw crawled posts, parse timestamps and list columns, and build the DID->handle map.

    Returns (None, None) if the raw file is missing.
    """
    for directory in ("data", "plots", "report"):
        os.makedirs(directory, exist_ok=True)

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

    return df, build_did_to_handle(df)


def load_processed_data(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load an already NLP-enriched dataset from CSV, restoring list columns and the DID->handle map."""
    df = pd.read_csv(filepath, parse_dates=['created_at'])
    list_columns = ['mentions', 'hashtags', 'links', 'linked_entities']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_col)
    return df, build_did_to_handle(df)


def prepare_dataset(raw_filepath: str, processed_filepath: str) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Return the NLP-enriched dataset and DID->handle map for the analysis pipeline.

    Loads a cached processed file when available; otherwise runs the full NLP
    enrichment over the raw crawled posts. Returns (None, None) if raw input is missing.
    """
    if os.path.exists(processed_filepath):
        print(f"[INFO] Processed dataset found at {processed_filepath}. Loading it directly...")
        return load_processed_data(processed_filepath)

    df, did_to_handle = load_and_preprocess_data(raw_filepath)
    if df is None:
        return None, None

    from social_sentiment_analysis import run_nlp_enrichment
    df_processed = run_nlp_enrichment(df, output_filepath=processed_filepath)["df"]
    return df_processed, did_to_handle
