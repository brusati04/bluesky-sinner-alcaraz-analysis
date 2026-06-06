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
import matplotlib.pyplot as plt

from utils import parse_list_col, build_did_to_handle, save_plot_copies

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


def clean_text_bert(text: Optional[str]) -> str:
    """Light normalisation for BERT/RoBERTa: unescape HTML, replace handles with @user, replace URLs with http, preserve punctuation and casing."""
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    # Replace links with 'http' (standard for CardiffNLP twitter-roberta-base-sentiment)
    text = re.sub(r'http\S+|www\.\S+', 'http', text)
    # Replace handles with '@user' (standard for CardiffNLP twitter-roberta-base-sentiment)
    text = re.sub(r'@\w+', '@user', text)
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


def load_data(filepath: str, parse_linked_entities: bool = False) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load, parse dates/lists, and build DID->handle map for raw or processed datasets."""
    for directory in ("data", "plots", "report"):
        os.makedirs(directory, exist_ok=True)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found!")
        return None, None

    # Parse timestamps
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    df = df.dropna(subset=['created_at'])

    # Parse list columns
    list_columns = ['mentions', 'hashtags', 'links']
    if parse_linked_entities:
        list_columns.append('linked_entities')

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_col)

    return df, build_did_to_handle(df)


def prepare_dataset(raw_filepath: str, processed_filepath: str) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Return the NLP-enriched dataset and DID->handle map for the analysis pipeline.

    Loads a cached processed file when available; otherwise runs the full NLP
    enrichment over the raw crawled posts. Returns (None, None) if raw input is missing.
    """
    df = None
    did_to_handle = None
    if os.path.exists(processed_filepath):
        print(f"[INFO] Processed dataset found at {processed_filepath}. Loading it directly...")
        df, did_to_handle = load_data(processed_filepath, parse_linked_entities=True)
    else:
        raw_df, did_to_handle = load_data(raw_filepath, parse_linked_entities=False)
        if raw_df is None:
            return None, None

        from social_sentiment_analysis import run_nlp_enrichment
        df = run_nlp_enrichment(raw_df, output_filepath=processed_filepath)["df"]

    if df is not None:
        # Enforce all Phase 1 stance & dominant emotion computations are present
        cols_to_check = ['author_stance_score', 'author_stance_leaning', 'author_dominant_emotion_nrc', 'author_dominant_emotion_bert']
        if not all(col in df.columns for col in cols_to_check):
            print("[INFO] Adding stance and dominant emotion columns to processed dataset...")
            from social_stance_analysis import compute_post_stances, compute_user_stances, classify_stances
            
            # Post stance
            df = compute_post_stances(df)
            # User stance
            user_stances = compute_user_stances(df, min_posts=1)
            user_stances = classify_stances(user_stances, threshold=0.05)
            
            # Map back to posts DataFrame
            df['author_stance_score'] = df['author_handle'].map(user_stances['net_stance']).fillna(0.0)
            df['author_stance_leaning'] = df['author_handle'].map(user_stances['stance_leaning']).fillna('neutral')
            
            # Map user dominant emotion back to df
            for backend in ("nrc", "bert"):
                col = "nrc_dominant_emotion" if backend == "nrc" else "bert_dominant_emotion"
                user_emotions = df.groupby("author_handle")[col].agg(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else "neutral"
                ).to_dict()
                df[f'author_dominant_emotion_{backend}'] = df['author_handle'].map(user_emotions).fillna("neutral")
                
            # Overwrite the cache file with completed dataset
            df.to_csv(processed_filepath, index=False)
            print(f"[INFO] Processed dataset cache updated at {processed_filepath}")
            
    return df, did_to_handle


def plot_community_wordcloud(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Generate and save a single word cloud for the entire community using preprocessed text."""
    
    if 'preprocessed_text' not in df.columns:
        print("Warning: 'preprocessed_text' column not found. Skipping word cloud.")
        return

    from wordcloud import WordCloud

    # Filter out empty preprocessed texts and combine
    valid_texts = df['preprocessed_text'].dropna().astype(str).tolist()
    valid_texts = [t for t in valid_texts if t.strip() != ""]

    if not valid_texts:
        print("Warning: No words found for community word cloud. Skipping.")
        return

    community_text = " ".join(valid_texts)
    
    # Generate single community word cloud
    wordcloud_community = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        random_state=42,
        max_words=300
    ).generate(community_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_community, interpolation='bilinear')
    plt.axis('off')
    plt.title("Bluesky Community - Overall Word Cloud\n(US Open 2025)", fontsize=16, pad=15, weight='bold')
    plt.tight_layout()
    save_plot_copies("community_wordcloud.png", output_dir=output_dir)
    plt.close()
