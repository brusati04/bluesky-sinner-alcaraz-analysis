import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ast
import json
import re
import os
import requests
import time
import html



# --- Download NLTK Resources Defensively ---
for _resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{_resource}' if _resource == 'stopwords' else f'tokenizers/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)

# Set styling for high-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Load the spaCy English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("spaCy model 'en_core_web_sm' not found, downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Shared NLP objects
tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({'tennis', 'match', 'play', 'player', 'set'})
analyzer = SentimentIntensityAnalyzer()

# US Open 2025 Key Match Dates
US_OPEN_EVENTS = [
    ("2025-08-24", "R1 begins\n(Sinner/Alcaraz)", "blue"),
    ("2025-08-30", "R3 matches", "blue"),                 
    ("2025-09-01", "R4 matches", "blue"),                 
    ("2025-09-03", "Quarterfinals", "orange"),            
    ("2025-09-05", "Semifinals", "red"),                
    ("2025-09-07", "Final\n(Alcaraz wins)", "darkred"),  
]


def parse_list_col(val):
    """Safely parse columns stored as string-formatted lists."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []


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
    
    # Use TweetTokenizer to preserve hashtags and emojis
    tokens = tokenizer.tokenize(cleaned.lower())
    
    # Create pre-tokenized Doc in spaCy
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    
    # Run active tagging and lemmatization components
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


def get_vader_sentiment(text):
    if pd.isna(text) or text == "":
        return "neutral", 0.0
    scores = analyzer.polarity_scores(text)
    comp = scores['compound']
    if comp >= 0.05:
        cat = "positive"
    elif comp <= -0.05:
        cat = "negative"
    else:
        cat = "neutral"
    return cat, comp


def extract_ner(text):
    if pd.isna(text) or text == "":
        return []
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            text_clean = ent.text.strip().replace("\n", " ")
            ents.append((text_clean, ent.label_))
    return list(set(ents))


ENTITY_MAP = {
    "http://dbpedia.org/resource/Jannik_Sinner": {
        "surface_form": "Sinner",
        "keys": ["sinner", "jannik", "de sinner"]
    },
    "http://dbpedia.org/resource/Carlos_Alcaraz": {
        "surface_form": "Alcaraz",
        "keys": ["alcaraz", "carlos", "carlitos"]
    },
    "http://dbpedia.org/resource/US_Open_(tennis)": {
        "surface_form": "US Open",
        "keys": ["us open", "usopen"]
    },
    "http://dbpedia.org/resource/Novak_Djokovic": {
        "surface_form": "Djokovic",
        "keys": ["djokovic"]
    }
}

def build_local_links(text_lower):
    links = []
    for uri, data in ENTITY_MAP.items():
        if any(key in text_lower for key in data["keys"]):
            links.append({
                "surface_form": data["surface_form"],
                "uri": uri,
                "similarity_score": 1.0
            })
    return links

def link_entities_dbpedia(row, dbpedia_cache, state, confidence=0.55):
    """
    Named Entity Linking via DBpedia Spotlight with circuit breaker and cooldown.
    """
    text = row['cleaned_text']
    spacy_ents = row['entities']

    if not text or pd.isna(text) or text.strip() == "":
        return []

    # Skip if spaCy NER found no relevant entities — avoids unnecessary API calls
    if not spacy_ents:
        return []

    # Cache check to avoid duplicate API calls
    if text in dbpedia_cache:
        return dbpedia_cache[text]

    text_lower = text.lower()
    local_links = build_local_links(text_lower)

    # Circuit breaker with cooldown
    if state.get("circuit_broken", False):
        broken_at = state.get("broken_at", 0.0)
        cooldown = state.get("cooldown_seconds", 60)
        if time.time() - broken_at > cooldown:
            print("[NEL] Cooldown expired. Resetting DBpedia Spotlight circuit breaker...")
            state["circuit_broken"] = False
        else:
            dbpedia_cache[text] = local_links
            return local_links

    url = "https://api.dbpedia-spotlight.org/en/annotate"
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
            params={"text": text, "confidence": confidence},
            timeout=1.5
        )
        if response.status_code == 200:
            api_linked = [
                {
                    "surface_form": r.get("@surfaceForm"),
                    "uri": r.get("@URI"),
                    "similarity_score": float(r.get("@similarityScore", 0))
                }
                for r in response.json().get("Resources", [])
            ]
            api_uris = {e["uri"] for e in api_linked}
            merged = api_linked + [e for e in local_links if e["uri"] not in api_uris]
            dbpedia_cache[text] = merged
            return merged

        elif response.status_code == 429 or response.status_code >= 500:
            print(f"[NEL] DBpedia Spotlight returned status {response.status_code}. Activating Circuit Breaker...")
            state.update({"circuit_broken": True, "broken_at": time.time()})

    except Exception as e:
        print(f"[NEL] DBpedia Spotlight connection timed out/failed ({e}). Activating Circuit Breaker...")
        state.update({"circuit_broken": True, "broken_at": time.time()})

    dbpedia_cache[text] = local_links
    return local_links


def build_did_to_handle(df):
    """Build a DID -> handle mapping from the posts DataFrame."""
    did_to_handle = {}
    for _, row in df.iterrows():
        did = row['author_did']
        handle = row['author_handle']
        if pd.notna(did) and pd.notna(handle):
            did_to_handle[did] = handle
    return did_to_handle


def save_plot_copies(filename):
    """Save the current figure to both plots/ and report/ directories."""
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.savefig(os.path.join("report", filename), dpi=300)


def load_and_preprocess_data(filepath="data/sinner_alcaraz_posts.csv"):
    print("\n=== Consolidated SNA & NLP Pipeline: Loading Data ===")
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found! Run the crawler first.")
        return None, None

    print(f"Successfully loaded {len(df)} posts.")

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    df = df.dropna(subset=['created_at'])

    df['hashtags'] = df['hashtags'].apply(parse_list_col)
    df['mentions'] = df['mentions'].apply(parse_list_col)
    df['links'] = df['links'].apply(parse_list_col)

    did_to_handle = build_did_to_handle(df)
    print(f"Mapped {len(did_to_handle)} unique author DIDs to Handles.")
    return df, did_to_handle

