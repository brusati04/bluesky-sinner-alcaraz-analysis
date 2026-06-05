import os
import re
import ast
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import save_plot_copies, US_OPEN_EVENTS, US_OPEN_ROUNDS
from preprocessing import clean_text, preprocess

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# The 8 primary NRC emotion categories
NRC_EMOTIONS = [
    'fear', 'anger', 'anticipation', 'trust',
    'surprise', 'sadness', 'disgust', 'joy'
]

# NRC emotion columns expected in df
POSITIVE_EMOTIONS = ["emotion_joy", "emotion_trust", "emotion_anticipation"]
NEGATIVE_EMOTIONS = ["emotion_anger", "emotion_disgust", "emotion_sadness", "emotion_fear"]

# Weights for multi-signal stance score
W_SENTIMENT = 0.50
W_EMOTION   = 0.35
W_FREQUENCY = 0.15

SINNER_URI  = "http://dbpedia.org/resource/Jannik_Sinner"
ALCARAZ_URI = "http://dbpedia.org/resource/Carlos_Alcaraz"

SINNER_KEYWORDS  = {"sinner", "jannik"}
ALCARAZ_KEYWORDS = {"alcaraz", "carlos", "carlitos"}

# ─────────────────────────────────────────────────────────────────────────────
# NLP, ENTITY MATCHING & EMOTION SCORING
# ─────────────────────────────────────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()
_dbpedia_cache = {}

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
    from preprocessing import nlp
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            text_clean = ent.text.strip().replace("\n", " ")
            ents.append((text_clean, ent.label_))
    return list(set(ents))


def link_entities_dbpedia(text, confidence=0.5):
    """
    Local Named Entity Linking (replacing slow DBpedia Spotlight API).
    Identifies core tennis rivalry entities locally using keyword matching
    to avoid heavy network request delays while retaining exact schemas.
    """
    if not text or pd.isna(text) or text.strip() == "":
        return []

    text_lower = text.lower()
    entities = []
    
    # Fast local keyword matches mapping to DBpedia URIs
    if "sinner" in text_lower or "jannik" in text_lower:
        entities.append({
            "surface_form": "Jannik Sinner",
            "uri": "http://dbpedia.org/resource/Jannik_Sinner"
        })
    if "alcaraz" in text_lower or "carlos" in text_lower or "carlitos" in text_lower:
        entities.append({
            "surface_form": "Carlos Alcaraz",
            "uri": "http://dbpedia.org/resource/Carlos_Alcaraz"
        })
        
    return entities


def score_emotions(text: str) -> dict:
    """
    Score a single text string across 8 NRC emotion categories.
    """
    try:
        if not text or pd.isna(text) or str(text).strip() == "":
            return {e: 0.0 for e in NRC_EMOTIONS}
    except (TypeError, ValueError):
        return {e: 0.0 for e in NRC_EMOTIONS}

    try:
        emotion_obj = NRCLex()
        emotion_obj.load_raw_text(str(text))
        freqs = emotion_obj.affect_frequencies
        return {e: freqs.get(e, 0.0) for e in NRC_EMOTIONS}
    except Exception:
        return {e: 0.0 for e in NRC_EMOTIONS}


def add_emotion_columns(df: pd.DataFrame, text_col: str = 'cleaned_text') -> pd.DataFrame:
    """
    Apply NRC emotion scoring to a DataFrame and return it with
    8 new columns and a 'dominant_emotion' column.
    """
    emotion_scores = df[text_col].apply(score_emotions)
    emotion_df = pd.DataFrame(emotion_scores.tolist(), index=df.index)
    emotion_df.columns = [f'emotion_{e}' for e in NRC_EMOTIONS]

    df = pd.concat([df, emotion_df], axis=1)

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')

    # Mark posts where all emotion scores are 0 as 'neutral'
    all_zero_mask = df[emotion_cols].sum(axis=1) == 0
    df.loc[all_zero_mask, 'dominant_emotion'] = 'neutral'

    return df


def run_nlp_enrichment(df, output_filepath="data/sinner_alcaraz_processed.csv"):
    df = df.copy()
    total_posts = len(df)
    print(f"[NLP] Starting NLP enrichment for {total_posts} posts...")

    print("[NLP] Step 1/5: Cleaning and preprocessing post text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['preprocessed_text'] = df['text'].apply(preprocess)
    print("[NLP] Step 1/5 completed.")

    print("[NLP] Step 2/5: Scoring VADER sentiments...")
    sents = []
    for idx, t in enumerate(df['cleaned_text']):
        sents.append(get_vader_sentiment(t))
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_posts} VADER compound scores...")
    df['sentiment_category'] = [s[0] for s in sents]
    df['sentiment_compound'] = [s[1] for s in sents]
    print("[NLP] Step 2/5 completed.")

    print("[NLP] Step 3/5: Running NRC Emotion Lexicon analysis...")
    # Track progress inside add_emotion_columns or here
    emotion_scores = []
    for idx, t in enumerate(df['cleaned_text']):
        emotion_scores.append(score_emotions(t))
        if (idx + 1) % 500 == 0:
            print(f"  Scored emotions for {idx + 1}/{total_posts} posts...")
    emotion_df = pd.DataFrame(emotion_scores, index=df.index)
    emotion_df.columns = [f'emotion_{e}' for e in NRC_EMOTIONS]
    df = pd.concat([df, emotion_df], axis=1)
    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
    all_zero_mask = df[emotion_cols].sum(axis=1) == 0
    df.loc[all_zero_mask, 'dominant_emotion'] = 'neutral'
    print("[NLP] Step 3/5 completed.")

    print("[NLP] Step 4/5: Extracting Named Entities (spaCy NER)...")
    entities = []
    for idx, t in enumerate(df['cleaned_text']):
        entities.append(extract_ner(t))
        if (idx + 1) % 500 == 0:
            print(f"  Extracted entities for {idx + 1}/{total_posts} posts...")
    df['entities'] = entities
    print("[NLP] Step 4/5 completed.")

    print("[NLP] Step 5/5: Querying DBpedia Spotlight for Entity Linking...")
    linked_entities = []
    for idx, t in enumerate(df['cleaned_text']):
        linked_entities.append(link_entities_dbpedia(t, confidence=0.5))
        if (idx + 1) % 100 == 0:
            print(f"  Linked entities for {idx + 1}/{total_posts} posts...")
    df['linked_entities'] = linked_entities
    print("[NLP] Step 5/5 completed.")

    # Correlate Sentiment via Named Entity Linking (NEL)
    sinner_scores = []
    alcaraz_scores = []
    for _, row in df.iterrows():
        linked_ents = row['linked_entities']
        comp = row['sentiment_compound']

        uris = [ent['uri'] for ent in linked_ents if isinstance(ent, dict)]
        is_sinner = "http://dbpedia.org/resource/Jannik_Sinner" in uris
        is_alcaraz = "http://dbpedia.org/resource/Carlos_Alcaraz" in uris

        t = str(row['text']).lower()
        if not is_sinner and ("sinner" in t or "jannik" in t):
            is_sinner = True
        if not is_alcaraz and ("alcaraz" in t or "carlos" in t):
            is_alcaraz = True

        if is_sinner:
            sinner_scores.append(comp)
        if is_alcaraz:
            alcaraz_scores.append(comp)

    df.to_csv(output_filepath, index=False)
    print(f"[NLP] Saved processed dataset to {output_filepath}")
    return {
        "df": df,
        "sinner_scores": sinner_scores,
        "alcaraz_scores": alcaraz_scores
    }


def plot_community_emotion_profiles(df, Gu, node_to_community, communities, output_dir="plots", title_suffix="", top_k=3, sort_by="node_count"):
    """
    Study 2: Emotional Profiling of Communities.
    Compare average scores of 8 NRC emotions across the top k filtered communities.
    sort_by: "node_count" or "post_volume" to select/sort top communities.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    # strictly target handles present in the filtered graph Gu
    author_community = {}
    for _, row in df.iterrows():
        handle = row.get('author_handle')
        if handle and handle in node_to_community and handle in Gu.nodes():
            author_community[handle] = node_to_community[handle]

    df = df.copy()
    df['community_id'] = df['author_handle'].map(author_community)
    df_with_comm = df.dropna(subset=['community_id']).copy()

    # Determine top communities based on sorting criteria
    if sort_by == "post_volume":
        comm_post_counts = df_with_comm['community_id'].value_counts()
        top_comms = comm_post_counts.head(top_k).index.tolist()
    else: # default to "node_count"
        from collections import Counter
        filtered_node_to_comm = {node: comm for node, comm in node_to_community.items() if node in Gu.nodes()}
        counts = Counter(filtered_node_to_comm.values())
        top_comms = [cid for cid, count in counts.most_common(top_k)]

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]

    records = []
    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
        if len(comm_posts) == 0:
            continue
        avg_scores = comm_posts[emotion_cols].mean()
        for col in emotion_cols:
            emotion = col.replace('emotion_', '')
            records.append({
                "Community": f"Comm {int(cid)} (N={len(comm_posts)})",
                "Emotion": emotion,
                "Score": avg_scores[col]
            })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        return

    from social_network_analysis import get_community_color_map
    color_map = get_community_color_map(node_to_community)
    custom_palette = {}
    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
        if len(comm_posts) == 0:
            continue
        label = f"Comm {int(cid)} (N={len(comm_posts)})"
        custom_palette[label] = color_map.get(cid)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Score", hue="Community", palette=custom_palette)
    plt.title(f"Average Emotion Profiles per Community{title_suffix}\n(NRC Emotion Lexicon)", fontsize=14, pad=15)
    plt.xlabel("Emotion Category")
    plt.ylabel("Mean Normalized Score")
    plt.legend(title="Community")
    plt.tight_layout()

    suffix_filename = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
    fname = f"community_emotion_profiles{suffix_filename}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()
