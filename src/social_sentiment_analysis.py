import os
from typing import Optional

# Set custom Hugging Face cache directory to avoid permissions/lock issues in user home directory
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(workspace_path, ".huggingface_cache"))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from nrclex import NRCLex
from transformers import pipeline

from utils import save_plot_copies
from preprocessing import clean_text, clean_text_bert, preprocess
from social_network_analysis import get_community_color_map

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# The 8 primary NRC emotion categories.
NRC_EMOTIONS = [
    'fear', 'anger', 'anticipation', 'trust',
    'surprise', 'sadness', 'disgust', 'joy'
]

# NOTE: the stance-weight and polarity groupings below are scaffolding for a
# multi-signal stance score that the current pipeline does not yet compute.
POSITIVE_EMOTIONS = ["emotion_joy", "emotion_trust", "emotion_anticipation"]
NEGATIVE_EMOTIONS = ["emotion_anger", "emotion_disgust", "emotion_sadness", "emotion_fear"]
W_SENTIMENT = 0.50
W_EMOTION = 0.35
W_FREQUENCY = 0.15

SINNER_URI = "http://dbpedia.org/resource/Jannik_Sinner"
ALCARAZ_URI = "http://dbpedia.org/resource/Carlos_Alcaraz"

SINNER_KEYWORDS = {"sinner", "jannik"}
ALCARAZ_KEYWORDS = {"alcaraz", "carlos", "carlitos"}

# ─────────────────────────────────────────────────────────────────────────────
# NLP, ENTITY MATCHING & EMOTION SCORING
# ─────────────────────────────────────────────────────────────────────────────


def extract_ner(text: Optional[str]) -> list[tuple[str, str]]:
    """Extract unique (entity_text, label) pairs for PERSON/ORG/GPE/LOC entities via spaCy."""
    if pd.isna(text) or text == "":
        return []
    from preprocessing import nlp
    ents = []
    for ent in nlp(text).ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            ents.append((ent.text.strip().replace("\n", " "), ent.label_))
    return list(set(ents))


def link_entities_dbpedia(text: Optional[str], confidence: float = 0.5) -> list[dict]:
    """Link the two rival players to their DBpedia URIs using local keyword matching.

    This replaces the slow DBpedia Spotlight API while preserving the same output schema.
    """
    if not text or pd.isna(text) or str(text).strip() == "":
        return []

    text_lower = text.lower()
    entities = []
    if any(k in text_lower for k in SINNER_KEYWORDS):
        entities.append({"surface_form": "Jannik Sinner", "uri": SINNER_URI})
    if any(k in text_lower for k in ALCARAZ_KEYWORDS):
        entities.append({"surface_form": "Carlos Alcaraz", "uri": ALCARAZ_URI})
    return entities


def score_emotions(text: str) -> dict[str, float]:
    """Score a single text across the 8 NRC emotion categories, returning zeros on empty/invalid input."""
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
    """Add the 8 NRC emotion columns plus a 'dominant_emotion' column to a DataFrame.

    Posts with no detected emotion are labelled 'neutral'.
    """
    emotion_df = pd.DataFrame(df[text_col].apply(score_emotions).tolist(), index=df.index)
    emotion_df.columns = [f'emotion_{e}' for e in NRC_EMOTIONS]
    df = pd.concat([df, emotion_df], axis=1)

    emotion_cols = list(emotion_df.columns)
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
    df.loc[df[emotion_cols].sum(axis=1) == 0, 'dominant_emotion'] = 'neutral'
    return df


def derive_player_sentiment_scores(df: pd.DataFrame) -> dict[str, list[float]]:
    """Bucket compound scores per player (Sinner / Alcaraz).

    A post counts toward a player when that player's DBpedia entity was linked
    or their name keywords appear in the raw post text.
    """
    sinner_scores: list[float] = []
    alcaraz_scores: list[float] = []
    for linked_ents, comp, text in zip(df['linked_entities'], df['sentiment_compound'], df['text']):
        uris = {ent['uri'] for ent in linked_ents if isinstance(ent, dict)}
        text_lower = str(text).lower()
        if SINNER_URI in uris or any(k in text_lower for k in SINNER_KEYWORDS):
            sinner_scores.append(comp)
        if ALCARAZ_URI in uris or any(k in text_lower for k in ALCARAZ_KEYWORDS):
            alcaraz_scores.append(comp)
    return {"sinner_scores": sinner_scores, "alcaraz_scores": alcaraz_scores}


def run_nlp_enrichment(df: pd.DataFrame, output_filepath: str = "data/sinner_alcaraz_processed.csv") -> dict:
    """Enrich crawled posts with cleaned text, RoBERTa sentiment, NRC emotions, NER and entity links.

    Writes the enriched DataFrame to output_filepath and returns it together with
    per-player sentiment score buckets.
    """
    df = df.copy()
    total_posts = len(df)
    print(f"[NLP] Starting BERT/RoBERTa NLP enrichment for {total_posts} posts...")

    print("[NLP] Step 1/5: Cleaning and preprocessing post text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['preprocessed_text'] = df['text'].apply(preprocess)
    print("[NLP] Step 1/5 completed.")

    print("[NLP] Step 2/5: Scoring RoBERTa sentiments on GPU...")
    # Initialize pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"[NLP] Initializing CardiffNLP RoBERTa model on device={device}...")
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        top_k=None
    )

    # Clean the text with clean_text_bert ONLY for BERT/RoBERTa input
    bert_inputs = df['text'].apply(clean_text_bert).tolist()
    batch_size = 128
    sentiment_categories = []
    sentiment_compounds = []

    print(f"[NLP] Running GPU parallelized inference with batch_size={batch_size}...")
    
    # Generator to pass to pipeline for parallelized dataloading and batching on GPU
    def input_generator():
        for t in bert_inputs:
            yield t if (isinstance(t, str) and t.strip() != "") else " "

    # Running classifier with generator yields results dynamically, which is parallelized under the hood
    results = classifier(input_generator(), batch_size=batch_size, truncation=True, max_length=512)

    for res in tqdm(results, total=len(bert_inputs), desc="RoBERTa Sentiment"):
        scores_dict = {d['label']: d['score'] for d in res}
        
        # Label mappings (handles standard string labels or LABEL_X identifiers)
        if 'LABEL_0' in scores_dict:
            p_neg = scores_dict.get('LABEL_0', 0.0)
            p_neu = scores_dict.get('LABEL_1', 0.0)
            p_pos = scores_dict.get('LABEL_2', 0.0)
            
            max_label = max(scores_dict, key=scores_dict.get)
            if max_label == 'LABEL_0':
                cat = 'negative'
            elif max_label == 'LABEL_2':
                cat = 'positive'
            else:
                cat = 'neutral'
        else:
            p_neg = scores_dict.get('negative', 0.0)
            p_neu = scores_dict.get('neutral', 0.0)
            p_pos = scores_dict.get('positive', 0.0)
            
            cat = max(scores_dict, key=scores_dict.get)
            
        # Compound score = P(positive) - P(negative)
        comp = p_pos - p_neg
        
        sentiment_categories.append(cat)
        sentiment_compounds.append(comp)

    df['sentiment_category'] = sentiment_categories
    df['sentiment_compound'] = sentiment_compounds
    print("[NLP] Step 2/5 completed.")

    print("[NLP] Step 3/5: Running NRC Emotion Lexicon analysis...")
    df = add_emotion_columns(df, text_col='cleaned_text')
    print("[NLP] Step 3/5 completed.")

    print("[NLP] Step 4/5: Extracting Named Entities (spaCy NER)...")
    df['entities'] = df['cleaned_text'].apply(extract_ner)
    print("[NLP] Step 4/5 completed.")

    print("[NLP] Step 5/5: Linking entities to DBpedia resources...")
    df['linked_entities'] = df['cleaned_text'].apply(link_entities_dbpedia)
    print("[NLP] Step 5/5 completed.")

    scores = derive_player_sentiment_scores(df)

    df.to_csv(output_filepath, index=False)
    print(f"[NLP] Saved processed dataset to {output_filepath}")
    return {"df": df, **scores}


def plot_community_emotion_profiles(
    df: pd.DataFrame,
    Gu,
    node_to_community: dict,
    output_dir: str = "plots",
    title_suffix: str = "",
    top_k: int = 5,
    sort_by: str = "post_volume",
    cmap_name: str = "tab20",
) -> None:
    """Plot average NRC emotion profiles for the top-k communities of the filtered graph Gu.

    Communities are selected either by post volume or by node count (sort_by).
    """
    author_community = {}
    for handle in df['author_handle']:
        if handle in node_to_community and handle in Gu.nodes():
            author_community[handle] = node_to_community[handle]

    df = df.copy()
    df['community_id'] = df['author_handle'].map(author_community)
    df_with_comm = df.dropna(subset=['community_id']).copy()

    if sort_by == "post_volume":
        top_comms = df_with_comm['community_id'].value_counts().head(top_k).index.tolist()
    else:
        from collections import Counter
        filtered = {n: c for n, c in node_to_community.items() if n in Gu.nodes()}
        top_comms = [cid for cid, _ in Counter(filtered.values()).most_common(top_k)]

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]
    records = []
    custom_palette = {}
    color_map = get_community_color_map(node_to_community, cmap_name=cmap_name)

    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
        if len(comm_posts) == 0:
            continue
        label = f"Comm {int(cid)} (N={len(comm_posts)})"
        custom_palette[label] = color_map.get(cid)
        avg_scores = comm_posts[emotion_cols].mean()
        for col in emotion_cols:
            records.append({"Community": label, "Emotion": col.replace('emotion_', ''), "Score": avg_scores[col]})

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Score", hue="Community", palette=custom_palette)
    plt.title(f"Average Emotion Profiles per Community{title_suffix}\n(NRC Emotion Lexicon)", fontsize=14, pad=15)
    plt.xlabel("Emotion Category")
    plt.ylabel("Mean Normalized Score")
    plt.legend(title="Community")
    plt.tight_layout()

    suffix_filename = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
    save_plot_copies(f"community_emotion_profiles{suffix_filename}.png", output_dir=output_dir)
    plt.close()
