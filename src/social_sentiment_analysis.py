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

# Mapping from GoEmotions (28 fine-grained labels) to the 8 NRC/Plutchik
# categories, so the BERT backend produces scores comparable to NRCLex.
GOEMOTIONS_TO_NRC = {
    "joy":          ["joy", "amusement", "excitement", "love", "pride", "relief"],
    "trust":        ["admiration", "approval", "gratitude", "caring"],
    "anticipation": ["curiosity", "desire", "optimism"],
    "surprise":     ["surprise", "realization", "confusion"],
    "fear":         ["fear", "nervousness"],
    "anger":        ["anger", "annoyance", "disapproval"],
    "disgust":      ["disgust"],
    "sadness":      ["sadness", "disappointment", "grief", "remorse", "embarrassment"],
}

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

# Lazily-initialised GoEmotions (RoBERTa) classifier for the BERT emotion backend.
_emotion_clf_bert = None


def _get_emotion_clf_bert():
    global _emotion_clf_bert
    if _emotion_clf_bert is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[NLP] Initializing GoEmotions BERT model on device={device}...")
        _emotion_clf_bert = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
            max_length=128,
            device=device,
        )
    return _emotion_clf_bert


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


def add_emotion_columns(df: pd.DataFrame, text_col: str = 'cleaned_text', backend: str = "nrc") -> pd.DataFrame:
    """Add the 8 emotion columns plus a 'dominant_emotion' column to a DataFrame.

    The ``backend`` selects the scoring engine:
      - "nrc"  : per-post NRCLex affect frequencies (the original behaviour).
      - "bert" : batched GoEmotions (RoBERTa) inference, with the 28 fine-grained
                 labels collapsed onto the 8 NRC categories via GOEMOTIONS_TO_NRC.

    Either way the output columns (`emotion_*`, `dominant_emotion`) and the neutral
    fallback (sum of scores == 0 → 'neutral') are identical. Posts with no detected
    emotion are labelled 'neutral'.
    """
    if backend == "bert":
        clf = _get_emotion_clf_bert()
        batch_size = 128
        emotion_inputs = df[text_col].tolist()

        # Generator to pass to pipeline for parallelized dataloading and batching on GPU
        def emotion_generator():
            for t in emotion_inputs:
                yield t if (isinstance(t, str) and t.strip() != "") else " "

        print(f"[NLP] Running GPU parallelized inference with batch_size={batch_size}...")
        raw_results = clf(
            emotion_generator(),
            batch_size=batch_size,
            truncation=True,
            max_length=128,
        )

        emotion_rows = []
        dominant_emotions = []
        for res in tqdm(raw_results, total=len(emotion_inputs), desc="GoEmotions BERT"):
            scores = {d["label"]: d["score"] for d in res}
            nrc_scores = {
                nrc: float(sum(scores.get(g, 0.0) for g in go))
                for nrc, go in GOEMOTIONS_TO_NRC.items()
            }
            emotion_rows.append(nrc_scores)
            
            # If the raw max predicted score is 'neutral', classify dominant as 'neutral'
            raw_max_label = max(scores, key=scores.get)
            if raw_max_label == "neutral":
                dominant_emotions.append("neutral")
            else:
                dominant_emotions.append(max(nrc_scores, key=nrc_scores.get))

        emotion_df = pd.DataFrame(emotion_rows, index=df.index)
        emotion_df = emotion_df[NRC_EMOTIONS]
    else:
        emotion_df = pd.DataFrame(df[text_col].apply(score_emotions).tolist(), index=df.index)

    emotion_df.columns = [f'emotion_{e}' for e in NRC_EMOTIONS]
    # Drop any generic emotion columns from a previous backend pass so a second
    # call (e.g. running both backends) overwrites rather than duplicates them.
    stale_cols = [c for c in emotion_df.columns if c in df.columns]
    if 'dominant_emotion' in df.columns:
        stale_cols.append('dominant_emotion')
    if stale_cols:
        df = df.drop(columns=stale_cols)
    df = pd.concat([df, emotion_df], axis=1)

    if backend == "bert":
        df['dominant_emotion'] = dominant_emotions
    else:
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


def run_nlp_enrichment(df: pd.DataFrame, output_filepath: str = "data/sinner_alcaraz_processed.csv", emotion_backend: str = "both") -> dict:
    """Enrich crawled posts with cleaned text, RoBERTa sentiment, emotions, NER and entity links.

    Writes the enriched DataFrame to output_filepath and returns it together with
    per-player sentiment score buckets.

    The ``emotion_backend`` parameter selects the emotion-scoring engine(s):
      - "nrc"  : NRCLex affect frequencies only.
      - "bert" : GoEmotions (RoBERTa) only; also aliased to `bert_emotion_*` /
                 `bert_dominant_emotion` columns.
      - "both" : (default) run both backends. The generic `emotion_*` /
                 `dominant_emotion` columns hold the most recently computed
                 (BERT) backend, and both are additionally preserved under the
                 `nrc_emotion_*` / `nrc_dominant_emotion` and `bert_emotion_*` /
                 `bert_dominant_emotion` column families for side-by-side
                 comparison.
    """
    df = df.copy()
    total_posts = len(df)
    print(f"[NLP] Starting BERT/RoBERTa NLP enrichment for {total_posts} posts...")

    print("[NLP] Step 1/6: Cleaning and preprocessing post text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['preprocessed_text'] = df['text'].apply(preprocess)
    print("[NLP] Step 1/6 completed.")

    print("[NLP] Step 2/6: Scoring RoBERTa sentiments on GPU...")
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
    print("[NLP] Step 2/6 completed.")

    if emotion_backend == "nrc":
        print("[NLP] Step 3/6: Running NRC Emotion Lexicon analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='nrc')
        print("[NLP] Step 3/6 completed.")
        # bert_emotion_* columns will be absent — that is expected.
    elif emotion_backend == "bert":
        print("[NLP] Step 3/6: Running GoEmotions BERT emotion analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='bert')
        # Alias bert_ prefix columns from the generic emotion_* ones:
        for e in NRC_EMOTIONS:
            df[f'bert_emotion_{e}'] = df[f'emotion_{e}']
        df['bert_dominant_emotion'] = df['dominant_emotion']
        print("[NLP] Step 3/6 completed.")
    else:
        print("[NLP] Step 3/6: Running NRC Emotion Lexicon analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='nrc')
        for e in NRC_EMOTIONS:
            df[f'nrc_emotion_{e}'] = df[f'emotion_{e}']
        df['nrc_dominant_emotion'] = df['dominant_emotion']
        print("[NLP] Step 3/6 completed.")

        print("[NLP] Step 4/6: Running GoEmotions BERT emotion analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='bert')
        for e in NRC_EMOTIONS:
            df[f'bert_emotion_{e}'] = df[f'emotion_{e}']
        df['bert_dominant_emotion'] = df['dominant_emotion']
        print("[NLP] Step 4/6 completed.")

    print("[NLP] Step 5/6: Extracting Named Entities (spaCy NER)...")
    df['entities'] = df['cleaned_text'].apply(extract_ner)
    print("[NLP] Step 5/6 completed.")

    print("[NLP] Step 6/6: Linking entities to DBpedia resources...")
    df['linked_entities'] = df['cleaned_text'].apply(link_entities_dbpedia)
    print("[NLP] Step 6/6 completed.")

    scores = derive_player_sentiment_scores(df)

    df.to_csv(output_filepath, index=False)
    print(f"[NLP] Saved processed dataset to {output_filepath}")
    return {"df": df, **scores}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_community_emotion_profiles(
    df: pd.DataFrame,
    Gu,
    node_to_community: dict,
    output_dir: str = "plots",
    title_suffix: str = "",
    top_k: int = 5,
    sort_by: str = "post_volume",
    cmap_name: str = "tab20",
    backend: str = "nrc",
) -> None:
    """Plot average emotion profiles for the top-k communities of the filtered graph Gu.

    Communities are selected either by post volume or by node count (sort_by).
    The ``backend`` chooses which emotion column family to chart:
      - "nrc"  : the generic `emotion_*` columns (subtitle "NRC Emotion Lexicon").
      - "bert" : the `bert_emotion_*` columns (subtitle "GoEmotions (BERT)").
    The output filename is suffixed with the backend name.
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

    if backend == "bert":
        col_prefix = 'bert_emotion_'
    else:
        # Check if specific nrc columns are present to avoid mixups with BERT overwrites
        col_prefix = 'nrc_emotion_' if 'nrc_emotion_fear' in df.columns else 'emotion_'

    emotion_cols = [f'{col_prefix}{e}' for e in NRC_EMOTIONS]
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
            records.append({"Community": label, "Emotion": col.replace(col_prefix, ''), "Score": avg_scores[col]})

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        return

    subtitle = "GoEmotions (BERT)" if backend == "bert" else "NRC Emotion Lexicon"
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Score", hue="Community", palette=custom_palette)
    plt.title(f"Average Emotion Profiles per Community{title_suffix}\n({subtitle})", fontsize=14, pad=15)
    plt.xlabel("Emotion Category")
    plt.ylabel("Mean Normalized Score")
    plt.legend(title="Community")
    plt.tight_layout()

    # Clean the title_suffix to isolate the algorithm name (Louvain / Infomap)
    if "Louvain" in title_suffix:
        algo = "Louvain"
    elif "Infomap" in title_suffix:
        algo = "Infomap"
    else:
        algo = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
        if algo.startswith("_"):
            algo = algo[1:]

    filename = f"community_emotion_{algo}_{backend}.png"
    save_plot_copies(filename, output_dir=output_dir)
    plt.close()


def plot_emotion_backend_comparison(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot a grouped bar chart comparing dominant-emotion distributions of both backends.

    Requires the `nrc_dominant_emotion` and `bert_dominant_emotion` columns produced
    by ``run_nlp_enrichment(..., emotion_backend='both')``; if either is missing the
    function prints a warning and returns without plotting.
    """
    if 'nrc_dominant_emotion' not in df.columns or 'bert_dominant_emotion' not in df.columns:
        print("Warning: 'nrc_dominant_emotion'/'bert_dominant_emotion' columns not found. Skipping plot.")
        return

    emotion_index = NRC_EMOTIONS + ["neutral"]
    nrc_dist = df['nrc_dominant_emotion'].value_counts(normalize=True).reindex(emotion_index, fill_value=0.0)
    bert_dist = df['bert_dominant_emotion'].value_counts(normalize=True).reindex(emotion_index, fill_value=0.0)

    records = []
    for emotion in emotion_index:
        records.append({"Emotion": emotion, "Share of posts": float(nrc_dist[emotion]), "Backend": "NRC"})
        records.append({"Emotion": emotion, "Share of posts": float(bert_dist[emotion]), "Backend": "BERT"})
    df_plot = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Share of posts", hue="Backend")
    plt.title("Dominant Emotion Distribution: NRC vs GoEmotions (BERT)", fontsize=14, pad=15)
    plt.xlabel("Emotion Category")
    plt.ylabel("Share of posts")
    plt.legend(title="Backend")
    plt.tight_layout()

    save_plot_copies("emotion_backend_comparison.png", output_dir=output_dir)
    plt.close()


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot a donut chart showing the overall distribution of sentiment categories."""
    if 'sentiment_category' not in df.columns:
        print("Warning: 'sentiment_category' column not found in DataFrame. Skipping plot.")
        return

    counts = df['sentiment_category'].value_counts()
    categories = counts.index.tolist()
    sizes = counts.values.tolist()

    # Color palette matching emerald green (pos), slate gray (neu), rose red (neg)
    colors_map = {
        'positive': '#2ec4b6',
        'neutral': '#a0aec0',
        'negative': '#e63946'
    }
    colors = [colors_map.get(cat, '#cbd5e0') for cat in categories]

    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=categories, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.75,
        textprops=dict(color="black", fontsize=12, weight="bold")
    )
    
    # Add center circle to make it a donut
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Style percentage text to be readable
    for autotext in autotexts:
        autotext.set_color('white')

    plt.title("Overall Sentiment Distribution\n(RoBERTa Sentiment Classifier)", fontsize=14, pad=20, weight="bold")
    plt.tight_layout()
    save_plot_copies("sentiment_distribution.png", output_dir=output_dir)
    plt.close()


def plot_sentiment_over_time(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot daily average sentiment compound scores for Sinner vs. Alcaraz over time."""
    if 'created_at' not in df.columns or 'sentiment_compound' not in df.columns or 'linked_entities' not in df.columns:
        print("Warning: Required columns for sentiment over time not found. Skipping plot.")
        return

    df = df.copy()
    df['date'] = pd.to_datetime(df['created_at'], format='mixed').dt.date

    sinner_records = []
    alcaraz_records = []

    for _, row in df.iterrows():
        linked_ents = row['linked_entities']
        if isinstance(linked_ents, str):
            import ast
            try:
                linked_ents = ast.literal_eval(linked_ents)
            except Exception:
                linked_ents = []
        
        uris = {ent['uri'] for ent in linked_ents if isinstance(ent, dict)}
        text_lower = str(row['text']).lower()
        
        is_sinner = SINNER_URI in uris or any(k in text_lower for k in SINNER_KEYWORDS)
        is_alcaraz = ALCARAZ_URI in uris or any(k in text_lower for k in ALCARAZ_KEYWORDS)

        if is_sinner:
            sinner_records.append({"date": row['date'], "sentiment": row['sentiment_compound']})
        if is_alcaraz:
            alcaraz_records.append({"date": row['date'], "sentiment": row['sentiment_compound']})

    df_sinner = pd.DataFrame(sinner_records)
    df_alcaraz = pd.DataFrame(alcaraz_records)

    sinner_trend = df_sinner.groupby('date')['sentiment'].mean() if not df_sinner.empty else pd.Series()
    alcaraz_trend = df_alcaraz.groupby('date')['sentiment'].mean() if not df_alcaraz.empty else pd.Series()

    sinner_trend = sinner_trend.sort_index()
    alcaraz_trend = alcaraz_trend.sort_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 1. Plot Post Volume on a secondary Y-axis (ax2) in the background
    ax2 = ax1.twinx()
    
    sinner_vol = df_sinner.groupby('date')['sentiment'].count() if not df_sinner.empty else pd.Series()
    alcaraz_vol = df_alcaraz.groupby('date')['sentiment'].count() if not df_alcaraz.empty else pd.Series()
    
    sinner_vol = sinner_vol.sort_index()
    alcaraz_vol = alcaraz_vol.sort_index()
    
    if not sinner_vol.empty:
        ax2.fill_between(sinner_vol.index, sinner_vol.values, alpha=0.12, color='#f39c12', label='Sinner Post Volume')
    if not alcaraz_vol.empty:
        ax2.fill_between(alcaraz_vol.index, alcaraz_vol.values, alpha=0.12, color='#00b4d8', label='Alcaraz Post Volume')
        
    ax2.set_ylabel("Daily Post Volume (Shaded)", color='gray', fontsize=11, labelpad=10)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(False)  # Disable grid lines for secondary axis to avoid clutter

    # 2. Plot Average Sentiment Compound Scores on the primary Y-axis (ax1)
    if not sinner_trend.empty:
        ax1.plot(sinner_trend.index, sinner_trend.values, marker='o', linewidth=2.5, color='#d35400', label='Jannik Sinner Sentiment')
    if not alcaraz_trend.empty:
        ax1.plot(alcaraz_trend.index, alcaraz_trend.values, marker='s', linewidth=2.5, color='#023e8a', label='Carlos Alcaraz Sentiment')

    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Import and annotate US Open 2025 key match events from utils
    from utils import US_OPEN_EVENTS
    
    # Force drawing/scaling so we can read the correct Y limits on the primary axis
    ax1.relim()
    ax1.autoscale_view()
    ymin, ymax = ax1.get_ylim()
    text_y = ymax - (ymax - ymin) * 0.08
    
    for date_str, label, color in US_OPEN_EVENTS:
        try:
            event_date = pd.to_datetime(date_str).date()
            ax1.axvline(event_date, color=color, linestyle=':', alpha=0.5, linewidth=1.2)
            ax1.text(
                event_date,
                text_y,
                f"  {label}",
                rotation=90,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=9,
                color=color,
                weight='semibold',
                alpha=0.8
            )
        except Exception as e:
            print(f"Warning: Could not annotate event {label} at {date_str}: {e}")

    plt.title("Sentiment Trajectory & Post Volume Over Time (US Open 2025)\n(Daily Avg RoBERTa Sentiment vs. Post Volume)", fontsize=14, pad=15, weight="bold")
    ax1.set_xlabel("Date", fontsize=11, labelpad=10)
    ax1.set_ylabel("Average Sentiment Compound Score (Lines)", fontsize=11, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_plot_copies("sentiment_over_time.png", output_dir=output_dir)
    plt.close()




