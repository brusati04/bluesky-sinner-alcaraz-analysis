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


def analyze_community_echo_chambers(df, Gu, node_to_community, communities, output_dir="plots", title_suffix=""):
    """
    Study 1: Polarization and Echo Chamber Analysis.
    Classify community members by their stance leaning and plot distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    stance_leanings = nx.get_node_attributes(Gu, "stance_leaning")

    user_community = {}
    for node in Gu.nodes():
        if node in node_to_community:
            user_community[node] = node_to_community[node]

    comm_leanings = {}
    for node, cid in user_community.items():
        leaning = stance_leanings.get(node, "neutral")
        if cid not in comm_leanings:
            comm_leanings[cid] = []
        comm_leanings[cid].append(leaning)

    top_comms = sorted(comm_leanings.keys(), key=lambda k: len(comm_leanings[k]), reverse=True)[:5]

    plot_data = []
    for cid in top_comms:
        counts = pd.Series(comm_leanings[cid]).value_counts()
        size = len(comm_leanings[cid])
        plot_data.append({
            "Community": f"Comm {cid}\n(N={size})",
            "Pro-Sinner": counts.get("sinner", 0) / size * 100,
            "Pro-Alcaraz": counts.get("alcaraz", 0) / size * 100,
            "Neutral": counts.get("neutral", 0) / size * 100
        })
    df_plot = pd.DataFrame(plot_data)

    if df_plot.empty:
        return

    ax = df_plot.plot(x="Community", y=["Pro-Sinner", "Neutral", "Pro-Alcaraz"], kind="bar", stacked=True,
                      color=["#3498db", "#bdc3c7", "#e67e22"], figsize=(10, 6), edgecolor="white")
    plt.title(f"User Stance Distribution per Community{title_suffix}\n(Echo Chamber Profile)", fontsize=14, pad=15)
    plt.xlabel("Community")
    plt.ylabel("Percentage of Users (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(loc="upper right")
    plt.tight_layout()

    suffix_filename = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
    fname = f"community_echo_chambers{suffix_filename}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_community_emotion_profiles(df, Gu, node_to_community, communities, output_dir="plots", title_suffix=""):
    """
    Study 2: Emotional Profiling of Communities.
    Compare average scores of 8 NRC emotions across the top 3 communities.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    author_community = {}
    for _, row in df.iterrows():
        handle = row.get('author_handle')
        if handle and handle in node_to_community and handle in Gu.nodes():
            author_community[handle] = node_to_community[handle]

    df = df.copy()
    df['community_id'] = df['author_handle'].map(author_community)
    df_with_comm = df.dropna(subset=['community_id']).copy()

    from collections import Counter
    counts = Counter(node_to_community.values())
    top_comms = [cid for cid, count in counts.most_common(3)]

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]

    records = []
    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
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
        label = f"Comm {int(cid)} (N={len(comm_posts)})"
        custom_palette[label] = color_map.get(cid, "#bdc3c7")

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


def plot_community_aspect_sentiment(df, Gu, node_to_community, communities, output_dir="plots", title_suffix=""):
    """
    Study 3: Aspect-Based Sentiment Analysis (ABSA) across Communities.
    Compare sentiment profiles towards Match Play, controversies, and general rivalry.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    aspects = {
        "Match Play/Contest": ["match", "set", "points", "serve", "game", "win", "play", "court"],
        "Controversies/Doping": ["doping", "clostebol", "test", "ban", "steroid", "innocent", "guilty", "controversy"],
        "Player Rivalry": ["rivalry", "rival", "face-off", "vs", "versus"]
    }

    df = df.copy()

    for name, keywords in aspects.items():
        pattern = "|".join(keywords)
        df[name] = df['text'].str.lower().str.contains(pattern, na=False)

    author_community = {}
    for _, row in df.iterrows():
        handle = row.get('author_handle')
        if handle and handle in node_to_community and handle in Gu.nodes():
            author_community[handle] = node_to_community[handle]

    df['community_id'] = df['author_handle'].map(author_community)
    df_with_comm = df.dropna(subset=['community_id', 'sentiment_compound']).copy()

    from collections import Counter
    counts = Counter(node_to_community.values())
    top_comms = [cid for cid, count in counts.most_common(3)]

    records = []
    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
        for aspect_name in aspects.keys():
            aspect_posts = comm_posts[comm_posts[aspect_name]]
            if len(aspect_posts) >= 2:
                records.append({
                    "Community": f"Comm {int(cid)} (N={len(comm_posts)})",
                    "Aspect": aspect_name,
                    "Avg Sentiment": aspect_posts["sentiment_compound"].mean(),
                    "Count": len(aspect_posts)
                })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        return

    from social_network_analysis import get_community_color_map
    color_map = get_community_color_map(node_to_community)
    custom_palette = {}
    for cid in top_comms:
        comm_posts = df_with_comm[df_with_comm['community_id'] == cid]
        label = f"Comm {int(cid)} (N={len(comm_posts)})"
        custom_palette[label] = color_map.get(cid, "#bdc3c7")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot, x="Aspect", y="Avg Sentiment", hue="Community", palette=custom_palette)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Aspect-Based Sentiment per Community{title_suffix}\n(Comparison across Core Topics)", fontsize=14, pad=15)
    plt.xlabel("Aspect Topic")
    plt.ylabel("Mean VADER Compound Sentiment")
    plt.ylim(-1, 1)
    plt.legend(title="Community")
    plt.tight_layout()

    suffix_filename = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
    fname = f"community_aspect_sentiment{suffix_filename}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_subjectivity_vs_centrality(df, df_cent, output_dir="plots"):
    """
    Study 4: Subjectivity vs. Objectivity of Influencers and Hubs.
    Linguistically scores subjectivity and correlates it with PageRank centrality.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    from preprocessing import nlp

    def get_subjectivity_ratio(text):
        if not text or pd.isna(text) or str(text).strip() == "":
            return 0.0
        doc = nlp(str(text))
        pos_tags = [token.pos_ for token in doc]
        if not pos_tags:
            return 0.0
        adj_adv = sum(1 for tag in pos_tags if tag in ["ADJ", "ADV"])
        return adj_adv / len(pos_tags)

    df = df.copy()
    df["subjectivity"] = df["text"].apply(get_subjectivity_ratio)

    user_subj = df.groupby("author_handle")["subjectivity"].mean().to_dict()

    df_merged = df_cent.copy()
    df_merged["subjectivity"] = df_merged["user"].map(user_subj).fillna(0.0)

    df_merged = df_merged[df_merged["user"].isin(user_subj.keys())].copy()

    if df_merged.empty:
        return

    pr_scores = df_merged["pagerank"]
    subj_scores = df_merged["subjectivity"]

    try:
        corr_coef, p_val = stats.pearsonr(pr_scores, subj_scores)
    except Exception:
        corr_coef, p_val = 0.0, 1.0

    plt.figure(figsize=(9, 6))
    ax = sns.regplot(data=df_merged, x="pagerank", y="subjectivity",
                     scatter_kws={"alpha": 0.6, "color": "#1abc9c"},
                     line_kws={"color": "#e74c3c", "linewidth": 2})
    ax.set_xscale('log')
    plt.title(f"User Subjectivity vs. PageRank Influence\n(Pearson r={corr_coef:.4f}, p={p_val:.4f})", fontsize=14, pad=15)
    plt.xlabel("PageRank Influence (Log Scale)")
    plt.ylabel("Linguistic Subjectivity Ratio\n(ADJ + ADV / Total Tokens)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    fname = "subjectivity_vs_centrality.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STANCE PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_players(row: pd.Series) -> tuple[bool, bool]:
    linked_ents = row.get("linked_entities", [])

    if isinstance(linked_ents, str):
        try:
            linked_ents = ast.literal_eval(linked_ents)
        except Exception:
            linked_ents = []

    if not isinstance(linked_ents, list):
        linked_ents = []

    uris = set()
    for ent in linked_ents:
        if isinstance(ent, dict):
            uris.add(ent.get("uri", ""))
        elif isinstance(ent, (tuple, list)) and len(ent) > 0:
            uris.add(ent[0])

    is_sinner  = SINNER_URI in uris
    is_alcaraz = ALCARAZ_URI in uris

    # Keyword fallback
    text_tokens = set(str(row.get("text", "")).lower().split())
    if not is_sinner:
        is_sinner  = bool(text_tokens & SINNER_KEYWORDS)
    if not is_alcaraz:
        is_alcaraz = bool(text_tokens & ALCARAZ_KEYWORDS)

    return is_sinner, is_alcaraz


def _compute_post_stances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Player detection
    detection = df.apply(_detect_players, axis=1, result_type="expand")
    detection.columns = ["is_sinner", "is_alcaraz"]
    df = pd.concat([df, detection], axis=1)

    # Emotion signal
    pos_cols = [c for c in POSITIVE_EMOTIONS if c in df.columns]
    neg_cols = [c for c in NEGATIVE_EMOTIONS if c in df.columns]

    if pos_cols and neg_cols:
        df["emotion_signal"] = (
            df[pos_cols].sum(axis=1) - df[neg_cols].sum(axis=1)
        )
    else:
        df["emotion_signal"] = 0.0

    compound = pd.to_numeric(df.get("sentiment_compound", 0), errors="coerce").fillna(0)
    combined = W_SENTIMENT * compound + W_EMOTION * df["emotion_signal"]

    df["stance_sinner"]  = np.where(df["is_sinner"]  & ~df["is_alcaraz"],  combined, 0.0)
    df["stance_alcaraz"] = np.where(df["is_alcaraz"] & ~df["is_sinner"],   combined, 0.0)

    return df


def _compute_user_stances(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("author_handle")

    agg = pd.DataFrame({
        "mean_sinner":  grouped["stance_sinner"].mean(),
        "mean_alcaraz": grouped["stance_alcaraz"].mean(),
        "n_sinner":     grouped["is_sinner"].sum(),
        "n_alcaraz":    grouped["is_alcaraz"].sum(),
        "n_posts":      grouped["stance_sinner"].count(),
    })

    total_mentions = (agg["n_sinner"] + agg["n_alcaraz"]).replace(0, np.nan)
    agg["freq_sinner"]  =  agg["n_sinner"]  / total_mentions
    agg["freq_alcaraz"] =  agg["n_alcaraz"] / total_mentions
    agg.fillna(0.0, inplace=True)

    freq_lean = agg["freq_sinner"] - agg["freq_alcaraz"]

    agg["net_stance"] = (
        (W_SENTIMENT + W_EMOTION) * (agg["mean_sinner"] - agg["mean_alcaraz"])
        + W_FREQUENCY * freq_lean
    )

    return agg


def _classify_stances_direct(user_stances: pd.DataFrame, nodes: list) -> tuple[dict, dict]:
    f_prop = {}
    stance_leanings = {}
    
    for node in nodes:
        score = 0.0
        if node in user_stances.index:
            score = float(user_stances.loc[node, "net_stance"])
            
        f_prop[node] = score
        
        # Simple threshold classification: Pro-Sinner (> 0.05), Pro-Alcaraz (< -0.05), else Neutral
        if score > 0.05:
            stance_leanings[node] = "sinner"
        elif score < -0.05:
            stance_leanings[node] = "alcaraz"
        else:
            stance_leanings[node] = "neutral"
            
    return f_prop, stance_leanings


def _community_stance_profiles_simple(G: nx.Graph, f_prop: dict, stance_leanings: dict) -> pd.DataFrame:
    partition = nx.get_node_attributes(G, "community")
    if not partition:
        partition = {n: 0 for n in G.nodes()}
        
    records = []
    community_ids = set(partition.values())

    for cid in community_ids:
        members  = [n for n, c in partition.items() if c == cid]
        scores   = [f_prop.get(m, 0.0)        for m in members]
        leanings = [stance_leanings.get(m, "neutral") for m in members]

        mean_score    = float(np.mean(scores)) if scores else 0.0
        dominant      = max(set(leanings), key=leanings.count) if leanings else "neutral"
        homogeneity   = leanings.count(dominant) / len(leanings) if leanings else 0.0

        records.append({
            "community_id":  cid,
            "size":          len(members),
            "mean_stance":   mean_score,
            "dominant_lean": dominant,
            "homogeneity":   round(homogeneity, 3),
        })

    return pd.DataFrame(records).sort_values("size", ascending=False).reset_index(drop=True)


def _compute_polarization_metrics_simple(G: nx.Graph, stance_leanings: dict, f_prop: dict) -> dict:
    metrics = {}
    try:
        nx.set_node_attributes(G, stance_leanings, "stance_leaning")
        metrics["stance_assortativity"] = round(
            nx.attribute_assortativity_coefficient(G, "stance_leaning"), 4
        )
    except Exception as e:
        metrics["stance_assortativity"] = 0.0
        print(f"[WARNING] Assortativity calculation failed: {e}")

    cross_edges = sum(
        1 for u, v in G.edges()
        if stance_leanings.get(u) != stance_leanings.get(v)
    )
    total_edges = G.number_of_edges()
    metrics["cross_stance_ratio"] = round(cross_edges / total_edges, 4) if total_edges else 0.0

    scores = list(f_prop.values())
    metrics["score_std"] = round(float(np.std(scores)), 4) if scores else 0.0

    return metrics


def _print_summary_simple(stance_leanings: dict, f_prop: dict,
                          community_profiles: pd.DataFrame,
                          polarization: dict) -> None:
    total = len(stance_leanings)
    sinner_n  = sum(1 for l in stance_leanings.values() if l == "sinner")
    alcaraz_n = sum(1 for l in stance_leanings.values() if l == "alcaraz")
    neutral_n = sum(1 for l in stance_leanings.values() if l == "neutral")

    print("\n" + "=" * 60)
    print("STANCE CLASSIFICATION — RESULTS")
    print("=" * 60)
    print(f"{'Total users in graph:':<35} {total}")
    print(f"\n--- Stance Distribution (Direct Sentiment Average) ---")
    print(f"  {'Pro-Sinner:':<20} {sinner_n:>5}  ({sinner_n/total*100:.1f}%)")
    print(f"  {'Pro-Alcaraz:':<20} {alcaraz_n:>5}  ({alcaraz_n/total*100:.1f}%)")
    print(f"  {'Neutral:':<20} {neutral_n:>5}  ({neutral_n/total*100:.1f}%)")
    print(f"\n--- Polarization Metrics ---")
    for k, v in polarization.items():
        print(f"  {k:<30} {v}")
    print(f"\n--- Top Communities by Size ---")
    print(community_profiles.head(10).to_string(index=False))
    print("\n" + "=" * 60)


def run_stance_propagation(
    df: pd.DataFrame,
    G: nx.Graph,
    Gd: nx.DiGraph,
    df_cent: pd.DataFrame,
    filepath: str = "data/network_centrality_metrics.csv"
) -> dict:
    """
    Simplified user stance scoring. Computes user stance directly from average post sentiment.
    """
    print("\n" + "=" * 60)
    print("STANCE CLASSIFICATION (SIMPLIFIED) — STARTING")
    print("=" * 60)

    # Compute post-level and user-level sentiment scores
    df = _compute_post_stances(df)
    user_stances = _compute_user_stances(df)

    # Directly classify stances based on user post-sentiment averages
    f_prop, stance_leanings = _classify_stances_direct(user_stances, list(G.nodes()))
    
    # Save scores on graph nodes
    nx.set_node_attributes(G,  f_prop, "stance_score")
    nx.set_node_attributes(Gd, f_prop, "stance_score")
    nx.set_node_attributes(G,  stance_leanings, "stance_leaning")
    nx.set_node_attributes(Gd, stance_leanings, "stance_leaning")

    # Compute community profiles and basic metrics
    community_profiles = _community_stance_profiles_simple(G, f_prop, stance_leanings)
    polarization = _compute_polarization_metrics_simple(G, stance_leanings, f_prop)

    # Update and save to centrality dataframe
    df_cent["stance_score"]   = df_cent["user"].map(f_prop)
    df_cent["stance_leaning"] = df_cent["user"].map(stance_leanings)
    df_cent.to_csv(filepath, index=False)

    # Print summary
    _print_summary_simple(stance_leanings, f_prop, community_profiles, polarization)

    return {
        "df_cent":            df_cent,
        "stance_leanings":    stance_leanings,
        "f_prop":             f_prop,
        "community_profiles": community_profiles,
        "polarization":       polarization,
        "thresholds":         (-0.05, 0.05)
    }

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT & EMOTION PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_emotion_distribution(df: pd.DataFrame, output_dir: str = "plots",
                               text_col: str = 'cleaned_text',
                               title_suffix: str = "") -> None:
    """
    Plot 1: Bar chart of overall emotion frequencies across the corpus.
    Plot 2: Grouped bar chart comparing emotion profiles for Sinner-related vs Alcaraz-related posts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]

    # --- Plot 1: Corpus-level emotion distribution ---
    avg_emotions = df[emotion_cols].mean().sort_values(ascending=False)
    avg_emotions.index = [c.replace('emotion_', '') for c in avg_emotions.index]

    palette = {
        'joy': '#f1c40f', 'trust': '#2ecc71', 'anticipation': '#e67e22',
        'surprise': '#9b59b6', 'fear': '#e74c3c', 'sadness': '#3498db',
        'anger': '#c0392b', 'disgust': '#795548'
    }
    colors = [palette.get(e, '#95a5a6') for e in avg_emotions.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(avg_emotions.index, avg_emotions.values, color=colors, edgecolor='white')
    ax.set_title(f"NRC Emotion Profile — Corpus Level{title_suffix}", fontsize=14)
    ax.set_xlabel("Emotion Category")
    ax.set_ylabel("Mean Normalised Score")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fname = f"emotion_distribution{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Sinner vs Alcaraz emotion comparison ---
    sinner_mask = df[text_col].str.lower().str.contains('sinner|jannik', na=False)
    alcaraz_mask = df[text_col].str.lower().str.contains('alcaraz|carlitos', na=False)

    sinner_profile = df[sinner_mask][emotion_cols].mean()
    alcaraz_profile = df[alcaraz_mask][emotion_cols].mean()

    emotions_clean = [c.replace('emotion_', '') for c in emotion_cols]
    x = np.arange(len(emotions_clean))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, sinner_profile.values, width, label='Sinner posts',
           color='#3498db', alpha=0.85, edgecolor='white')
    ax.bar(x + width/2, alcaraz_profile.values, width, label='Alcaraz posts',
           color='#e74c3c', alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions_clean)
    ax.set_ylabel("Mean Normalised Emotion Score")
    ax.set_title(f"NRC Emotion Profile: Sinner vs Alcaraz Posts{title_suffix}")
    ax.legend()
    plt.tight_layout()
    fname2 = f"emotion_rivalry_comparison{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, fname2), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname2), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_distribution(df, output_dir="plots"):
    """
    Plot sentiment category distribution (positive, neutral, negative) as a bar chart.
    """
    plt.figure(figsize=(8, 5))
    sentiment_counts = df['sentiment_category'].value_counts()
    palette = {"positive": "#2ecc71", "neutral": "#bdc3c7", "negative": "#e74c3c"}
    colors = [palette.get(cat, "#95a5a6") for cat in sentiment_counts.index]

    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, hue=sentiment_counts.index, legend=False)
    plt.title("Sentiment Distribution of Bluesky Tennis Posts")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Posts")

    total_posts = len(df)
    for i, val in enumerate(sentiment_counts.values):
        percentage = (val / total_posts) * 100
        plt.text(i, val + (total_posts * 0.01), f"{val} ({percentage:.1f}%)", ha='center', fontweight='semibold')

    plt.tight_layout()
    save_plot_copies("sentiment_distribution.png")
    plt.close()


def plot_sentiment_over_time(df, output_dir="plots"):
    """
    Plot sentiment trajectory over time with rolling average and match milestones.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df.copy()
    df_sorted['created_at'] = pd.to_datetime(df_sorted['created_at'], errors='coerce', utc=True)
    df_sorted = df_sorted.dropna(subset=['created_at']).sort_values(by="created_at")
    df_sorted['rolling_sentiment'] = df_sorted['sentiment_compound'].rolling(
        window=min(30, len(df_sorted)), min_periods=5
    ).mean()

    ax.plot(df_sorted['created_at'], df_sorted['rolling_sentiment'],
            color="#34495e", linewidth=2.5, label="30-Post Rolling Mean")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6, label="Neutral baseline")

    ax.fill_between(df_sorted['created_at'], 0, df_sorted['rolling_sentiment'],
                    where=(df_sorted['rolling_sentiment'] >= 0),
                    color='#2ecc71', alpha=0.15, label="Positive region")
    ax.fill_between(df_sorted['created_at'], 0, df_sorted['rolling_sentiment'],
                    where=(df_sorted['rolling_sentiment'] < 0),
                    color='#e74c3c', alpha=0.15, label="Negative region")

    for event_date_str, label, color in US_OPEN_EVENTS:
        event_dt = pd.to_datetime(event_date_str, utc=True)
        if df_sorted['created_at'].min() <= event_dt <= df_sorted['created_at'].max():
            ax.axvline(x=event_dt, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
            y_pos = ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.3
            ax.text(event_dt, y_pos, label,
                    rotation=90, ha='right', va='top',
                    fontsize=7, color=color, alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_title("Sentiment Dynamics Over Time (30-Post Rolling Average)\nWith US Open 2025 Key Match Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("VADER Compound Sentiment Score")
    ax.legend(loc='lower right', fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_plot_copies("sentiment_over_time.png")
    plt.close()


def plot_rivalry_comparison(sinner_scores, alcaraz_scores, output_dir="plots"):
    """
    Rivalry volume vs sentiment averages bar/line comparison plot.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    categories = ["Jannik Sinner", "Carlos Alcaraz"]
    avg_sinner_sent = np.mean(sinner_scores) if sinner_scores else 0.0
    avg_alcaraz_sent = np.mean(alcaraz_scores) if alcaraz_scores else 0.0
    averages = [avg_sinner_sent, avg_alcaraz_sent]
    mention_counts = [len(sinner_scores), len(alcaraz_scores)]

    x = np.arange(len(categories))
    width = 0.35

    color1 = '#3498db'
    ax1.set_xlabel('Tennis Player')
    ax1.set_ylabel('Number of Mentions', color=color1)
    bars1 = ax1.bar(x, mention_counts, width, label='Mention Volume', color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)

    ax2 = ax1.twinx()
    color2 = '#d35400'
    ax2.set_ylabel('Average VADER Compound Sentiment', color=color2)
    line2 = ax2.plot(x, averages, color=color2, marker='o', markersize=10, linewidth=3, label='Avg Sentiment')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.5, 0.5)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, height/2.0, f'{int(height)}', ha='center', va='center', color='white', fontweight='bold')

    for i, val in enumerate(averages):
        ax2.text(i, val + 0.05, f'{val:.3f}', ha='center', color=color2, fontweight='bold')

    plt.title("Rivalry Comparison: Mentions vs. Public Sentiment Profile")
    fig.tight_layout()
    save_plot_copies("rivalry_comparison.png")
    plt.close()


def plot_median_sentiment_over_rounds(df, output_dir="plots"):
    """
    Dual-axis bar + line chart: Median Sentiment + Post Volume per US Open Round.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)

    labels, medians, volumes = [], [], []
    for r in US_OPEN_ROUNDS:
        start = pd.to_datetime(r["start"], utc=True)
        end   = pd.to_datetime(r["end"], utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask  = (df['created_at'] >= start) & (df['created_at'] <= end)
        window_df = df.loc[mask]
        med = window_df['sentiment_compound'].median()
        vol = len(window_df)
        if pd.notna(med):
            labels.append(r["label"])
            medians.append(med)
            volumes.append(vol)

    if not labels:
        print("[WARNING] No data found for US Open round windows. Skipping median sentiment plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))

    # Bars: median sentiment
    bars = ax1.bar(x, medians, color='#6fa8dc', alpha=0.85, edgecolor='white', zorder=2)
    ax1.set_ylabel("Med. Sentiment Value", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)

    # Line: post volume
    ax2 = ax1.twinx()
    ax2.plot(x, volumes, color='#e67e22', marker='o', linewidth=2.5, zorder=3)
    ax2.set_ylabel("Post Volume", fontsize=12)

    ax1.set_title("Median Sentiment Value Over US Open 2025 Rounds", fontsize=14, pad=15)
    fig.tight_layout()
    save_plot_copies("median_sentiment_over_rounds.png")
    plt.close()


def plot_fanbase_wordclouds(df, df_cent, output_dir="plots"):
    """
    Generate Word Clouds for the overall conversation and fanbase subsets.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[WARNING] 'wordcloud' package not installed. Skipping Word Cloud generation.")
        return

    # Map user leaning
    user_leaning = df_cent.set_index('user')['stance_leaning'].to_dict()
    df_with_leaning = df.copy()
    df_with_leaning['user_leaning'] = df_with_leaning['author_handle'].map(user_leaning)

    # Subset posts
    overall_posts = df_with_leaning['preprocessed_text'].dropna().tolist()
    sinner_posts = df_with_leaning[df_with_leaning['user_leaning'] == 'sinner']['preprocessed_text'].dropna().tolist()
    alcaraz_posts = df_with_leaning[df_with_leaning['user_leaning'] == 'alcaraz']['preprocessed_text'].dropna().tolist()

    overall_text_raw = " ".join(overall_posts)
    overall_text = " ".join(overall_posts)
    sinner_text = " ".join(sinner_posts)
    alcaraz_text = " ".join(alcaraz_posts)

    # Remove query terms to expose actual discussion topics
    query_terms = ['sinner', 'jannik', 'alcaraz', 'carlos', 'carlitos', 'tennis', 'match', 'play', 'player', 'set']
    for term in query_terms:
        overall_text = re.sub(rf'\b{term}\b', '', overall_text, flags=re.IGNORECASE)
        sinner_text = re.sub(rf'\b{term}\b', '', sinner_text, flags=re.IGNORECASE)
        alcaraz_text = re.sub(rf'\b{term}\b', '', alcaraz_text, flags=re.IGNORECASE)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    # 1a. Generate Standalone Overall Word Cloud (Filtered - Topics)
    if len(overall_text.strip()) > 10:
        plt.figure(figsize=(10, 8))
        wc_overall = WordCloud(width=800, height=800, background_color='white', colormap='viridis', max_words=100).generate(overall_text)
        plt.imshow(wc_overall, interpolation='bilinear')
        plt.title("Overall Conversation Word Cloud (Topic Words - Player Names Filtered)", fontsize=16, pad=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_overall.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join("report", "wordcloud_overall.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 1b. Generate Standalone Overall Word Cloud (Raw - Unfiltered)
    if len(overall_text_raw.strip()) > 10:
        plt.figure(figsize=(10, 8))
        wc_overall_raw = WordCloud(width=800, height=800, background_color='white', colormap='viridis', max_words=100).generate(overall_text_raw)
        plt.imshow(wc_overall_raw, interpolation='bilinear')
        plt.title("Overall Conversation Word Cloud (Raw - Player Names Included)", fontsize=16, pad=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_overall_raw.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join("report", "wordcloud_overall_raw.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Generate 3-Panel Fanbase Comparison Word Cloud (Overall vs. Sinner vs. Alcaraz)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: Overall
    if len(overall_text.strip()) > 10:
        wc_all = WordCloud(width=800, height=800, background_color='white', colormap='viridis', max_words=80).generate(overall_text)
        ax1.imshow(wc_all, interpolation='bilinear')
        ax1.set_title("Overall Conversation (All Users)", fontsize=16, pad=10)
    else:
        ax1.text(0.5, 0.5, "Not enough data", ha='center', va='center')
    ax1.axis('off')

    # Panel 2: Sinner
    if len(sinner_text.strip()) > 10:
        wc_s_obj = WordCloud(width=800, height=800, background_color='white', colormap='Blues', max_words=80).generate(sinner_text)
        ax2.imshow(wc_s_obj, interpolation='bilinear')
        ax2.set_title("Pro-Jannik Sinner Fanbase", fontsize=16, pad=10)
    else:
        ax2.text(0.5, 0.5, "Not enough Sinner fanbase data", ha='center', va='center')
    ax2.axis('off')

    # Panel 3: Alcaraz
    if len(alcaraz_text.strip()) > 10:
        wc_a_obj = WordCloud(width=800, height=800, background_color='white', colormap='Oranges', max_words=80).generate(alcaraz_text)
        ax3.imshow(wc_a_obj, interpolation='bilinear')
        ax3.set_title("Pro-Carlos Alcaraz Fanbase", fontsize=16, pad=10)
    else:
        ax3.text(0.5, 0.5, "Not enough Alcaraz fanbase data", ha='center', va='center')
    ax3.axis('off')

    fig.suptitle("Word Cloud Fanbase Comparison & Global Conversation", fontsize=22, y=1.02)
    fig.tight_layout()

    fname = "wordcloud_fanbase_comparison.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join("report", fname), dpi=300, bbox_inches='tight')
    plt.close()
