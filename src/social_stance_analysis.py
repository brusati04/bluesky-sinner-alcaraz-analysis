# stance_analysis.py
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional

# Import constants from the NLP module
from social_sentiment_analysis import (
    W_SENTIMENT,
    W_FREQUENCY,
    SINNER_URI,
    ALCARAZ_URI,
    SINNER_KEYWORDS,
    ALCARAZ_KEYWORDS,
)

# For community colors
from social_network_analysis import get_community_color_map

# For saving plots
from utils import save_plot_copies

# Sentence tokenization
try:
    from nltk.tokenize import sent_tokenize
except LookupError:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize

# Configuration Constants
STANCE_THRESHOLD = 0.05
MIN_POSTS_FOR_STANCE = 1
DEBUG = True

# Lazily-initialised RoBERTa sentiment classifier
_sentiment_clf_roberta = None

def _get_sentiment_clf_roberta():
    global _sentiment_clf_roberta
    if _sentiment_clf_roberta is None:
        import torch
        from transformers import pipeline
        # Set cache dir
        workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["HF_HOME"] = os.path.abspath(os.path.join(workspace_path, ".huggingface_cache"))
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"[STANCE] Initializing CardiffNLP RoBERTa sentiment model for sentence scoring on device={device}...")
        _sentiment_clf_roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            top_k=None
        )
    return _sentiment_clf_roberta


def _sentence_sentiment(sentence: str) -> float:
    """Return CardiffNLP RoBERTa compound score (P(positive) - P(negative)) for a single sentence."""
    if not sentence or not sentence.strip():
        return 0.0
    
    from preprocessing import clean_text_bert
    cleaned = clean_text_bert(sentence)
    if not cleaned.strip():
        return 0.0
        
    clf = _get_sentiment_clf_roberta()
    res = clf(cleaned, truncation=True, max_length=512)
    scores_dict = {d['label']: d['score'] for d in res[0]}
    
    if 'LABEL_0' in scores_dict:
        p_neg = scores_dict.get('LABEL_0', 0.0)
        p_pos = scores_dict.get('LABEL_2', 0.0)
    else:
        p_neg = scores_dict.get('negative', 0.0)
        p_pos = scores_dict.get('positive', 0.0)
        
    return p_pos - p_neg


def _sentence_mentions(sentence: str) -> Tuple[bool, bool]:
    """Return (mentions_sinner, mentions_alcaraz) for a sentence."""
    text_lower = sentence.lower()
    has_sinner = any(kw in text_lower for kw in SINNER_KEYWORDS)
    has_alcaraz = any(kw in text_lower for kw in ALCARAZ_KEYWORDS)
    return has_sinner, has_alcaraz


def _split_dual_mention(text: str, sentence_scores: Dict[str, float]) -> Tuple[float, float]:
    """
    Given a post mentioning both players, return (sinner_score, alcaraz_score)
    based on sentence-level sentiment using precomputed sentence scores.
    """
    sentences = sent_tokenize(str(text))
    
    sinner_scores = []
    alcaraz_scores = []
    
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        mentions_sinner, mentions_alcaraz = _sentence_mentions(sent_clean)
        sentiment = sentence_scores.get(sent_clean, 0.0)
        
        if mentions_sinner and not mentions_alcaraz:
            sinner_scores.append(sentiment)
        elif mentions_alcaraz and not mentions_sinner:
            alcaraz_scores.append(sentiment)
        elif mentions_sinner and mentions_alcaraz:
            sinner_scores.append(sentiment * 0.5)
            alcaraz_scores.append(sentiment * 0.5)
    
    sinner_score = np.mean(sinner_scores) if sinner_scores else 0.0
    alcaraz_score = np.mean(alcaraz_scores) if alcaraz_scores else 0.0
    
    return sinner_score, alcaraz_score


# ═════════════════════════════════════════════════════════════════════
# PLAYER DETECTION
# ═════════════════════════════════════════════════════════════════════

def detect_players(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_sinner / is_alcaraz boolean columns using linked_entities and keyword fallback."""
    def _detect_row(row):
        uris = set()
        linked = row.get('linked_entities', [])
        if isinstance(linked, list):
            for ent in linked:
                if isinstance(ent, dict):
                    uris.add(ent.get('uri', ''))
                elif isinstance(ent, (tuple, list)) and len(ent) > 0:
                    uris.add(ent[0])
        
        is_sinner = SINNER_URI in uris
        is_alcaraz = ALCARAZ_URI in uris
        
        if not is_sinner and not is_alcaraz:
            text = str(row.get('text', '')).lower()
            is_sinner = any(kw in text for kw in SINNER_KEYWORDS)
            is_alcaraz = any(kw in text for kw in ALCARAZ_KEYWORDS)
        
        return is_sinner, is_alcaraz
    
    detection = df.apply(_detect_row, axis=1, result_type='expand')
    detection.columns = ['is_sinner', 'is_alcaraz']
    return pd.concat([df, detection], axis=1)


# ═════════════════════════════════════════════════════════════════════
# POST-LEVEL STANCE SCORES
# ═════════════════════════════════════════════════════════════════════

def compute_post_stances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stance_sinner / stance_alcaraz columns.
    - Single-mention posts: full RoBERTa compound assigned to that player
    - Dual-mention posts: sentence-level sentiment split
    - No-mention posts: 0 for both
    """
    compound = df['sentiment_compound'].fillna(0)
    
    single_sinner = df['is_sinner'] & ~df['is_alcaraz']
    single_alcaraz = df['is_alcaraz'] & ~df['is_sinner']
    both_players = df['is_sinner'] & df['is_alcaraz']
    neither = ~df['is_sinner'] & ~df['is_alcaraz']
    
    df['stance_sinner'] = 0.0
    df['stance_alcaraz'] = 0.0
    
    df.loc[single_sinner, 'stance_sinner'] = compound[single_sinner]
    df.loc[single_alcaraz, 'stance_alcaraz'] = compound[single_alcaraz]
    
    if both_players.any():
        if DEBUG:
            print(f"\n[STANCE] Extracting sentences for {both_players.sum()} dual-mention posts...")
        dual_indices = df[both_players].index
        dual_texts = df.loc[both_players, 'text']
        
        post_sentences = {}
        all_unique_sentences = set()
        
        for idx, text in zip(dual_indices, dual_texts):
            sents = sent_tokenize(str(text))
            post_sentences[idx] = sents
            for s in sents:
                s_clean = s.strip()
                if s_clean:
                    all_unique_sentences.add(s_clean)
                    
        # Score all unique sentences in batch
        unique_list = list(all_unique_sentences)
        sentence_scores = {}
        
        if unique_list:
            clf = _get_sentiment_clf_roberta()
            batch_size = 128
            
            from preprocessing import clean_text_bert
            bert_inputs = [clean_text_bert(s) for s in unique_list]
            
            def input_generator():
                for t in bert_inputs:
                    yield t if (isinstance(t, str) and t.strip() != "") else " "
            
            print(f"[STANCE] Scoring {len(unique_list)} unique sentences in parallel batches of size {batch_size}...")
            raw_results = clf(
                input_generator(),
                batch_size=batch_size,
                truncation=True,
                max_length=512
            )
            
            for s, res in zip(unique_list, raw_results):
                scores_dict = {d['label']: d['score'] for d in res}
                if 'LABEL_0' in scores_dict:
                    p_neg = scores_dict.get('LABEL_0', 0.0)
                    p_pos = scores_dict.get('LABEL_2', 0.0)
                else:
                    p_neg = scores_dict.get('negative', 0.0)
                    p_pos = scores_dict.get('positive', 0.0)
                sentence_scores[s] = p_pos - p_neg
                
        # Split dual mentions using the precomputed sentence scores
        sinner_scores = []
        alcaraz_scores = []
        for idx in dual_indices:
            s_score, a_score = _split_dual_mention(df.loc[idx, 'text'], sentence_scores)
            sinner_scores.append(s_score)
            alcaraz_scores.append(a_score)
            
        df.loc[both_players, 'stance_sinner'] = sinner_scores
        df.loc[both_players, 'stance_alcaraz'] = alcaraz_scores
        
    if DEBUG:
        print(f"[STANCE] Post-level stance scores computed.")
        print(f"  Single Sinner:  {single_sinner.sum():>6} posts")
        print(f"  Single Alcaraz: {single_alcaraz.sum():>6} posts")
        print(f"  Both players:   {both_players.sum():>6} posts (sentence-split)")
        print(f"  Neither:        {neither.sum():>6} posts")
        print(f"  Stance Sinner range:  {df['stance_sinner'].min():.3f} to {df['stance_sinner'].max():.3f}")
        print(f"  Stance Alcaraz range: {df['stance_alcaraz'].min():.3f} to {df['stance_alcaraz'].max():.3f}")
        
    return df


# ═════════════════════════════════════════════════════════════════════
# USER-LEVEL NET STANCE
# ═════════════════════════════════════════════════════════════════════

def compute_user_stances(df: pd.DataFrame, min_posts: int = MIN_POSTS_FOR_STANCE) -> pd.DataFrame:
    """
    Aggregate per author: mean stance scores, frequencies, and net stance.
    """
    grouped = df.groupby('author_handle')
    user_stances = grouped.agg(
        mean_sinner=('stance_sinner', 'mean'),
        mean_alcaraz=('stance_alcaraz', 'mean'),
        n_sinner=('is_sinner', 'sum'),
        n_alcaraz=('is_alcaraz', 'sum'),
        post_count=('stance_sinner', 'count')
    )
    
    total_users_before = len(user_stances)
    user_stances = user_stances[user_stances['post_count'] >= min_posts]
    if DEBUG:
        print(f"\n[STANCE] User filtering: {total_users_before} -> {len(user_stances)} users (min {min_posts} posts)")
    
    user_stances[['mean_sinner', 'mean_alcaraz']] = user_stances[['mean_sinner', 'mean_alcaraz']].fillna(0.0)
    
    total_mentions = user_stances['n_sinner'] + user_stances['n_alcaraz']
    user_stances['freq_sinner'] = (user_stances['n_sinner'] / total_mentions.replace(0, np.nan)).fillna(0)
    user_stances['freq_alcaraz'] = (user_stances['n_alcaraz'] / total_mentions.replace(0, np.nan)).fillna(0)
    
    user_stances['net_stance'] = (
        W_SENTIMENT * (user_stances['mean_sinner'] - user_stances['mean_alcaraz'])
        + W_FREQUENCY * (user_stances['freq_sinner'] - user_stances['freq_alcaraz'])
    )
    
    if DEBUG:
        print(f"[STANCE] Net stance range: {user_stances['net_stance'].min():.3f} to {user_stances['net_stance'].max():.3f}")
        print(f"[STANCE] Net stance mean:  {user_stances['net_stance'].mean():.3f}")
        print(f"[STANCE] Net stance std:   {user_stances['net_stance'].std():.3f}")
    
    return user_stances


# ═════════════════════════════════════════════════════════════════════
# STANCE CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════

def classify_stances(user_stances: pd.DataFrame, threshold: float = STANCE_THRESHOLD) -> pd.DataFrame:
    """Assign stance_leaning based on net_stance threshold."""
    def label(score):
        if score > threshold:
            return 'sinner'
        elif score < -threshold:
            return 'alcaraz'
        return 'neutral'
    
    user_stances['stance_leaning'] = user_stances['net_stance'].apply(label)
    
    if DEBUG:
        counts = user_stances['stance_leaning'].value_counts()
        total = len(user_stances)
        for stance in ['sinner', 'alcaraz', 'neutral']:
            n = counts.get(stance, 0)
            print(f"[STANCE] {stance.capitalize():>8}: {n:>5} ({n/total*100:.1f}%)")
    
    return user_stances


# ═════════════════════════════════════════════════════════════════════
# COMMUNITY PROFILES
# ═════════════════════════════════════════════════════════════════════

def community_stance_profiles(
    G: nx.Graph,
    user_stances: pd.DataFrame,
    partition: Dict
) -> pd.DataFrame:
    """
    For each community: size, mean stance, dominant leaning, homogeneity.
    """
    global_mean = user_stances['net_stance'].mean()
    communities = set(partition.values())
    records = []
    
    for cid in communities:
        members = [n for n, c in partition.items() if c == cid]
        member_data = user_stances.reindex(members)
        scores = member_data['net_stance'].dropna()
        leanings = member_data['stance_leaning'].dropna()
        
        mean_score = scores.mean() if len(scores) > 0 else 0.0
        
        if len(leanings) == 0:
            dominant = 'neutral'
            homogeneity = 0.0
        else:
            dominant = leanings.mode().iloc[0] if not leanings.mode().empty else 'neutral'
            homogeneity = (leanings == dominant).mean()
        
        records.append({
            'community_id': int(cid),
            'size': len(members),
            'mean_stance': round(mean_score, 4),
            'deviation_from_global': round(mean_score - global_mean, 4),
            'dominant_lean': dominant,
            'homogeneity': round(homogeneity, 3)
        })
    
    return pd.DataFrame(records).sort_values('size', ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# POLARIZATION METRICS
# ═════════════════════════════════════════════════════════════════════

def polarization_metrics(G: nx.Graph, user_stances: pd.DataFrame) -> Dict:
    """Compute assortativity, cross-stance edge ratio, and score std."""
    metrics = {}
    
    leaning_map = user_stances['stance_leaning'].to_dict()
    nx.set_node_attributes(G, leaning_map, 'stance_leaning')
    
    try:
        metrics['stance_assortativity'] = round(
            nx.attribute_assortativity_coefficient(G, 'stance_leaning'), 4
        )
    except Exception:
        metrics['stance_assortativity'] = 0.0
    
    cross = 0
    total = G.number_of_edges()
    for u, v in G.edges():
        l_u = leaning_map.get(u, 'neutral')
        l_v = leaning_map.get(v, 'neutral')
        if l_u != l_v and 'neutral' not in (l_u, l_v):
            cross += 1
    metrics['cross_stance_ratio'] = round(cross / total, 4) if total else 0.0
    
    scores = user_stances['net_stance'].dropna()
    metrics['score_std'] = round(float(scores.std()), 4) if len(scores) > 1 else 0.0
    
    return metrics


# ═════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════

def plot_stance_results(
    user_stances: pd.DataFrame,
    community_profiles: pd.DataFrame,
    polar_metrics: Dict,
    node_to_community: Dict,
    output_dir: str = "plots"
) -> None:
    """Create a 4-panel figure summarizing stance results."""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Stance Analysis Summary", fontsize=18, weight='bold', y=0.97)
    
    # 1. User Stance Distribution
    ax1 = fig.add_subplot(2, 2, 1)
    stance_counts = user_stances['stance_leaning'].value_counts()
    colors = {'sinner': '#1f77b4', 'alcaraz': '#d62728', 'neutral': '#7f7f7f'}
    bar_colors = [colors.get(label, '#7f7f7f') for label in stance_counts.index]
    ax1.bar(stance_counts.index, stance_counts.values, color=bar_colors, edgecolor='black')
    ax1.set_title("User Stance Distribution", fontsize=13, weight='bold')
    ax1.set_ylabel("Number of users")
    for i, v in enumerate(stance_counts.values):
        ax1.text(i, v + max(stance_counts.values) * 0.02, str(v), ha='center', fontweight='bold')
    
    # 2. Net Stance Score Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    scores = user_stances['net_stance'].dropna()
    ax2.hist(scores, bins=50, color='darkorange', edgecolor='black', alpha=0.8)
    ax2.axvline(STANCE_THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Sinner threshold (+{STANCE_THRESHOLD})')
    ax2.axvline(-STANCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Alcaraz threshold (-{STANCE_THRESHOLD})')
    ax2.axvline(0, color='gray', linestyle='-', linewidth=1)
    ax2.set_title("Distribution of Net Stance Scores", fontsize=13, weight='bold')
    ax2.set_xlabel("Net Stance Score")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=9)
    
    # 3. Community Stance Profiles (top 10)
    ax3 = fig.add_subplot(2, 2, 3)
    top_comms = community_profiles.head(10).copy()
    cmap = get_community_color_map(node_to_community)
    top_comms = top_comms.sort_values('mean_stance')
    top_comms['label'] = top_comms.apply(
        lambda r: f"C{r['community_id']} ({r['dominant_lean'][:1].upper()}, n={r['size']})", axis=1
    )
    bars = ax3.barh(top_comms['label'], top_comms['mean_stance'], 
                    color=[cmap.get(cid, 'gray') for cid in top_comms['community_id']],
                    edgecolor='black')
    ax3.axvline(0, color='gray', linewidth=1)
    ax3.set_title("Top Communities - Mean Stance Score", fontsize=13, weight='bold')
    ax3.set_xlabel("Mean Stance (positive = pro-Sinner)")
    
    # 4. Polarization Metrics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    text = f"""
    Polarization Metrics
    ------------------------------
    Stance Assortativity:     {polar_metrics.get('stance_assortativity', 'N/A')}
    Cross-Stance Edge Ratio:  {polar_metrics.get('cross_stance_ratio', 'N/A')}
    Score Standard Deviation: {polar_metrics.get('score_std', 'N/A')}
    
    Total Users Analyzed:     {len(user_stances)}
    Threshold:                +/-{STANCE_THRESHOLD}
    Min Posts per User:       {MIN_POSTS_FOR_STANCE}
    """
    ax4.text(0.1, 0.5, text, fontsize=13, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_plot_copies("stance_analysis_summary.png", output_dir=output_dir)
    plt.close()
    print(f"[STANCE] Plot saved to {output_dir}/stance_analysis_summary.png")


# ═════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════

def sample_dual_mention_posts(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Return a random sample of dual-mention posts with split scores."""
    both = df[df['is_sinner'] & df['is_alcaraz']].copy()
    if len(both) == 0:
        return pd.DataFrame()
    sample = both.sample(min(n, len(both)), random_state=42)
    
    # Pre-score sentences for these sample posts using RoBERTa
    sentence_scores = {}
    for text in sample['text']:
        sents = sent_tokenize(str(text))
        for s in sents:
            s_clean = s.strip()
            if s_clean and s_clean not in sentence_scores:
                sentence_scores[s_clean] = _sentence_sentiment(s_clean)
                
    splits = sample['text'].apply(lambda t: _split_dual_mention(t, sentence_scores))
    sample['split_sinner'] = [s[0] for s in splits]
    sample['split_alcaraz'] = [s[1] for s in splits]
    sample['split_diff'] = abs(sample['split_sinner'] - sample['split_alcaraz'])
    
    sample = sample.sort_values('split_diff', ascending=False)
    return sample[['text', 'sentiment_compound', 'split_sinner', 'split_alcaraz', 'split_diff']]


def print_post_samples(df: pd.DataFrame):
    """Print example dual-mention posts with their split scores."""
    sample = sample_dual_mention_posts(df, n=30)
    if sample.empty:
        print("\n[DIAGNOSTIC] No dual-mention posts to sample.")
        return
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1: Sample Dual-Mention Posts (sorted by sentiment difference)")
    print("=" * 80)
    
    for i, (_, row) in enumerate(sample.head(15).iterrows()):
        text = str(row['text'])[:120].replace('\n', ' ')
        print(f"\n--- Post {i+1} (diff={row['split_diff']:.2f}) ---")
        print(f"Text:       {text}...")
        print(f"Compound:   {row['sentiment_compound']:.3f}")
        print(f"Sinner:     {row['split_sinner']:.3f}")
        print(f"Alcaraz:    {row['split_alcaraz']:.3f}")
    
    print(f"\n--- Summary of {len(sample)} sampled posts ---")
    print(f"Mean absolute difference: {sample['split_diff'].mean():.3f}")
    print(f"Median absolute difference: {sample['split_diff'].median():.3f}")
    print(f"Posts with diff > 0.3: {(sample['split_diff'] > 0.3).sum()}/{len(sample)}")
    print(f"Posts with diff > 0.5: {(sample['split_diff'] > 0.5).sum()}/{len(sample)}")
    print("=" * 80)


def sentiment_distribution_analysis(df: pd.DataFrame):
    """Show sentiment distribution across different mention types."""
    compound = df['sentiment_compound'].fillna(0)
    
    types = {
        'Sinner only': df['is_sinner'] & ~df['is_alcaraz'],
        'Alcaraz only': df['is_alcaraz'] & ~df['is_sinner'],
        'Both players': df['is_sinner'] & df['is_alcaraz'],
        'Neither': ~df['is_sinner'] & ~df['is_alcaraz']
    }
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: Sentiment Distribution by Mention Type")
    print("=" * 80)
    print(f"{'Type':<20} {'Total':>8} {'Strong +':>10} {'Strong -':>10} {'Neutral':>10} {'% Opinionated':>14}")
    print("-" * 72)
    
    for label, mask in types.items():
        subset = compound[mask]
        total = len(subset)
        if total == 0:
            continue
        pos = (subset > 0.3).sum()
        neg = (subset < -0.3).sum()
        neut = ((subset >= -0.1) & (subset <= 0.1)).sum()
        opinionated = ((subset > 0.3) | (subset < -0.3)).sum()
        print(f"{label:<20} {total:>8} {pos:>10} {neg:>10} {neut:>10} {opinionated/total*100:>13.1f}%")
    
    print("=" * 80)
    print("'Opinionated' = |compound| > 0.3")
    print("'Neutral' = |compound| <= 0.1")
    print("=" * 80)


def threshold_sensitivity_analysis(user_stances: pd.DataFrame):
    """Show classification across different thresholds."""
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    results = []
    
    for t in thresholds:
        def label(score):
            if score > t: return 'sinner'
            elif score < -t: return 'alcaraz'
            return 'neutral'
        
        leanings = user_stances['net_stance'].apply(label)
        counts = leanings.value_counts()
        results.append({
            'threshold': t,
            'sinner': counts.get('sinner', 0),
            'alcaraz': counts.get('alcaraz', 0),
            'neutral': counts.get('neutral', 0),
            'sinner_pct': counts.get('sinner', 0) / len(leanings) * 100,
            'alcaraz_pct': counts.get('alcaraz', 0) / len(leanings) * 100,
            'neutral_pct': counts.get('neutral', 0) / len(leanings) * 100,
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: Threshold Sensitivity Analysis")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)
    print(f"As threshold increases from {thresholds[0]} to {thresholds[-1]}:")
    print(f"  Neutral % goes from {results_df['neutral_pct'].iloc[0]:.1f}% to {results_df['neutral_pct'].iloc[-1]:.1f}%")
    neutral_change = results_df['neutral_pct'].iloc[-1] - results_df['neutral_pct'].iloc[0]
    print(f"  This means the data is {'highly' if neutral_change > 30 else 'not'} sensitive to threshold choice.")
    print("=" * 80)
    
    return results_df


def synthetic_polarization_test():
    """Run the pipeline on synthetic data with known polarization."""
    np.random.seed(42)
    
    users = []
    for i in range(50):
        for _ in range(np.random.randint(3, 8)):
            users.append({
                'author_handle': f'sinner_fan_{i}',
                'text': 'Sinner is incredible, best player ever',
                'sentiment_compound': np.random.uniform(0.5, 0.95),
                'is_sinner': True,
                'is_alcaraz': False,
                'stance_sinner': 0.0,
                'stance_alcaraz': 0.0,
            })
            users.append({
                'author_handle': f'sinner_fan_{i}',
                'text': 'Alcaraz is overrated',
                'sentiment_compound': np.random.uniform(-0.95, -0.5),
                'is_sinner': False,
                'is_alcaraz': True,
                'stance_sinner': 0.0,
                'stance_alcaraz': 0.0,
            })
    
    for i in range(50):
        for _ in range(np.random.randint(3, 8)):
            users.append({
                'author_handle': f'alcaraz_fan_{i}',
                'text': 'Sinner cannot compare to Alcaraz',
                'sentiment_compound': np.random.uniform(-0.95, -0.5),
                'is_sinner': True,
                'is_alcaraz': False,
                'stance_sinner': 0.0,
                'stance_alcaraz': 0.0,
            })
            users.append({
                'author_handle': f'alcaraz_fan_{i}',
                'text': 'Alcaraz amazing performance',
                'sentiment_compound': np.random.uniform(0.5, 0.95),
                'is_sinner': False,
                'is_alcaraz': True,
                'stance_sinner': 0.0,
                'stance_alcaraz': 0.0,
            })
    
    df_synth = pd.DataFrame(users)
    
    # Assign stance scores (simplified for synthetic test)
    df_synth.loc[df_synth['is_sinner'] & ~df_synth['is_alcaraz'], 'stance_sinner'] = df_synth['sentiment_compound']
    df_synth.loc[df_synth['is_alcaraz'] & ~df_synth['is_sinner'], 'stance_alcaraz'] = df_synth['sentiment_compound']
    
    user_synth = compute_user_stances(df_synth, min_posts=1)
    user_synth = classify_stances(user_synth, threshold=0.05)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: Synthetic Polarization Test")
    print("=" * 80)
    print(f"Synthetic users (50 pro-Sinner, 50 pro-Alcaraz, strong opinions):")
    print(f"  Net stance range: {user_synth['net_stance'].min():.3f} to {user_synth['net_stance'].max():.3f}")
    print(f"  Net stance std:   {user_synth['net_stance'].std():.3f}")
    
    counts = user_synth['stance_leaning'].value_counts()
    for stance in ['sinner', 'alcaraz', 'neutral']:
        n = counts.get(stance, 0)
        print(f"  {stance}: {n} ({n/len(user_synth)*100:.0f}%)")
    
    neutrals = counts.get('neutral', 0)
    print(f"\n  Expected: ~50 sinner, ~50 alcaraz, ~0 neutral")
    print(f"  Pipeline detection: {'PASS' if neutrals < 10 else 'FAIL'} - {neutrals} neutrals detected")
    print("  (If this shows mostly neutral, the PIPELINE is broken)")
    print("  (If this shows mostly partisan, the DATA is the issue)")
    print("=" * 80)
    
    return user_synth


def post_quality_analysis(df: pd.DataFrame):
    """Check if posts are suitable for sentiment analysis."""
    df = df.copy()
    df['post_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['non_ascii_ratio'] = df['text'].apply(
        lambda t: sum(1 for c in str(t) if ord(c) > 127) / max(len(str(t)), 1)
    )
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 5: Post Quality Analysis")
    print("=" * 80)
    print(f"Total posts:         {len(df)}")
    print(f"Median post length:  {df['post_length'].median():.0f} characters")
    print(f"Median word count:   {df['word_count'].median():.0f} words")
    print(f"Posts with < 5 words:    {(df['word_count'] < 5).sum()} ({(df['word_count'] < 5).sum()/len(df)*100:.1f}%)")
    print(f"Posts with < 10 words:   {(df['word_count'] < 10).sum()} ({(df['word_count'] < 10).sum()/len(df)*100:.1f}%)")
    print(f"Posts with > 10% non-ASCII: {(df['non_ascii_ratio'] > 0.1).sum()} ({(df['non_ascii_ratio'] > 0.1).sum()/len(df)*100:.1f}%)")
    
    compound = df['sentiment_compound'].fillna(0)
    print(f"\nSentiment compound distribution:")
    for label, low, high in [('Strong negative', -1.0, -0.3), ('Mild negative', -0.3, -0.1),
                              ('Neutral', -0.1, 0.1), ('Mild positive', 0.1, 0.3),
                              ('Strong positive', 0.3, 1.0)]:
        count = ((compound >= low) & (compound < high)).sum()
        bar = '#' * int(count / len(df) * 50)
        print(f"  {label:<20}: {count:>6} ({count/len(df)*100:5.1f}%) {bar}")
    print("=" * 80)


def run_all_diagnostics(df: pd.DataFrame, user_stances: pd.DataFrame):
    """Run all diagnostic tests and print results."""
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "   COMPREHENSIVE DATA QUALITY DIAGNOSTICS".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print_post_samples(df)
    sentiment_distribution_analysis(df)
    threshold_sensitivity_analysis(user_stances)
    synthetic_polarization_test()
    post_quality_analysis(df)
    
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "   DIAGNOSTICS COMPLETE".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)


# ═════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════

def run_stance_propagation(
    df: pd.DataFrame,
    Gu: nx.Graph,
    Gd: nx.Graph,
    df_cent: pd.DataFrame,
    threshold: float = STANCE_THRESHOLD,
    min_posts: int = MIN_POSTS_FOR_STANCE,
    output_dir: str = "plots",
    run_diagnostics: bool = True
) -> Dict:
    """
    Main stance pipeline: post -> user -> graph -> community -> metrics -> plots.
    
    Parameters:
        df: Processed posts DataFrame with columns:
            linked_entities, sentiment_compound, text, author_handle
        Gu: Undirected NetworkX graph
        Gd: Directed NetworkX graph
        df_cent: Centrality DataFrame with 'user' column
        threshold: Net stance threshold for partisan classification
        min_posts: Minimum posts required for a user to be analyzed
        output_dir: Directory to save plots
        run_diagnostics: Whether to print comprehensive diagnostics
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 60)
    print("STANCE PROPAGATION - STARTING")
    print("=" * 60)
    
    # 1. Player detection
    df = detect_players(df)
    
    # 2. Post-level stance scores
    df = compute_post_stances(df)
    
    # 3. User-level net stance
    user_stances = compute_user_stances(df, min_posts=min_posts)
    user_stances = classify_stances(user_stances, threshold=threshold)
    
    # 4. Map onto graph nodes
    f_prop = user_stances['net_stance'].to_dict()
    stance_leanings = user_stances['stance_leaning'].to_dict()
    
    nx.set_node_attributes(Gu, f_prop, 'stance_score')
    nx.set_node_attributes(Gd, f_prop, 'stance_score')
    nx.set_node_attributes(Gu, stance_leanings, 'stance_leaning')
    nx.set_node_attributes(Gd, stance_leanings, 'stance_leaning')
    
    # 5. Update centrality DataFrame
    df_cent['stance_score'] = df_cent['user'].map(f_prop)
    df_cent['stance_leaning'] = df_cent['user'].map(stance_leanings)
    
    # 6. Community profiles
    partition = nx.get_node_attributes(Gu, 'community')
    if not partition:
        print("[STANCE] Warning: No community partition found. Assigning all nodes to community 0.")
        partition = {n: 0 for n in Gu.nodes()}
    community_profiles = community_stance_profiles(Gu, user_stances, partition)
    
    # 7. Polarization metrics
    polar_metrics = polarization_metrics(Gu, user_stances)
    
    # 8. Summary
    total = len(user_stances)
    counts = user_stances['stance_leaning'].value_counts()
    sinner_n = counts.get('sinner', 0)
    alcaraz_n = counts.get('alcaraz', 0)
    neutral_n = counts.get('neutral', 0)
    
    print(f"\n[STANCE] Classification Complete")
    print(f"  Total users analyzed: {total}")
    print(f"  Pro-Sinner:  {sinner_n:>5} ({sinner_n/total*100:.1f}%)")
    print(f"  Pro-Alcaraz: {alcaraz_n:>5} ({alcaraz_n/total*100:.1f}%)")
    print(f"  Neutral:     {neutral_n:>5} ({neutral_n/total*100:.1f}%)")
    print(f"\n[STANCE] Polarization Metrics:")
    for k, v in polar_metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[STANCE] Top 5 Communities by Size:")
    print(community_profiles.head(5).to_string(index=False))
    
    # 9. Plot
    try:
        plot_stance_results(user_stances, community_profiles, polar_metrics, 
                           partition, output_dir=output_dir)
    except Exception as e:
        print(f"[STANCE] Plotting failed: {e}")
    
    # 10. Run Fanbase Study
    try:
        run_fanbase_study(user_stances, output_dir=output_dir)
    except Exception as e:
        print(f"[STANCE] Fanbase study failed: {e}")

    # 11. Diagnostics
    if run_diagnostics:
        run_all_diagnostics(df, user_stances)
    
    print("\n" + "=" * 60)
    print("STANCE PROPAGATION - COMPLETE")
    print("=" * 60 + "\n")
    
    return {
        'df_cent': df_cent,
        'user_stances': user_stances,
        'community_profiles': community_profiles,
        'polarization': polar_metrics,
        'threshold': threshold,
        'min_posts': min_posts
    }


def run_fanbase_study(user_stances: pd.DataFrame, output_dir: str = "plots") -> pd.DataFrame:
    """
    Perform a comparative study of classification methods:
    1. Baseline: Pure frequency-based fanbase (Sinner if freq_sinner > freq_alcaraz, etc.)
    2. Refined: Sentiment-aware net stance fanbase (stance_leaning)
    
    Generates statistics and comparison plots.
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE STUDY: BASELINE FREQUENCY vs SENTIMENT-AWARE FANBASE")
    print("=" * 80)
    
    # Compute baseline frequency leaning
    def get_baseline_leaning(row):
        f_s = row.get('freq_sinner', 0)
        f_a = row.get('freq_alcaraz', 0)
        if f_s > f_a:
            return 'sinner'
        elif f_a > f_s:
            return 'alcaraz'
        return 'neutral'
        
    user_stances = user_stances.copy()
    user_stances['baseline_leaning'] = user_stances.apply(get_baseline_leaning, axis=1)
    
    # Compute counts
    baseline_counts = user_stances['baseline_leaning'].value_counts()
    refined_counts = user_stances['stance_leaning'].value_counts()
    
    total_users = len(user_stances)
    print(f"Total Users: {total_users}")
    print("\nBaseline Fanbase (Frequency-Only):")
    for stance in ['sinner', 'alcaraz', 'neutral']:
        count = baseline_counts.get(stance, 0)
        print(f"  {stance.capitalize():<8}: {count:>5} ({count/total_users*100:.1f}%)")
        
    print("\nRefined Fanbase (Sentiment + Frequency):")
    for stance in ['sinner', 'alcaraz', 'neutral']:
        count = refined_counts.get(stance, 0)
        print(f"  {stance.capitalize():<8}: {count:>5} ({count/total_users*100:.1f}%)")
        
    # Contingency Table (Overlap)
    overlap = pd.crosstab(
        user_stances['baseline_leaning'], 
        user_stances['stance_leaning'],
        rownames=['Baseline (Freq)'],
        colnames=['Refined (Net Stance)']
    )
    print("\nTransition / Overlap Matrix:")
    print(overlap)
    
    # Transition explanations
    switched_s_to_a = user_stances[(user_stances['baseline_leaning'] == 'sinner') & (user_stances['stance_leaning'] == 'alcaraz')]
    switched_a_to_s = user_stances[(user_stances['baseline_leaning'] == 'alcaraz') & (user_stances['stance_leaning'] == 'sinner')]
    s_to_n = user_stances[(user_stances['baseline_leaning'] == 'sinner') & (user_stances['stance_leaning'] == 'neutral')]
    a_to_n = user_stances[(user_stances['baseline_leaning'] == 'alcaraz') & (user_stances['stance_leaning'] == 'neutral')]
    
    print("\nKey Insights:")
    print(f"  * Users matching Sinner by mentions but classified Alcaraz by sentiment: {len(switched_s_to_a)}")
    print(f"  * Users matching Alcaraz by mentions but classified Sinner by sentiment: {len(switched_a_to_s)}")
    print(f"  * Partisans moved to Neutral when sentiment filter applied: Sinner -> {len(s_to_n)}, Alcaraz -> {len(a_to_n)}")
    
    # Plotting comparison
    try:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Sinner Fans', 'Alcaraz Fans', 'Neutral']
        baseline_vals = [baseline_counts.get('sinner', 0), baseline_counts.get('alcaraz', 0), baseline_counts.get('neutral', 0)]
        refined_vals = [refined_counts.get('sinner', 0), refined_counts.get('alcaraz', 0), refined_counts.get('neutral', 0)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Sinner: Orange (#d35400 / #f39c12), Alcaraz: Blue (#023e8a / #00b4d8), Neutral: Gray
        rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Frequency-Only)', color=['#f39c12', '#00b4d8', '#cbd5e0'], edgecolor='black', alpha=0.6)
        rects2 = ax.bar(x + width/2, refined_vals, width, label='Refined (RoBERTa Sentiment-Aware)', color=['#d35400', '#023e8a', '#7f7f7f'], edgecolor='black')
        
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title('Fanbase Classification: Baseline Frequency vs. Sentiment-Aware Leaning', fontsize=14, weight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=10)
        
        # Label values on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        save_plot_copies("fanbase_classification_comparison.png", output_dir=output_dir)
        plt.close()
        print(f"[STANCE] Fanbase comparison plot saved to {output_dir}/fanbase_classification_comparison.png")
    except Exception as e:
        print(f"[STANCE] Fanbase comparison plotting failed: {e}")
        
    print("=" * 80 + "\n")
    return user_stances
# Add to stance_analysis.py and call from main.py

def print_community_196_posts(df: pd.DataFrame, Gu: nx.Graph, community_id: int = 196):
    """Print all post texts from members of a specific community."""
    
    partition = nx.get_node_attributes(Gu, 'community')
    if not partition:
        print("No community partition found.")
        return
    
    members = [node for node, comm in partition.items() if comm == community_id]
    
    if not members:
        print(f"Community {community_id} not found.")
        return
    
    comm_posts = df[df['author_handle'].isin(members)].copy()
    comm_posts = comm_posts.sort_values('sentiment_compound')
    
    print(f"\nCommunity {community_id} — {len(members)} members, {len(comm_posts)} posts")
    print("=" * 80)
    
    for i, (_, row) in enumerate(comm_posts.iterrows(), 1):
        text = str(row['text']).replace('\n', ' ').replace('\r', '')
        sentiment = row.get('sentiment_compound', 0)
        author = row.get('author_handle', 'unknown')
        print(f"\n[{i}/{len(comm_posts)}] [{sentiment:+.3f}] {author}")
        print(text)
        print("-" * 80)