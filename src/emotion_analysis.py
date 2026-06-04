"""
Emotion Analysis Module — NRC Emotion Lexicon
Adds emotion profiling to supplement VADER sentiment analysis.
Required by WSA project instructions: "emotion analysis" is a mandatory
content analysis component alongside sentiment analysis.
"""

from nrclex import NRCLex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# The 8 primary NRC emotion categories
NRC_EMOTIONS = [
    'fear', 'anger', 'anticipation', 'trust',
    'surprise', 'sadness', 'disgust', 'joy'
]


def score_emotions(text: str) -> dict:
    """
    Score a single text string across 8 NRC emotion categories.
    Returns a dict with emotion name → normalised frequency score.
    Returns all-zero dict for empty/NaN input.
    Uses nrclex v4 API: NRCLex() + load_raw_text(text) + affect_frequencies.
    """
    if not text or pd.isna(text) or str(text).strip() == "":
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
    8 new columns: emotion_fear, emotion_anger, ..., emotion_joy.
    Also adds 'dominant_emotion' column (the highest-scoring emotion per post).
    """
    print(f"Running NRC Emotion Analysis on {len(df)} posts...")

    emotion_scores = df[text_col].apply(score_emotions)
    emotion_df = pd.DataFrame(emotion_scores.tolist(), index=df.index)
    emotion_df.columns = [f'emotion_{e}' for e in NRC_EMOTIONS]

    df = pd.concat([df, emotion_df], axis=1)

    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')

    # Mark posts where all emotion scores are 0 as 'neutral'
    all_zero_mask = df[emotion_cols].sum(axis=1) == 0
    df.loc[all_zero_mask, 'dominant_emotion'] = 'neutral'

    print(f"Emotion analysis complete. Dominant emotion distribution:")
    print(df['dominant_emotion'].value_counts().to_string())

    return df


def plot_emotion_distribution(df: pd.DataFrame, output_dir: str,
                               title_suffix: str = "") -> None:
    """
    Plot 1: Bar chart of overall emotion frequencies across the corpus.
    Plot 2: Stacked bar chart comparing emotion profiles for
            Sinner-related vs Alcaraz-related posts.
    """
    os.makedirs(output_dir, exist_ok=True)
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
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.savefig(os.path.join("report", fname), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(output_dir, fname)}")

    # --- Plot 2: Sinner vs Alcaraz emotion comparison ---
    sinner_mask = df['text'].str.lower().str.contains('sinner|jannik', na=False)
    alcaraz_mask = df['text'].str.lower().str.contains('alcaraz|carlitos', na=False)

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
    plt.savefig(os.path.join(output_dir, fname2), dpi=300)
    plt.savefig(os.path.join("report", fname2), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(output_dir, fname2)}")


def print_emotion_summary(df: pd.DataFrame) -> None:
    """Print a structured summary of the emotion analysis results."""
    emotion_cols = [f'emotion_{e}' for e in NRC_EMOTIONS]

    print("\n--- NRC Emotion Analysis Summary ---")
    print(f"{'Emotion':<15} {'Mean Score':>12} {'% Posts w/ signal':>18}")
    print("-" * 47)
    for col in emotion_cols:
        emotion = col.replace('emotion_', '')
        mean_score = df[col].mean()
        pct_nonzero = (df[col] > 0).mean() * 100
        print(f"{emotion:<15} {mean_score:>12.4f} {pct_nonzero:>17.1f}%")

    print(f"\nDominant Emotion Distribution:")
    print(df['dominant_emotion'].value_counts().to_string())
