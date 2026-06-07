
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import render_flat_chart, NRC_EMOTIONS
from social_network_analysis import get_community_color_map


def _prepare_community_emotion_data(
    df: pd.DataFrame,
    Gu,
    node_to_community: dict,
    top_k: int,
    sort_by: str,
    backend: str,
    cmap_name: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    author_community = {}
    for handle in df['author_handle']:
        if handle in node_to_community and handle in Gu.nodes():
            author_community[handle] = node_to_community[handle]

    df_copy = df.copy()
    df_copy['community_id'] = df_copy['author_handle'].map(author_community)
    df_with_comm = df_copy.dropna(subset=['community_id']).copy()

    if sort_by == "post_volume":
        top_comms = df_with_comm['community_id'].value_counts().head(top_k).index.tolist()
    else:
        from collections import Counter
        filtered = {n: c for n, c in node_to_community.items() if n in Gu.nodes()}
        top_comms = [cid for cid, _ in Counter(filtered.values()).most_common(top_k)]

    col_prefix = 'bert_emotion_' if backend == "bert" else 'nrc_emotion_'

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
    return df_plot, custom_palette


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

    df_plot, custom_palette = _prepare_community_emotion_data(
        df, Gu, node_to_community, top_k, sort_by, backend, cmap_name
    )
    if df_plot.empty:
        return

    subtitle = "GoEmotions (BERT)" if backend == "bert" else "NRC Emotion Lexicon"
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Score", hue="Community", palette=custom_palette)

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
    render_flat_chart(
        fig=fig,
        filename=filename,
        output_dir=output_dir,
        title=f"Average Emotion Profiles per Community{title_suffix}\n({subtitle})",
        xlabel="Emotion Category",
        ylabel="Mean Normalized Score",
        legend_title="Community",
    )


def _prepare_emotion_backend_comparison_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if 'nrc_dominant_emotion' not in df.columns or 'bert_dominant_emotion' not in df.columns:
        return None

    emotion_index = NRC_EMOTIONS + ["neutral"]
    nrc_dist = df['nrc_dominant_emotion'].value_counts(normalize=True).reindex(emotion_index, fill_value=0.0)
    bert_dist = df['bert_dominant_emotion'].value_counts(normalize=True).reindex(emotion_index, fill_value=0.0)

    records = []
    for emotion in emotion_index:
        records.append({"Emotion": emotion, "Share of posts": float(nrc_dist[emotion]), "Backend": "NRC"})
        records.append({"Emotion": emotion, "Share of posts": float(bert_dist[emotion]), "Backend": "BERT"})
    return pd.DataFrame(records)


def plot_emotion_backend_comparison(df: pd.DataFrame, output_dir: str = "plots") -> None:
    df_plot = _prepare_emotion_backend_comparison_data(df)
    if df_plot is None:
        print("Warning: 'nrc_dominant_emotion'/'bert_dominant_emotion' columns not found. Skipping plot.")
        return

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Emotion", y="Share of posts", hue="Backend")

    render_flat_chart(
        fig=fig,
        filename="emotion_backend_comparison.png",
        output_dir=output_dir,
        title="Dominant Emotion Distribution: NRC vs GoEmotions (BERT)",
        xlabel="Emotion Category",
        ylabel="Share of posts",
        legend_title="Backend",
    )


def _prepare_sentiment_distribution_data(df: pd.DataFrame) -> Optional[tuple[list, list]]:
    if 'sentiment_category' not in df.columns:
        return None
    counts = df['sentiment_category'].value_counts()
    return counts.index.tolist(), counts.values.tolist()


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: str = "plots") -> None:
    data = _prepare_sentiment_distribution_data(df)
    if data is None:
        print("Warning: 'sentiment_category' column not found in DataFrame. Skipping plot.")
        return

    categories, sizes = data
    # Color palette matching emerald green (pos), slate gray (neu), rose red (neg)
    colors_map = {
        'positive': '#2ec4b6',
        'neutral': '#a0aec0',
        'negative': '#e63946'
    }
    colors = [colors_map.get(cat, '#cbd5e0') for cat in categories]

    fig = plt.figure(figsize=(7, 7))
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
    fig.gca().add_artist(centre_circle)

    # Style percentage text to be readable
    for autotext in autotexts:
        autotext.set_color('white')

    render_flat_chart(
        fig=fig,
        filename="sentiment_distribution.png",
        output_dir=output_dir,
        title="Overall Sentiment Distribution\n(RoBERTa Sentiment Classifier)",
        title_weight="bold",
        title_pad=20,
    )


def plot_sentiment_over_time(df: pd.DataFrame, output_dir: str = "plots", user_stances: Optional[pd.DataFrame] = None) -> None:
    if 'created_at' not in df.columns or 'sentiment_compound' not in df.columns:
        print("Warning: Required columns for sentiment over time not found. Skipping plot.")
        return

    df = df.copy()
    df['date'] = pd.to_datetime(df['created_at'], format='mixed').dt.date

    sinner_records = []
    alcaraz_records = []

    # Determine whether we use codebase fanbase partition or keyword mentions
    use_fanbase = user_stances is not None and 'stance_leaning' in user_stances.columns

    if use_fanbase:
        user_leanings = user_stances['stance_leaning'].to_dict()
        print("[NLP] Plotting sentiment trajectory grouped by codebase fanbase partition...")
        for _, row in df.iterrows():
            author = row['author_handle']
            leaning = user_leanings.get(author, 'neutral')
            if leaning == 'sinner':
                sinner_records.append({"date": row['date'], "sentiment": row['sentiment_compound']})
            elif leaning == 'alcaraz':
                alcaraz_records.append({"date": row['date'], "sentiment": row['sentiment_compound']})
    else:
        print("[NLP] Plotting sentiment trajectory grouped by keyword mentions (fallback)...")
        for _, row in df.iterrows():
            if row.get('is_sinner', False):
                sinner_records.append({"date": row['date'], "sentiment": row['sentiment_compound']})
            if row.get('is_alcaraz', False):
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

    vol_label_s = 'Sinner Post Volume'
    vol_label_a = 'Alcaraz Post Volume'

    if not sinner_vol.empty:
        ax2.fill_between(sinner_vol.index, sinner_vol.values, alpha=0.12, color='#f39c12', label=vol_label_s)
    if not alcaraz_vol.empty:
        ax2.fill_between(alcaraz_vol.index, alcaraz_vol.values, alpha=0.12, color='#00b4d8', label=vol_label_a)

    ax2.set_ylabel("Daily Post Volume (Shaded)", color='gray', fontsize=11, labelpad=10)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(False)  # Disable grid lines for secondary axis to avoid clutter

    # 2. Plot Average Sentiment Compound Scores on the primary Y-axis (ax1)
    line_label_s = 'Sinner Sentiment'
    line_label_a = 'Alcaraz Sentiment'

    if not sinner_trend.empty:
        ax1.plot(sinner_trend.index, sinner_trend.values, marker='o', linewidth=2.5, color='#d35400', label=line_label_s)
    if not alcaraz_trend.empty:
        ax1.plot(alcaraz_trend.index, alcaraz_trend.values, marker='s', linewidth=2.5, color='#023e8a', label=line_label_a)

    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Import and annotate US Open 2025 key match events from utils
    from utils import US_OPEN_EVENTS

    ax1.set_ylim(-0.1, 0.5)
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

    title_text = ("Sentiment Trajectory & Post Volume Over Time (US Open 2025)\n"
                  "(Daily Avg RoBERTa Sentiment vs. Post Volume by Fanbase Leaning)" if use_fanbase else
                  "Sentiment Trajectory & Post Volume Over Time (US Open 2025)\n"
                  "(Daily Avg RoBERTa Sentiment vs. Post Volume)")

    ax1.set_xlabel("Date", fontsize=11, labelpad=10)
    ax1.set_ylabel("Average Sentiment Compound Score (Lines)", fontsize=11, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Place legend horizontally below the plot to avoid overlapping any lines or event labels
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=10)

    fig.autofmt_xdate()

    render_flat_chart(
        fig=fig,
        filename="sentiment_over_time.png",
        output_dir=output_dir,
        title=title_text,
        title_weight="bold",
        title_pad=15,
    )
