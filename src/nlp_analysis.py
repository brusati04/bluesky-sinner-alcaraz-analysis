import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from utils import clean_text, preprocess, get_vader_sentiment, extract_ner, link_entities_dbpedia
from emotion_analysis import add_emotion_columns, print_emotion_summary


def analyze_community_sentiment_polarization(df, G, node_to_community, communities,
                                              output_dir, title_suffix=""):
    """
    Join Louvain community IDs back to post-level data and compare
    sentiment profiles across communities.
    """
    author_community = {}
    for _, row in df.iterrows():
        handle = row.get('author_handle')
        if handle and handle in node_to_community:
            author_community[handle] = node_to_community[handle]

    df = df.copy()
    df['community_id'] = df['author_handle'].map(author_community)
    df_with_comm = df.dropna(subset=['community_id', 'sentiment_compound']).copy()
    df_with_comm['community_id'] = df_with_comm['community_id'].astype(int)

    comm_counts = df_with_comm['community_id'].value_counts()
    valid_comms = comm_counts[comm_counts >= 5].index.tolist()
    df_valid = df_with_comm[df_with_comm['community_id'].isin(valid_comms)]

    n_valid = len(valid_comms)
    print(f"\n--- Community Sentiment Polarization Analysis{title_suffix} ---")
    print(f"Communities with >=5 posts: {n_valid} (out of {len(communities)} total)")

    if n_valid < 2:
        print("Not enough communities with sufficient posts for polarization analysis.")
        return df

    comm_stats = df_valid.groupby('community_id')['sentiment_compound'].agg(
        ['mean', 'std', 'count', 'median']
    ).reset_index()
    comm_stats.columns = ['community_id', 'mean_sentiment', 'std_sentiment',
                           'post_count', 'median_sentiment']
    comm_stats = comm_stats.sort_values('post_count', ascending=False)

    print("\nTop communities by post count with sentiment profiles:")
    print(comm_stats.head(10).to_string(index=False))

    groups = [
        df_valid[df_valid['community_id'] == cid]['sentiment_compound'].values
        for cid in valid_comms
    ]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis H-test across {n_valid} communities:")
    print(f"  H = {kw_stat:.4f}, p = {kw_p:.4f}")
    if kw_p < 0.05:
        print("  -> Statistically significant sentiment differences across communities (p < 0.05).")
        print("  -> This provides evidence of attitudinal polarization between sub-communities.")
    else:
        print("  -> No statistically significant sentiment polarization found (p >= 0.05).")
        print("  -> Communities are structurally separate but NOT sentimentally polarized.")

    suffix_filename = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
    comm_stats.to_csv(
        f"data/community_sentiment_stats{suffix_filename}.csv",
        index=False
    )

    top_n = min(8, n_valid)
    top_comm_ids = comm_stats.head(top_n)['community_id'].tolist()
    df_plot = df_valid[df_valid['community_id'].isin(top_comm_ids)].copy()
    df_plot['community_label'] = 'Comm ' + df_plot['community_id'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    order = ['Comm ' + str(c) for c in top_comm_ids]
    sns.boxplot(data=df_plot, x='community_label', y='sentiment_compound',
                order=order, hue='community_label', palette='Set2', legend=False, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7, label='Neutral baseline')
    ax.set_title(
        f"Sentiment Distribution per Community (Top {top_n} by size){title_suffix}\n"
        f"Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.4f}"
    )
    ax.set_xlabel("Community (Louvain partition)")
    ax.set_ylabel("VADER Compound Sentiment Score")
    ax.legend()
    plt.tight_layout()
    fname = f"community_sentiment_polarization{suffix_filename}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.savefig(os.path.join("report", fname), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(output_dir, fname)}")

    return df


def run_nlp_enrichment(df, output_filepath="data/sinner_alcaraz_processed.csv"):
    print("\n--- PHASE 3: Natural Language Processing ---")

    df = df.copy()
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['preprocessed_text'] = df['text'].apply(preprocess)

    sents = [get_vader_sentiment(t) for t in df['cleaned_text']]
    df['sentiment_category'] = [s[0] for s in sents]
    df['sentiment_compound'] = [s[1] for s in sents]

    # --- Emotion Analysis (NRC Emotion Lexicon) ---
    df = add_emotion_columns(df, text_col='cleaned_text')
    print_emotion_summary(df)

    # Named Entity Recognition (NER) with spaCy
    df['entities'] = df['cleaned_text'].apply(extract_ner)

    # Named Entity Linking (NEL) with DBpedia Spotlight
    dbpedia_cache = {}
    state = {"circuit_broken": False, "broken_at": 0.0, "cooldown_seconds": 60}

    print("Executing Named Entity Linking via DBpedia Spotlight...")
    df['linked_entities'] = df.apply(lambda r: link_entities_dbpedia(r, dbpedia_cache, state), axis=1)

    all_uris = []
    for _, row in df.iterrows():
        for ent in row['linked_entities']:
            all_uris.append((ent['uri'], ent['surface_form']))

    if all_uris:
        uri_df = pd.DataFrame(all_uris, columns=['uri', 'surface_form'])
        top_linked = uri_df.groupby(['uri', 'surface_form']).size().reset_index(name='count')
        top_linked = top_linked.sort_values(by='count', ascending=False).head(15)
        print("\n--- Top Linked Entities (DBpedia Spotlight NEL) ---")
        for idx, r in top_linked.iterrows():
            print(f"- {r['surface_form']} ({r['uri']}): {r['count']} mentions")

    # Correlate Sentiment via Named Entity Linking (NEL)
    sinner_scores = []
    alcaraz_scores = []
    for _, row in df.iterrows():
        linked_ents = row['linked_entities']
        comp = row['sentiment_compound']

        uris = [ent['uri'] for ent in linked_ents]
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
    print(f"Enriched NLP features saved to {output_filepath}")
    
    return {
        "df": df,
        "sinner_scores": sinner_scores,
        "alcaraz_scores": alcaraz_scores
    }
