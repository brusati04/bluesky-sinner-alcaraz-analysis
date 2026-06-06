"""Pipeline entrypoint — a strict, ordered DAG of analysis stages.

    [1] crawl       (src/crawler.py, run separately) -> data/sinner_alcaraz_posts.csv
    [2] preprocess  raw posts            -> data/sinner_alcaraz_processed.csv (FROZEN)
    [3] network     processed CSV        -> network_*.csv, plots/network_*.png
    [4] sentiment   processed + network  -> plots/sentiment_*, community_emotion_*
    [5] stance      processed + network  -> plots/stance_*, fanbase_*

Each stage reads only the artifact(s) of an upstream stage and is independently
runnable if its inputs exist. The single source of truth is the frozen processed
CSV produced by stage [2]; downstream stages never rewrite it.
"""

import os
import sys
import random

import numpy as np
import networkx as nx

random.seed(42)
np.random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

from preprocessing import prepare_dataset, plot_community_wordcloud
from social_network_analysis import (build_networks, calculate_centralities,
                                      run_community_detection, save_initial_centrality_csv,
                                      plot_network_graphs, plot_filtered_network_graph,
                                      plot_network_graphs_by_emotion,
                                      plot_network_graphs_by_fanbase,
                                      get_filtered_networks)
from social_sentiment_analysis import (plot_community_emotion_profiles,
                                      plot_sentiment_distribution,
                                      plot_sentiment_over_time,
                                      plot_emotion_backend_comparison)
from social_stance_analysis import (run_stance_propagation,
                                    print_community_196_posts)


RAW_DATA_PATH = "data/sinner_alcaraz_posts.csv"
PROCESSED_DATA_PATH = "data/sinner_alcaraz_processed.csv"
CENTRALITY_METRICS_PATH = "data/network_centrality_metrics.csv"


# =============================================================================
# STAGE [2]: Preprocessing — the single source of truth
# =============================================================================
def stage_preprocess(raw_path: str = RAW_DATA_PATH, processed_path: str = PROCESSED_DATA_PATH):
    """Build (or load from cache) the frozen, fully-enriched processed dataset.

    Reads the raw crawled posts and returns ``(df_processed, did_to_handle)``.
    All NLP/stance enrichment is owned here; if the processed CSV already exists it
    is loaded as-is and never recomputed.
    """
    print("\n>>> STAGE [2] Preprocessing & NLP/Stance Enrichment...")
    df_processed, did_to_handle = prepare_dataset(raw_path, processed_path)
    if df_processed is None:
        print("Error: Processing raw dataset failed.")
    return df_processed, did_to_handle


# =============================================================================
# STAGE [3]: Social network analysis (graph, centralities, communities, plots)
# =============================================================================
def stage_network(df_processed, did_to_handle, metrics_path: str = CENTRALITY_METRICS_PATH):
    """Build the interaction graphs, compute centralities + communities, and render
    the network plots. Reads only the frozen processed dataset (+ did->handle map).

    Returns a dict of network artifacts consumed by the sentiment/stance stages.
    """
    print("\n>>> STAGE [3] Network Structural Analysis & Centrality Calculations...")
    Gu, Gd = build_networks(df_processed, did_to_handle)
    if Gu.number_of_nodes() == 0:
        print("Warning: Network has 0 nodes. SNA modeling cannot proceed.")
        return None

    centralities = calculate_centralities(Gu, Gd)
    comm_data = run_community_detection(Gu, Gd)

    # Build df_cent by pulling stance mappings directly from the frozen df_processed columns
    df_cent = save_initial_centrality_csv(Gu, centralities, comm_data)

    user_stance_scores = df_processed.groupby('author_handle')['author_stance_score'].first().to_dict()
    user_stance_leanings = df_processed.groupby('author_handle')['author_stance_leaning'].first().to_dict()

    df_cent['stance_score'] = df_cent['user'].map(user_stance_scores).fillna(0.0)
    df_cent['stance_leaning'] = df_cent['user'].map(user_stance_leanings).fillna('neutral')
    df_cent.to_csv(metrics_path, index=False)

    # Populate stance mapping onto graph nodes directly from the precomputed frozen columns
    nx.set_node_attributes(Gu, user_stance_scores, 'stance_score')
    nx.set_node_attributes(Gd, user_stance_scores, 'stance_score')
    nx.set_node_attributes(Gu, user_stance_leanings, 'stance_leaning')
    nx.set_node_attributes(Gd, user_stance_leanings, 'stance_leaning')

    # Draw pure network structures and layouts
    pos, pos_dir = plot_network_graphs(Gu, Gd, df_cent, comm_data, centralities)
    Gu_plot, Gd_plot, pos_f, pos_dir_f = plot_filtered_network_graph(
        Gu, Gd, df_cent, comm_data, centralities, df_processed)

    # Render network plots using precalculated dominant_emotion and stance_leaning columns
    for backend in ("nrc", "bert"):
        plot_network_graphs_by_emotion(Gu, Gd, df_cent, centralities, df_processed,
                                       backend=backend, pos=pos, pos_dir=pos_dir)
        plot_network_graphs_by_emotion(Gu_plot, Gd_plot, df_cent, centralities, df_processed,
                                       backend=backend, output_dir="plots/filtered",
                                       pos=pos_f, pos_dir=pos_dir_f)

    print("[STANCE] Generating fanbase-coloured network graphs...")
    plot_network_graphs_by_fanbase(Gu, Gd, df_cent, centralities, pos=pos, pos_dir=pos_dir)
    plot_network_graphs_by_fanbase(Gu_plot, Gd_plot, df_cent, centralities, output_dir="plots/filtered", pos=pos_f, pos_dir=pos_dir_f)

    Gu_filtered, Gd_filtered = get_filtered_networks(Gu, Gd)

    return {
        "Gu": Gu, "Gd": Gd,
        "centralities": centralities,
        "comm_data": comm_data,
        "df_cent": df_cent,
        "Gu_filtered": Gu_filtered,
        "Gd_filtered": Gd_filtered,
    }


# =============================================================================
# STAGE [5]: Social stance analysis (aggregation, polarization, fanbase, diagnostics)
# =============================================================================
def stage_stance(df_processed, net):
    """Aggregate user/community stances, polarization and fanbase study from the
    frozen columns + filtered graphs, render stance plots, and run diagnostics.

    Returns the reconstructed ``user_stances`` frame (consumed by the sentiment stage).
    """
    print("\n>>> STAGE [5] Stance Aggregation, Polarization & Fanbase Study...")
    stance_results = run_stance_propagation(
        df=df_processed,
        Gu=net["Gu_filtered"],
        Gd=net["Gd_filtered"],
        df_cent=net["df_cent"],
        threshold=0.05,
        min_posts=1,
        output_dir="plots",
        run_diagnostics=True
    )
    return stance_results['user_stances']


# =============================================================================
# STAGE [4]: Social sentiment analysis (emotion clusters, sentiment plots)
# =============================================================================
def stage_sentiment(df_processed, net, user_stances):
    """Render emotion-cluster, sentiment-distribution, sentiment-over-time and
    backend-comparison plots from the frozen columns + filtered graphs."""
    print("\n>>> STAGE [4] Sentiment & Emotion Visualizations...")
    comm_data = net["comm_data"]
    Gu_filtered, Gd_filtered = net["Gu_filtered"], net["Gd_filtered"]

    # Generate flat sentiment profiles and distributions
    plot_community_emotion_profiles(df_processed, Gu_filtered, comm_data["node_to_louvain"],
                                    title_suffix=" (Louvain - Filtered)", backend="nrc")
    plot_community_emotion_profiles(df_processed, Gu_filtered, comm_data["node_to_louvain"],
                                    title_suffix=" (Louvain - Filtered)", backend="bert")
    plot_community_emotion_profiles(df_processed, Gd_filtered, comm_data["node_to_infomap"],
                                    title_suffix=" (Infomap - Filtered)", backend="nrc")
    plot_community_emotion_profiles(df_processed, Gd_filtered, comm_data["node_to_infomap"],
                                    title_suffix=" (Infomap - Filtered)", backend="bert")

    print("[NLP] Generating sentiment visualization plots...")
    plot_sentiment_distribution(df_processed)
    plot_sentiment_over_time(df_processed, user_stances=user_stances)
    plot_emotion_backend_comparison(df_processed)

    print("[NLP] Generating community word cloud...")
    plot_community_wordcloud(df_processed)


def main() -> None:
    """Run the ordered DAG: preprocess -> network -> stance -> sentiment."""
    # ---- Stage [2]
    df_processed, did_to_handle = stage_preprocess()
    if df_processed is None:
        return

    # ---- Stage [3]
    net = stage_network(df_processed, did_to_handle)
    if net is None:
        return

    # ---- Stage [5] (produces user_stances consumed by stage [4]) then Stage [4]
    user_stances = stage_stance(df_processed, net)
    stage_sentiment(df_processed, net, user_stances)

    print("[STANCE] Printing community 196 diagnostic posts...")
    print_community_196_posts(df_processed, net["Gu_filtered"], community_id=196)

    print("\n>>> Social network analysis pipeline successfully completed.")


if __name__ == "__main__":
    main()
