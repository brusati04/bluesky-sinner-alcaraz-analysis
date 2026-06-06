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


def main() -> None:
    """Run the strict 3-phase social network and sentiment analysis pipeline."""
    
    # =========================================================================
    # PHASE 1: Data Preparation (The single source of truth)
    # =========================================================================
    print("\n>>> PHASE 1: Data Preparation & NLP/Stance Enrichment...")
    df_processed, did_to_handle = prepare_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    if df_processed is None:
        print("Error: Processing raw dataset failed.")
        return

    # =========================================================================
    # PHASE 2: Network Structural Analysis
    # =========================================================================
    print("\n>>> PHASE 2: Network Structural Analysis & Centrality Calculations...")
    Gu, Gd = build_networks(df_processed, did_to_handle)
    if Gu.number_of_nodes() == 0:
        print("Warning: Network has 0 nodes. SNA modeling cannot proceed.")
        return

    centralities = calculate_centralities(Gu, Gd)
    comm_data = run_community_detection(Gu, Gd)
    
    # Build df_cent by pulling stance mappings directly from the frozen df_processed columns
    df_cent = save_initial_centrality_csv(Gu, centralities, comm_data)
    
    user_stance_scores = df_processed.groupby('author_handle')['author_stance_score'].first().to_dict()
    user_stance_leanings = df_processed.groupby('author_handle')['author_stance_leaning'].first().to_dict()
    
    df_cent['stance_score'] = df_cent['user'].map(user_stance_scores).fillna(0.0)
    df_cent['stance_leaning'] = df_cent['user'].map(user_stance_leanings).fillna('neutral')
    df_cent.to_csv("data/network_centrality_metrics.csv", index=False)
    
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

    # =========================================================================
    # PHASE 3: Aggregation & Visualizations
    # =========================================================================
    print("\n>>> PHASE 3: Aggregations & Visualization Renderings...")
    Gu_filtered, Gd_filtered = get_filtered_networks(Gu, Gd)

    # Run stance calculations Phase 3 (aggregates community profiles, fanbase comparison studies, diagnostics)
    stance_results = run_stance_propagation(
        df=df_processed,
        Gu=Gu_filtered,
        Gd=Gd_filtered,
        df_cent=df_cent,
        threshold=0.05,
        min_posts=1,
        output_dir="plots",
        run_diagnostics=True
    )
    user_stances = stance_results['user_stances']

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
    
    print("[STANCE] Printing community 196 diagnostic posts...")
    print_community_196_posts(df_processed, Gu_filtered, community_id=196)
    
    print("\n>>> Social network analysis pipeline successfully completed.")


if __name__ == "__main__":
    main()
