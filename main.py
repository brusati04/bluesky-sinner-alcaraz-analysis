import os
import sys
import random

import numpy as np

random.seed(42)
np.random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

from preprocessing import prepare_dataset, plot_community_wordcloud
from social_network_analysis import (build_networks, calculate_centralities,
                                      run_community_detection, save_initial_centrality_csv,
                                      plot_network_graphs, plot_filtered_network_graph,
                                      get_filtered_networks)
from social_sentiment_analysis import (plot_community_emotion_profiles,
                                      plot_sentiment_distribution,
                                      plot_sentiment_over_time)

RAW_DATA_PATH = "data/sinner_alcaraz_posts.csv"
PROCESSED_DATA_PATH = "data/sinner_alcaraz_processed.csv"


def main() -> None:
    """Run the end-to-end social network and sentiment analysis pipeline."""
    df_processed, did_to_handle = prepare_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    if df_processed is None:
        return

    Gu, Gd = build_networks(df_processed, did_to_handle)
    if Gu.number_of_nodes() == 0:
        print("Warning: Network has 0 nodes. SNA modeling cannot proceed.")
        return

    centralities = calculate_centralities(Gu, Gd)
    comm_data = run_community_detection(Gu, Gd)
    df_cent = save_initial_centrality_csv(Gu, centralities, comm_data)

    Gu_filtered, Gd_filtered = get_filtered_networks(Gu, Gd)

    plot_network_graphs(Gu, Gd, df_cent, comm_data, centralities)
    plot_filtered_network_graph(Gu, Gd, df_cent, comm_data, centralities, df_processed)

    plot_community_emotion_profiles(df_processed, Gu_filtered, comm_data["node_to_louvain"],
                                    title_suffix=" (Louvain - Filtered)")
    plot_community_emotion_profiles(df_processed, Gd_filtered, comm_data["node_to_infomap"],
                                    title_suffix=" (Infomap - Filtered)")

    print("[NLP] Generating sentiment visualization plots...")
    plot_sentiment_distribution(df_processed)
    plot_sentiment_over_time(df_processed)
    print("[NLP] Generating community word cloud...")
    plot_community_wordcloud(df_processed)
    print("[NLP] Plots successfully generated and saved to plots/ and report/")


if __name__ == "__main__":
    main()
