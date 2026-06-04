import os
import sys
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Ensure src/ is in the import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from utils import load_and_preprocess_data
from network_analysis import (build_networks, calculate_centralities,
                              run_community_detection, save_initial_centrality_csv)
from nlp_analysis import run_nlp_enrichment, analyze_community_sentiment_polarization
from stance_propagation import run_stance_propagation
from visualization import generate_visualizations
import export_dashboard_data

def main():
    df, did_to_handle = load_and_preprocess_data("data/sinner_alcaraz_posts.csv")
    if df is None:
        return

    nlp_results = run_nlp_enrichment(df, output_filepath="data/sinner_alcaraz_processed.csv")
    df_processed = nlp_results["df"]

    G, Gd = build_networks(df_processed, did_to_handle)
    if len(G.nodes()) == 0:
        print("Warning: Network has 0 nodes. SNA modeling cannot proceed.")
        return

    centralities = calculate_centralities(G, Gd)
    comm_data = run_community_detection(G, Gd)
    df_cent = save_initial_centrality_csv(G, centralities, comm_data, filepath="data/network_centrality_metrics.csv")

    analyze_community_sentiment_polarization(
        df_processed, G, comm_data["node_to_community"], comm_data["communities"],
        output_dir="plots", title_suffix=""
    )

    stance_results = run_stance_propagation(df_processed, G, Gd, df_cent, filepath="data/network_centrality_metrics.csv")
    generate_visualizations(G, Gd, stance_results["df_cent"], comm_data, centralities, stance_results, nlp_results)
    export_dashboard_data.main()

if __name__ == "__main__":
    main()
