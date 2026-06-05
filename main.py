import os
import sys
import random
import numpy as np

random.seed(42)
np.random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

from preprocessing import load_and_preprocess_data
from social_network_analysis import (build_networks, calculate_centralities,
                                      run_community_detection, save_initial_centrality_csv,
                                      plot_network_graphs, plot_filtered_network_graph,
                                      get_filtered_networks)
from social_sentiment_analysis import (run_nlp_enrichment,
                                        plot_community_emotion_profiles)
import export_dashboard_data

def main():
    processed_filepath = "data/sinner_alcaraz_processed.csv"

    if os.path.exists(processed_filepath):
        print(f"[INFO] Processed dataset found at {processed_filepath}. Loading it directly...")
        df_processed = pd.read_csv(processed_filepath, parse_dates=['created_at'])
        
        from utils import parse_list_col, build_did_to_handle
        for col in ['mentions', 'hashtags', 'links', 'linked_entities']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(parse_list_col)
        
        did_to_handle = build_did_to_handle(df_processed)
        
        # Re-derive sinner and alcaraz scores from the loaded dataset
        sinner_scores = []
        alcaraz_scores = []
        for _, row in df_processed.iterrows():
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
                
        nlp_results = {
            "df": df_processed,
            "sinner_scores": sinner_scores,
            "alcaraz_scores": alcaraz_scores
        }
    else:
        df, did_to_handle = load_and_preprocess_data("data/sinner_alcaraz_posts.csv")
        if df is None:
            return
        nlp_results = run_nlp_enrichment(df, output_filepath=processed_filepath)
        df_processed = nlp_results["df"]

    Gu, Gd = build_networks(df_processed, did_to_handle)
    if len(Gu.nodes()) == 0:
        print("Warning: Network has 0 nodes. SNA modeling cannot proceed.")
        return

    centralities = calculate_centralities(Gu, Gd)
    comm_data = run_community_detection(Gu, Gd)
    df_cent = save_initial_centrality_csv(Gu, centralities, comm_data, filepath="data/network_centrality_metrics.csv")

    # Retrieve filtered subgraphs (aligned with the plots in the 'filtered' folder)
    Gu_filtered, Gd_filtered = get_filtered_networks(Gu, Gd, min_component_size=10)

    # Configuration for Study 2: Emotional Profiling selection filter
    TOP_K_COMMUNITIES = 3          # Limit to top k communities
    SORT_BY_CRITERIA = "node_count" # "node_count" or "post_volume"

    # 2. Study 2: Emotional Profiling of Communities (strictly filtered)
    plot_community_emotion_profiles(df_processed, Gu_filtered, comm_data["node_to_louvain"], comm_data["louvain_communities"], output_dir="plots", title_suffix=" (Louvain - Filtered)", top_k=TOP_K_COMMUNITIES, sort_by=SORT_BY_CRITERIA)
    plot_community_emotion_profiles(df_processed, Gd_filtered, comm_data["node_to_infomap"], comm_data["infomap_communities"], output_dir="plots", title_suffix=" (Infomap - Filtered)", top_k=TOP_K_COMMUNITIES, sort_by=SORT_BY_CRITERIA)

    # Generate network graph visualizations
    plot_network_graphs(Gu, Gd, df_cent, comm_data, centralities, output_dir="plots")
    plot_filtered_network_graph(Gu, Gd, df_cent, comm_data, centralities, df_processed=df_processed, output_dir="plots/filtered", top_k=TOP_K_COMMUNITIES, sort_by=SORT_BY_CRITERIA)

if __name__ == "__main__":
    import pandas as pd
    main()
