import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import save_plot_copies

def build_networks(df, did_to_handle):
    G = nx.Graph()
    Gd = nx.DiGraph()

    for _, row in df.iterrows():
        src_did = row['author_did']
        if pd.isna(src_did):
            continue
        source = did_to_handle.get(src_did, src_did)

        # A. Add reply relations
        parent_did = row['reply_parent_did']
        if pd.notna(parent_did):
            target = did_to_handle.get(parent_did, parent_did)
            if source != target: # Avoid self-replies
                # Undirected
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1, relationship="REPLY")
                # Directed
                if Gd.has_edge(source, target):
                    Gd[source][target]['weight'] += 1
                else:
                    Gd.add_edge(source, target, weight=1, relationship="REPLY")

        # B. Add mention relations
        for m in row['mentions']:
            m_did = m.get('did') if isinstance(m, dict) else m
            if m_did:
                target = did_to_handle.get(m_did, m_did)
                if source != target: # Avoid self-mentions
                    # Undirected
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1, relationship="MENTION")
                    # Directed
                    if Gd.has_edge(source, target):
                        Gd[source][target]['weight'] += 1
                    else:
                        Gd.add_edge(source, target, weight=1, relationship="MENTION")

    return G, Gd


def calculate_centralities(G, Gd):
    # Calculate Centrality Metrics for Undirected Graph
    deg_cent = nx.degree_centrality(G)
    close_cent = nx.closeness_centrality(G)
    between_cent = nx.betweenness_centrality(G)

    # Calculate Centrality Metrics for Directed Graph
    in_deg_cent = nx.in_degree_centrality(Gd)
    out_deg_cent = nx.out_degree_centrality(Gd)
    close_cent_dir = nx.closeness_centrality(Gd)
    between_cent_dir = nx.betweenness_centrality(Gd)
    
    try:
        pagerank = nx.pagerank(Gd, weight='weight')
    except Exception:
        pagerank = {node: 0.0 for node in Gd.nodes()}

    # Set attributes
    nx.set_node_attributes(G, deg_cent, "degree_centrality")
    nx.set_node_attributes(G, close_cent, "closeness_centrality")
    nx.set_node_attributes(G, between_cent, "betweenness_centrality")

    return {
        "deg_cent": deg_cent,
        "close_cent": close_cent,
        "between_cent": between_cent,
        "in_deg_cent": in_deg_cent,
        "out_deg_cent": out_deg_cent,
        "close_cent_dir": close_cent_dir,
        "between_cent_dir": between_cent_dir,
        "pagerank": pagerank
    }


def run_community_detection(G, Gd):
    if len(G.nodes()) <= 1:
        print("Graph too small for community detection / GCC calculation.")
        return {
            "communities": [list(G.nodes())],
            "modularity_score": 0.0,
            "node_to_community": {},
            "infomap_communities": [],
            "infomap_modularity": 0.0,
            "node_to_infomap": {},
            "gcc": G,
            "gcc_size": len(G.nodes()),
            "gcc_fraction": 1.0 if len(G.nodes()) > 0 else 0.0,
            "deg_assort_undir": 0.0,
            "deg_assort_dir": 0.0,
            "comm_assort_undir": 0.0
        }

    # Louvain Method (Undirected)
    communities = nx.community.louvain_communities(G)
    modularity_score = nx.community.modularity(G, communities)

    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    nx.set_node_attributes(G, node_to_community, "community")
    nx.set_node_attributes(Gd, node_to_community, "community")

    # Global Graph Statistics
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    avg_clustering = nx.average_clustering(G)

    # Compute Giant Connected Component (GCC) on G
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc = G.subgraph(components[0])
    gcc_size = gcc.number_of_nodes()
    gcc_fraction = gcc_size / G.number_of_nodes()

    # GCC statistics: Average shortest path, diameter, radius, eccentricity (Lab 3)
    if gcc_size > 1:
        try:
            gcc_avg_path_length = nx.average_shortest_path_length(gcc)
        except Exception:
            gcc_avg_path_length = 0.0
        try:
            gcc_ecc = nx.eccentricity(gcc)
            gcc_diameter = nx.diameter(gcc, gcc_ecc)
            gcc_radius = nx.radius(gcc, gcc_ecc)
            min_ecc = min(gcc_ecc.values())
            max_ecc = max(gcc_ecc.values())
        except Exception as e:
            gcc_avg_path_length = 0.0
            gcc_diameter = 0.0
            gcc_radius = 0.0
            min_ecc = 0.0
            max_ecc = 0.0
    else:
        gcc_avg_path_length = 0.0
        gcc_diameter = 0.0
        gcc_radius = 0.0
        min_ecc = 0.0
        max_ecc = 0.0

    print("\n" + "=" * 50)
    print("GLOBAL SOCIAL NETWORK STATISTICS (Lab 3)")
    print("=" * 50)
    print(f"Network Density:                    {density:.6f}")
    print(f"Network Transitivity:               {transitivity:.6f}")
    print(f"Average Clustering Coefficient:     {avg_clustering:.6f}")
    print(f"GCC Size:                           {gcc_size} nodes ({gcc_fraction*100:.2f}% of graph)")
    print(f"GCC Average Shortest Path Length:   {gcc_avg_path_length:.4f}")
    print(f"GCC Diameter (Max Eccentricity):    {gcc_diameter:.1f}")
    print(f"GCC Radius (Min Eccentricity):      {gcc_radius:.1f}")
    print("=" * 50)

    # Save metrics to CSV
    try:
        os.makedirs("data", exist_ok=True)
        stats_df = pd.DataFrame({
            "Metric": [
                "Density", "Transitivity", "Average Clustering", 
                "GCC Size", "GCC Fraction", "GCC Avg Path Length", 
                "GCC Diameter", "GCC Radius", "GCC Min Eccentricity", "GCC Max Eccentricity"
            ],
            "Value": [
                density, transitivity, avg_clustering, 
                gcc_size, gcc_fraction, gcc_avg_path_length, 
                gcc_diameter, gcc_radius, min_ecc, max_ecc
            ]
        })
        stats_df.to_csv("data/network_global_metrics.csv", index=False)
    except Exception as e:
        print(f"Error saving global metrics: {e}")

    # Assortativity Analysis
    try:
        deg_assort_undir = nx.degree_assortativity_coefficient(G)
    except Exception as e:
        deg_assort_undir = 0.0
        print("Error calculating undirected degree assortativity:", e)

    try:
        deg_assort_dir = nx.degree_assortativity_coefficient(Gd)
    except Exception as e:
        deg_assort_dir = 0.0
        print("Error calculating directed degree assortativity:", e)

    try:
        comm_assort_undir = nx.attribute_assortativity_coefficient(G, "community")
    except Exception as e:
        comm_assort_undir = 0.0
        print("Error calculating undirected community assortativity:", e)

    return {
        "communities": communities,
        "modularity_score": modularity_score,
        "node_to_community": node_to_community,
        "infomap_communities": [],
        "infomap_modularity": 0.0,
        "node_to_infomap": {},
        "gcc": gcc,
        "gcc_size": gcc_size,
        "gcc_fraction": gcc_fraction,
        "deg_assort_undir": deg_assort_undir,
        "deg_assort_dir": deg_assort_dir,
        "comm_assort_undir": comm_assort_undir
    }


def save_initial_centrality_csv(G, centralities, comm_data, filepath="data/network_centrality_metrics.csv"):
    centrality_data = []
    
    # Unpack centralities
    deg_cent = centralities["deg_cent"]
    in_deg_cent = centralities["in_deg_cent"]
    out_deg_cent = centralities["out_deg_cent"]
    close_cent = centralities["close_cent"]
    close_cent_dir = centralities["close_cent_dir"]
    between_cent = centralities["between_cent"]
    between_cent_dir = centralities["between_cent_dir"]
    pagerank = centralities["pagerank"]
    
    # Unpack community mappings
    node_to_community = comm_data["node_to_community"]
    
    for node in G.nodes():
        centrality_data.append({
            "user": node,
            "community": node_to_community.get(node, 0) if len(G.nodes()) > 1 else 0,
            "degree_centrality_undirected": deg_cent.get(node, 0.0),
            "in_degree_centrality_directed": in_deg_cent.get(node, 0.0),
            "out_degree_centrality_directed": out_deg_cent.get(node, 0.0),
            "closeness_centrality_undirected": close_cent.get(node, 0.0),
            "closeness_centrality_directed": close_cent_dir.get(node, 0.0),
            "betweenness_centrality_undirected": between_cent.get(node, 0.0),
            "betweenness_centrality_directed": between_cent_dir.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0)
        })
    df_cent = pd.DataFrame(centrality_data)
    df_cent.to_csv(filepath, index=False)

    return df_cent


def plot_network_graphs(G, Gd, df_cent, comm_data, centralities, stance_results, output_dir="plots"):
    """
    Generate the Louvain undirected, Stance, and PageRank directed network visualisations.
    """
    # Unpack centralities
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    
    # Unpack community data
    communities = comm_data["communities"]
    node_to_community = comm_data["node_to_community"]
    modularity_score = comm_data["modularity_score"]
    
    # Unpack stance results
    stance_leanings = stance_results["stance_leanings"]
    stance_assort = stance_results["polarization"]["stance_assortativity"]

    nodes_in_relations = [n for n, d in G.degree() if d > 0]
    if len(nodes_in_relations) > 0:
        subG = G.subgraph(nodes_in_relations)
        pos = nx.spring_layout(subG, k=0.15, iterations=40, seed=42)
        top_10_nodes = df_cent.sort_values(by="degree_centrality_undirected", ascending=False).head(10)['user'].tolist()
        labels_to_draw = {node: node for node in subG.nodes() if node in top_10_nodes}
    else:
        print("Isolated graph / Not enough relationships to plot.")
        return

    # 1. Louvain Network Graph
    plt.figure(figsize=(12, 12))
    node_colors = []
    if len(subG.nodes()) > 1 and len(communities) > 0:
        cmap = plt.cm.get_cmap('tab20', len(communities))
        for node in subG.nodes():
            comm_id = node_to_community.get(node, 0)
            node_colors.append(cmap(comm_id))
    else:
        node_colors = '#3A6073'
        
    node_sizes = [50 + (deg_cent[node] * 1200) for node in subG.nodes()]
    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Undirected Social Network Graph (degree centrality & Louvain partitions)\n(Modularity Q: {modularity_score:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("network_graph.png")
    plt.close()

    # 2. Stance Network Graph
    plt.figure(figsize=(12, 12))
    node_colors_stance = []
    for node in subG.nodes():
        leaning = stance_leanings.get(node, "neutral")
        if leaning == "sinner":
            node_colors_stance.append("#3498db") # Blue
        elif leaning == "alcaraz":
            node_colors_stance.append("#e67e22") # Orange
        else:
            node_colors_stance.append("#95a5a6") # Grey

    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors_stance, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Stance Network Graph (Direct Sentiment Average)\n(Stance Assortativity: {stance_assort:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("stance_network_graph.png")
    plt.close()

    # 3. Directed Social Network Graph
    plt.figure(figsize=(12, 12))
    nodes_in_relations_dir = [n for n, d in Gd.degree() if d > 0]
    if len(nodes_in_relations_dir) > 0:
        subG_dir = Gd.subgraph(nodes_in_relations_dir)
        pos_dir = nx.spring_layout(subG_dir, k=0.15, iterations=40, seed=42)

        node_colors_dir = []
        if len(subG_dir.nodes()) > 1 and len(communities) > 0:
            cmap = plt.cm.get_cmap('tab20', len(communities))
            for node in subG_dir.nodes():
                comm_id = node_to_community.get(node, 0)
                node_colors_dir.append(cmap(comm_id))
        else:
            node_colors_dir = '#3A6073'

        node_sizes_dir = [50 + (pagerank.get(node, 0.0) * 18000) for node in subG_dir.nodes()]
        nx.draw_networkx_edges(subG_dir, pos_dir, alpha=0.2, edge_color="grey",
                               arrows=True, arrowstyle='-|>', arrowsize=12,
                               connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_nodes(subG_dir, pos_dir, node_size=node_sizes_dir, node_color=node_colors_dir, alpha=0.9)

        top_10_nodes_dir = df_cent.sort_values(by="pagerank", ascending=False).head(10)['user'].tolist()
        labels_to_draw_dir = {node: node for node in subG_dir.nodes() if node in top_10_nodes_dir}
        nx.draw_networkx_labels(subG_dir, pos_dir, labels=labels_to_draw_dir, font_size=9, font_weight="bold", font_color="#1e272c")
    else:
        plt.text(0.5, 0.5, "Isolated graph / Not enough relationships", ha='center', va='center')

    plt.title(f"Directed Social Network Graph (PageRank prestige & Louvain partitions)\n(Projected Modularity Q: {modularity_score:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("network_graph_directed.png")
    plt.close()
