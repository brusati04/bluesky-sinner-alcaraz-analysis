import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import save_plot_copies

def build_networks(df, did_to_handle):
    Gu = nx.Graph()
    Gd = nx.DiGraph()

    for _, row in df.iterrows():
        src_did = row['author_did']
        if pd.isna(src_did):
            continue
        source = did_to_handle.get(src_did, src_did)

        # Add reply relations
        parent_did = row['reply_parent_did']
        if pd.notna(parent_did):
            target = did_to_handle.get(parent_did, parent_did)
            if source != target: # Avoid self-replies
                # Undirected
                if Gu.has_edge(source, target):
                    Gu[source][target]['weight'] += 1
                else:
                    Gu.add_edge(source, target, weight=1, relationship="REPLY")
                # Directed
                if Gd.has_edge(source, target):
                    Gd[source][target]['weight'] += 1
                else:
                    Gd.add_edge(source, target, weight=1, relationship="REPLY")

        # Add mention relations
        for m in row['mentions']:
            m_did = m.get('did') if isinstance(m, dict) else m
            if m_did:
                target = did_to_handle.get(m_did, m_did)
                if source != target: # Avoid self-mentions
                    # Undirected
                    if Gu.has_edge(source, target):
                        Gu[source][target]['weight'] += 1
                    else:
                        Gu.add_edge(source, target, weight=1, relationship="MENTION")
                    # Directed
                    if Gd.has_edge(source, target):
                        Gd[source][target]['weight'] += 1
                    else:
                        Gd.add_edge(source, target, weight=1, relationship="MENTION")

    return Gu, Gd


def calculate_centralities(Gu, Gd):
    # Calculate Centrality Metrics for Undirected Graph
    deg_cent = nx.degree_centrality(Gu)
    close_cent = nx.closeness_centrality(Gu)
    between_cent = nx.betweenness_centrality(Gu)

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
    nx.set_node_attributes(Gu, deg_cent, "degree_centrality")
    nx.set_node_attributes(Gu, close_cent, "closeness_centrality")
    nx.set_node_attributes(Gu, between_cent, "betweenness_centrality")

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


def run_community_detection(Gu, Gd):
    if len(Gu.nodes()) <= 1:
        print("Graph too small for community detection / GCC calculation.")
        return {
            "communities": [list(Gu.nodes())],
            "modularity_score": 0.0,
            "node_to_community": {},
            "infomap_communities": [],
            "infomap_modularity": 0.0,
            "node_to_infomap": {},
            "gcc": Gu,
            "gcc_size": len(Gu.nodes()),
            "gcc_fraction": 1.0 if len(Gu.nodes()) > 0 else 0.0,
            "deg_assort_undir": 0.0,
            "deg_assort_dir": 0.0,
            "comm_assort_undir": 0.0
        }

    # 1. Louvain Method (Undirected)
    communities = nx.community.louvain_communities(Gu)
    modularity_score = nx.community.modularity(Gu, communities)

    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    nx.set_node_attributes(Gu, node_to_community, "community")
    nx.set_node_attributes(Gd, node_to_community, "community")

    # 2. Infomap Method (Directed Flow)
    import infomap
    im = infomap.Infomap("--two-level --silent")
    
    # Map node names to integer IDs for the C++ Infomap engine
    node_to_id = {node: idx for idx, node in enumerate(Gd.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}
    
    for u, v, data in Gd.edges(data=True):
        im.add_link(node_to_id[u], node_to_id[v], float(data.get('weight', 1.0)))
        
    im.run()
    
    node_to_infomap = {}
    infomap_communities_dict = {}
    for node_it in im.iterLeafNodes():
        n_id = node_it.physicalId
        m_id = node_it.module_id
        if n_id in id_to_node:
            orig_node = id_to_node[n_id]
            node_to_infomap[orig_node] = m_id
            if m_id not in infomap_communities_dict:
                infomap_communities_dict[m_id] = []
            infomap_communities_dict[m_id].append(orig_node)
            
    # Assign default/fallback community ID for any isolated nodes that Infomap might skip
    for node in Gd.nodes():
        if node not in node_to_infomap:
            node_to_infomap[node] = -1
            
    infomap_communities = [set(nodes) for nodes in infomap_communities_dict.values()]
    
    try:
        infomap_modularity = nx.community.modularity(Gd, infomap_communities)
    except Exception:
        infomap_modularity = 0.0
        
    nx.set_node_attributes(Gu, node_to_infomap, "community_infomap")
    nx.set_node_attributes(Gd, node_to_infomap, "community_infomap")

    # Global Graph Statistics
    density = nx.density(Gu)
    transitivity = nx.transitivity(Gu)
    avg_clustering = nx.average_clustering(Gu)

    # Compute Giant Connected Component (GCC) on Gu
    components = sorted(nx.connected_components(Gu), key=len, reverse=True)
    gcc = Gu.subgraph(components[0])
    gcc_size = gcc.number_of_nodes()
    gcc_fraction = gcc_size / Gu.number_of_nodes()

    # GCC statistics: Average shortest path, diameter, radius, eccentricity
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
    print("GLOBAL SOCIAL NETWORK STATISTICS")
    print("=" * 50)
    print(f"Network Density:                    {density:.6f}")
    print(f"Network Transitivity:               {transitivity:.6f}")
    print(f"Average Clustering Coefficient:     {avg_clustering:.6f}")
    print(f"GCC Size:                           {gcc_size} nodes ({gcc_fraction*100:.2f}% of graph)")
    print(f"GCC Average Shortest Path Length:   {gcc_avg_path_length:.4f}")
    print(f"GCC Diameter (Max Eccentricity):    {gcc_diameter:.1f}")
    print(f"GCC Radius (Min Eccentricity):      {gcc_radius:.1f}")
    print(f"Louvain Communities:                {len(communities)} (Modularity Q: {modularity_score:.4f})")
    print(f"Infomap Communities:                {len(infomap_communities)} (Modularity Q: {infomap_modularity:.4f})")
    print("=" * 50)

    # Save metrics to CSV
    try:
        os.makedirs("data", exist_ok=True)
        stats_df = pd.DataFrame({
            "Metric": [
                "Density", "Transitivity", "Average Clustering", 
                "GCC Size", "GCC Fraction", "GCC Avg Path Length", 
                "GCC Diameter", "GCC Radius", "GCC Min Eccentricity", "GCC Max Eccentricity",
                "Louvain Modularity", "Infomap Modularity"
            ],
            "Value": [
                density, transitivity, avg_clustering, 
                gcc_size, gcc_fraction, gcc_avg_path_length, 
                gcc_diameter, gcc_radius, min_ecc, max_ecc,
                modularity_score, infomap_modularity
            ]
        })
        stats_df.to_csv("data/network_global_metrics.csv", index=False)
    except Exception as e:
        print(f"Error saving global metrics: {e}")

    # Assortativity Analysis
    try:
        deg_assort_undir = nx.degree_assortativity_coefficient(Gu)
    except Exception as e:
        deg_assort_undir = 0.0
        print("Error calculating undirected degree assortativity:", e)

    try:
        deg_assort_dir = nx.degree_assortativity_coefficient(Gd)
    except Exception as e:
        deg_assort_dir = 0.0
        print("Error calculating directed degree assortativity:", e)

    try:
        comm_assort_undir = nx.attribute_assortativity_coefficient(Gu, "community")
    except Exception as e:
        comm_assort_undir = 0.0
        print("Error calculating undirected community assortativity:", e)

    return {
        "louvain_communities": communities,
        "modularity_score": modularity_score,
        "node_to_louvain": node_to_community,
        "infomap_communities": infomap_communities,
        "infomap_modularity": infomap_modularity,
        "node_to_infomap": node_to_infomap,
        "gcc": gcc,
        "gcc_size": gcc_size,
        "gcc_fraction": gcc_fraction,
        "deg_assort_undir": deg_assort_undir,
        "deg_assort_dir": deg_assort_dir,
        "comm_assort_undir": comm_assort_undir
    }


def save_initial_centrality_csv(Gu, centralities, comm_data, filepath="data/network_centrality_metrics.csv"):
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
    node_to_louvain = comm_data["node_to_louvain"]
    node_to_infomap = comm_data.get("node_to_infomap", {})
    
    for node in Gu.nodes():
        centrality_data.append({
            "user": node,
            "community": node_to_louvain.get(node, 0) if len(Gu.nodes()) > 1 else 0,
            "community_infomap": node_to_infomap.get(node, -1) if len(Gu.nodes()) > 1 else -1,
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


def get_filtered_networks(Gu, Gd, min_component_size=10):
    Gu_filtered = Gu.copy()
    for component in list(nx.connected_components(Gu_filtered)):
        if len(component) <= min_component_size:
            Gu_filtered.remove_nodes_from(component)

    Gd_filtered = Gd.copy()
    for component in list(nx.weakly_connected_components(Gd_filtered)):
        if len(component) <= min_component_size:
            Gd_filtered.remove_nodes_from(component)

    return Gu_filtered, Gd_filtered


def plot_filtered_network_graph(Gu, Gd, df_cent, comm_data, centralities, stance_results, min_component_size=10, output_dir="plots"):
    """
    Same as plot_network_graphs but drops connected components with <= min_component_size nodes.
    Positions are computed from the full original graphs so surviving nodes stay in place.
    Does not modify the original graphs.
    """
    # Compute layouts from the original graphs before filtering
    nodes_with_edges = [n for n, d in Gu.degree() if d > 0]
    pos = nx.spring_layout(Gu.subgraph(nodes_with_edges), k=0.15, iterations=40, seed=42)

    nodes_with_edges_dir = [n for n, d in Gd.degree() if d > 0]
    pos_dir = nx.spring_layout(Gd.subgraph(nodes_with_edges_dir), k=0.15, iterations=40, seed=42)

    Gu_plot = Gu.copy()
    for component in list(nx.connected_components(Gu_plot)):
        if len(component) <= min_component_size:
            Gu_plot.remove_nodes_from(component)

    Gd_plot = Gd.copy()
    for component in list(nx.weakly_connected_components(Gd_plot)):
        if len(component) <= min_component_size:
            Gd_plot.remove_nodes_from(component)

    plot_network_graphs(Gu_plot, Gd_plot, df_cent, comm_data, centralities, stance_results, output_dir=output_dir, pos=pos, pos_dir=pos_dir)


def plot_network_graphs(Gu, Gd, df_cent, comm_data, centralities, stance_results, output_dir="plots", pos=None, pos_dir=None):
    """
    Generate the Louvain undirected, Stance, and PageRank directed network visualisations.
    pos / pos_dir: precomputed layouts; if provided they are reused as-is (extra nodes are ignored).
    """
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    communities = comm_data["louvain_communities"]
    node_to_community = comm_data["node_to_louvain"]
    modularity_score = comm_data["modularity_score"]
    stance_leanings = stance_results["stance_leanings"]
    stance_assort = stance_results["polarization"]["stance_assortativity"]

    nodes_in_relations = [n for n, d in Gu.degree() if d > 0]
    if len(nodes_in_relations) > 0:
        subG = Gu.subgraph(nodes_in_relations)
        if pos is None:
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
    save_plot_copies("network_graph.png", output_dir=output_dir)
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
    save_plot_copies("stance_network_graph.png", output_dir=output_dir)
    plt.close()

    # 3. Directed Social Network Graph
    plt.figure(figsize=(12, 12))
    nodes_in_relations_dir = [n for n, d in Gd.degree() if d > 0]
    if len(nodes_in_relations_dir) > 0:
        subG_dir = Gd.subgraph(nodes_in_relations_dir)
        if pos_dir is None:
            pos_dir = nx.spring_layout(subG_dir, k=0.15, iterations=40, seed=42)

        node_colors_dir = []
        infomap_communities = comm_data.get("infomap_communities", [])
        node_to_infomap = comm_data.get("node_to_infomap", {})
        infomap_modularity = comm_data.get("infomap_modularity", 0.0)

        if len(subG_dir.nodes()) > 1 and len(infomap_communities) > 0:
            cmap = plt.cm.get_cmap('tab20', len(infomap_communities))
            for node in subG_dir.nodes():
                comm_id = node_to_infomap.get(node, 0)
                node_colors_dir.append(cmap(comm_id % len(infomap_communities)))
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

    plt.title(f"Directed Social Network Graph (PageRank prestige & Infomap partitions)\n(Projected Modularity Q: {infomap_modularity:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("network_graph_directed.png", output_dir=output_dir)
    plt.close()
