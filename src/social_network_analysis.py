import os
from collections import Counter
from typing import Optional

import infomap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import save_plot_copies


def build_networks(df: pd.DataFrame, did_to_handle: dict) -> tuple[nx.Graph, nx.DiGraph]:
    """Build undirected and directed interaction graphs from reply and mention relations.

    Nodes are author handles (falling back to DIDs); edge weights count repeated
    interactions. Self-loops are skipped.
    """
    Gu = nx.Graph()
    Gd = nx.DiGraph()

    def add_edge(source: str, target: str, relationship: str) -> None:
        if source == target:
            return
        for graph in (Gu, Gd):
            if graph.has_edge(source, target):
                graph[source][target]['weight'] += 1
            else:
                graph.add_edge(source, target, weight=1, relationship=relationship)

    for _, row in df.iterrows():
        src_did = row['author_did']
        if pd.isna(src_did):
            continue
        source = did_to_handle.get(src_did, src_did)

        parent_did = row['reply_parent_did']
        if pd.notna(parent_did):
            add_edge(source, did_to_handle.get(parent_did, parent_did), "REPLY")

        for m in row['mentions']:
            m_did = m.get('did') if isinstance(m, dict) else m
            if m_did:
                add_edge(source, did_to_handle.get(m_did, m_did), "MENTION")

    return Gu, Gd


def calculate_centralities(Gu: nx.Graph, Gd: nx.DiGraph) -> dict:
    """Compute degree/closeness/betweenness centralities (both graphs) plus PageRank on Gd.

    Undirected centralities are also written back as node attributes on Gu.
    """
    deg_cent = nx.degree_centrality(Gu)
    close_cent = nx.closeness_centrality(Gu)
    between_cent = nx.betweenness_centrality(Gu)

    in_deg_cent = nx.in_degree_centrality(Gd)
    out_deg_cent = nx.out_degree_centrality(Gd)
    close_cent_dir = nx.closeness_centrality(Gd)
    between_cent_dir = nx.betweenness_centrality(Gd)

    try:
        pagerank = nx.pagerank(Gd, weight='weight')
    except Exception:
        pagerank = {node: 0.0 for node in Gd.nodes()}

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
        "pagerank": pagerank,
    }


def run_community_detection(Gu: nx.Graph, Gd: nx.DiGraph) -> dict:
    """Detect communities (Louvain on Gu, Infomap on Gd) and compute global network statistics.

    Writes community attributes back onto both graphs, prints a summary, saves global
    metrics to data/network_global_metrics.csv, and returns all derived structures.
    """
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
            "comm_assort_undir": 0.0,
        }

    communities = nx.community.louvain_communities(Gu)
    modularity_score = nx.community.modularity(Gu, communities)

    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    nx.set_node_attributes(Gu, node_to_community, "community")
    nx.set_node_attributes(Gd, node_to_community, "community")

    # Infomap operates on integer node IDs, so map handles to indices and back.
    im = infomap.Infomap("--two-level --silent")
    node_to_id = {node: idx for idx, node in enumerate(Gd.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}
    for u, v, data in Gd.edges(data=True):
        im.add_link(node_to_id[u], node_to_id[v], float(data.get('weight', 1.0)))
    im.run()

    node_to_infomap = {}
    infomap_communities_dict: dict[int, list] = {}
    for node_it in im.iterLeafNodes():
        orig_node = id_to_node.get(node_it.physicalId)
        if orig_node is not None:
            node_to_infomap[orig_node] = node_it.module_id
            infomap_communities_dict.setdefault(node_it.module_id, []).append(orig_node)

    for node in Gd.nodes():
        node_to_infomap.setdefault(node, -1)

    infomap_communities = [set(nodes) for nodes in infomap_communities_dict.values()]
    try:
        infomap_modularity = nx.community.modularity(Gd, infomap_communities)
    except Exception:
        infomap_modularity = 0.0

    nx.set_node_attributes(Gu, node_to_infomap, "community_infomap")
    nx.set_node_attributes(Gd, node_to_infomap, "community_infomap")

    density = nx.density(Gu)
    transitivity = nx.transitivity(Gu)
    avg_clustering = nx.average_clustering(Gu)

    components = sorted(nx.connected_components(Gu), key=len, reverse=True)
    gcc = Gu.subgraph(components[0])
    gcc_size = gcc.number_of_nodes()
    gcc_fraction = gcc_size / Gu.number_of_nodes()

    gcc_avg_path_length = gcc_diameter = gcc_radius = min_ecc = max_ecc = 0.0
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
        except Exception:
            gcc_diameter = gcc_radius = min_ecc = max_ecc = 0.0

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

    try:
        os.makedirs("data", exist_ok=True)
        stats_df = pd.DataFrame({
            "Metric": [
                "Density", "Transitivity", "Average Clustering",
                "GCC Size", "GCC Fraction", "GCC Avg Path Length",
                "GCC Diameter", "GCC Radius", "GCC Min Eccentricity", "GCC Max Eccentricity",
                "Louvain Modularity", "Infomap Modularity",
            ],
            "Value": [
                density, transitivity, avg_clustering,
                gcc_size, gcc_fraction, gcc_avg_path_length,
                gcc_diameter, gcc_radius, min_ecc, max_ecc,
                modularity_score, infomap_modularity,
            ],
        })
        stats_df.to_csv("data/network_global_metrics.csv", index=False)
    except Exception as e:
        print(f"Error saving global metrics: {e}")

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
        "comm_assort_undir": comm_assort_undir,
    }


def save_initial_centrality_csv(
    Gu: nx.Graph,
    centralities: dict,
    comm_data: dict,
    filepath: str = "data/network_centrality_metrics.csv",
) -> pd.DataFrame:
    """Assemble per-node centrality and community metrics into a DataFrame and save it to CSV."""
    multi_node = len(Gu.nodes()) > 1
    node_to_louvain = comm_data["node_to_louvain"]
    node_to_infomap = comm_data.get("node_to_infomap", {})

    centrality_data = [{
        "user": node,
        "community": node_to_louvain.get(node, 0) if multi_node else 0,
        "community_infomap": node_to_infomap.get(node, -1) if multi_node else -1,
        "degree_centrality_undirected": centralities["deg_cent"].get(node, 0.0),
        "in_degree_centrality_directed": centralities["in_deg_cent"].get(node, 0.0),
        "out_degree_centrality_directed": centralities["out_deg_cent"].get(node, 0.0),
        "closeness_centrality_undirected": centralities["close_cent"].get(node, 0.0),
        "closeness_centrality_directed": centralities["close_cent_dir"].get(node, 0.0),
        "betweenness_centrality_undirected": centralities["between_cent"].get(node, 0.0),
        "betweenness_centrality_directed": centralities["between_cent_dir"].get(node, 0.0),
        "pagerank": centralities["pagerank"].get(node, 0.0),
    } for node in Gu.nodes()]

    df_cent = pd.DataFrame(centrality_data)
    df_cent.to_csv(filepath, index=False)
    return df_cent


def get_community_color_map(node_to_community: dict, cmap_name: str = "viridis") -> dict:
    """Map each community ID to a hex colour drawn from the named colormap.

    Qualitative colormaps are indexed directly; continuous ones are sampled evenly.
    """
    unique_cids = sorted(set(node_to_community.values()))
    n = len(unique_cids)

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    is_qualitative = cmap_name.lower().startswith(('pastel', 'paired', 'accent', 'dark2', 'set', 'tab'))

    color_map = {}
    for idx, cid in enumerate(unique_cids):
        if is_qualitative:
            rgba = cmap(idx % cmap.N)
        else:
            rgba = cmap(idx / (n - 1) if n > 1 else 0.5)
        color_map[cid] = mcolors.to_hex(rgba)
    return color_map


def get_filtered_networks(Gu: nx.Graph, Gd: nx.DiGraph, min_component_size: int = 10) -> tuple[nx.Graph, nx.DiGraph]:
    """Return copies of Gu/Gd with connected components of size <= min_component_size removed."""
    Gu_filtered = Gu.copy()
    for component in list(nx.connected_components(Gu_filtered)):
        if len(component) <= min_component_size:
            Gu_filtered.remove_nodes_from(component)

    Gd_filtered = Gd.copy()
    for component in list(nx.weakly_connected_components(Gd_filtered)):
        if len(component) <= min_component_size:
            Gd_filtered.remove_nodes_from(component)

    return Gu_filtered, Gd_filtered


def _select_top_communities(
    graph,
    node_to_community: dict,
    df_processed: pd.DataFrame,
    top_k: int,
    sort_by: str,
) -> set:
    """Return the set of top-k community IDs within `graph`, ranked by post volume or node count."""
    if sort_by == "post_volume":
        author_community = {
            handle: node_to_community[handle]
            for handle in df_processed['author_handle']
            if handle in node_to_community and handle in graph.nodes()
        }
        comm_ids = df_processed['author_handle'].map(author_community).dropna()
        return set(comm_ids.value_counts().head(top_k).index.tolist())

    filtered = {n: c for n, c in node_to_community.items() if n in graph.nodes()}
    return {cid for cid, _ in Counter(filtered.values()).most_common(top_k)}


def plot_filtered_network_graph(
    Gu: nx.Graph,
    Gd: nx.DiGraph,
    df_cent: pd.DataFrame,
    comm_data: dict,
    centralities: dict,
    df_processed: pd.DataFrame,
    min_component_size: int = 10,
    output_dir: str = "plots/filtered",
    top_k: int = 5,
    sort_by: str = "post_volume",
    cmap_undirected: str = "tab20",
    cmap_directed: str = "tab20",
) -> None:
    """Render network graphs restricted to large components and the top-k communities.

    Drops components of size <= min_component_size, keeps only the top-k Louvain
    (undirected) and Infomap (directed) communities, and reuses the full-graph spring
    layouts so node positions stay stable.
    """
    Gu_plot = Gu.copy()
    for component in list(nx.connected_components(Gu_plot)):
        if len(component) <= min_component_size:
            Gu_plot.remove_nodes_from(component)

    Gd_plot = Gd.copy()
    for component in list(nx.weakly_connected_components(Gd_plot)):
        if len(component) <= min_component_size:
            Gd_plot.remove_nodes_from(component)

    node_to_louvain = comm_data["node_to_louvain"]
    top_louvain = _select_top_communities(Gu_plot, node_to_louvain, df_processed, top_k, sort_by)
    Gu_plot.remove_nodes_from([n for n in list(Gu_plot.nodes()) if node_to_louvain.get(n) not in top_louvain])

    node_to_infomap = comm_data["node_to_infomap"]
    top_infomap = _select_top_communities(Gd_plot, node_to_infomap, df_processed, top_k, sort_by)
    Gd_plot.remove_nodes_from([n for n in list(Gd_plot.nodes()) if node_to_infomap.get(n) not in top_infomap])

    nodes_with_edges = [n for n, d in Gu.degree() if d > 0]
    pos = nx.spring_layout(Gu.subgraph(nodes_with_edges), k=0.3, iterations=60, seed=42)

    nodes_with_edges_dir = [n for n, d in Gd.degree() if d > 0]
    pos_dir = nx.spring_layout(Gd.subgraph(nodes_with_edges_dir), k=0.3, iterations=60, seed=42)

    plot_network_graphs(
        Gu_plot, Gd_plot, df_cent, comm_data, centralities,
        output_dir=output_dir, pos=pos, pos_dir=pos_dir,
        cmap_undirected=cmap_undirected, cmap_directed=cmap_directed,
    )


def plot_network_graphs(
    Gu: nx.Graph,
    Gd: nx.DiGraph,
    df_cent: pd.DataFrame,
    comm_data: dict,
    centralities: dict,
    output_dir: str = "plots",
    pos: Optional[dict] = None,
    pos_dir: Optional[dict] = None,
    cmap_undirected: str = "tab20",
    cmap_directed: str = "tab20",
) -> None:
    """Render the undirected (Louvain/degree) and directed (Infomap/PageRank) network figures.

    Node colour encodes community, node size encodes centrality, and only the top-10
    most central nodes are labelled. Precomputed layouts (pos/pos_dir) are reused as-is.
    """
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    node_to_community = comm_data["node_to_louvain"]
    modularity_score = comm_data["modularity_score"]

    nodes_in_relations = [n for n, d in Gu.degree() if d > 0]
    if not nodes_in_relations:
        print("Isolated graph / Not enough relationships to plot.")
        return

    subG = Gu.subgraph(nodes_in_relations)
    if pos is None:
        pos = nx.spring_layout(subG, k=0.3, iterations=60, seed=42)
        
    top_10_nodes = df_cent.sort_values(by="degree_centrality_undirected", ascending=False).head(10)['user'].tolist()
    labels_to_draw = {node: node for node in subG.nodes() if node in top_10_nodes}

    plt.figure(figsize=(12, 12))
    louvain_color_map = get_community_color_map(node_to_community, cmap_name=cmap_undirected)
    node_colors = [louvain_color_map.get(node_to_community.get(node, 0), "#bdc3c7") for node in subG.nodes()]
    node_sizes = [50 + (deg_cent[node] * 1200) for node in subG.nodes()]

    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Undirected Social Network Graph (degree centrality & Louvain partitions)\n(Modularity Q: {modularity_score:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("network_graph.png", output_dir=output_dir)
    plt.close()

    plt.figure(figsize=(12, 12))
    nodes_in_relations_dir = [n for n, d in Gd.degree() if d > 0]
    infomap_modularity = comm_data.get("infomap_modularity", 0.0)
    if nodes_in_relations_dir:
        subG_dir = Gd.subgraph(nodes_in_relations_dir)
        if pos_dir is None:
            pos_dir = nx.spring_layout(subG_dir, k=0.3, iterations=60, seed=42)

        node_to_infomap = comm_data.get("node_to_infomap", {})
        infomap_color_map = get_community_color_map(node_to_infomap, cmap_name=cmap_directed)
        node_colors_dir = [infomap_color_map.get(node_to_infomap.get(node, 0), "#bdc3c7") for node in subG_dir.nodes()]
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
