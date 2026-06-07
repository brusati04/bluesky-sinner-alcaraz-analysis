import os
from collections import Counter
from typing import Optional

import infomap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from utils import save_plot_copies

EMOTION_COLORS: dict[str, str] = {
    "anger":        "#e74c3c",
    "anticipation": "#e67e22",
    "disgust":      "#795548",
    "fear":         "#9b59b6",
    "joy":          "#f1c40f",
    "sadness":      "#3498db",
    "surprise":     "#1abc9c",
    "trust":        "#2ecc71",
    "neutral":      "#95a5a6",
}


def build_networks(df: pd.DataFrame, did_to_handle: dict) -> tuple[nx.Graph, nx.DiGraph]:

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
    if len(Gu.nodes()) <= 1:
        print("Graph too small for community detection / GCC calculation.")
        return {
            "communities": [list(Gu.nodes())], "modularity_score": 0.0, "node_to_community": {},
            "infomap_communities": [], "infomap_modularity": 0.0, "node_to_infomap": {},
            "gcc": Gu, "gcc_size": len(Gu.nodes()), "gcc_fraction": 1.0,
            "deg_assort_undir": 0.0, "deg_assort_dir": 0.0, "comm_assort_undir": 0.0,
        }

    communities = nx.community.louvain_communities(Gu)
    modularity_score = nx.community.modularity(Gu, communities)

    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    nx.set_node_attributes(Gu, node_to_community, "community")
    nx.set_node_attributes(Gd, node_to_community, "community")

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

    density_dir = nx.density(Gd)
    transitivity_dir = nx.transitivity(Gd)
    avg_clustering_dir = nx.average_clustering(Gd)

    wcc_dir = sorted(nx.weakly_connected_components(Gd), key=len, reverse=True)
    wcc_size_dir = len(wcc_dir[0]) if wcc_dir else 0
    wcc_fraction_dir = wcc_size_dir / len(Gd.nodes()) if Gd.nodes() else 0

    scc_dir = sorted(nx.strongly_connected_components(Gd), key=len, reverse=True)
    scc_size_dir = len(scc_dir[0]) if scc_dir else 0
    scc_fraction_dir = scc_size_dir / len(Gd.nodes()) if Gd.nodes() else 0

    gcc_avg_path_length = gcc_diameter = gcc_radius = min_ecc = max_ecc = 0.0
    if gcc_size > 1:
        try: gcc_avg_path_length = nx.average_shortest_path_length(gcc)
        except Exception: gcc_avg_path_length = 0.0
        try:
            gcc_ecc = nx.eccentricity(gcc)
            gcc_diameter = nx.diameter(gcc, gcc_ecc)
            gcc_radius = nx.radius(gcc, gcc_ecc)
            min_ecc = min(gcc_ecc.values())
            max_ecc = max(gcc_ecc.values())
        except Exception: pass

    print("\n" + "=" * 50)
    print("GLOBAL SOCIAL NETWORK STATISTICS")
    print("=" * 50)
    print(f"Network Density (Undir/Dir):        {density:.6f} / {density_dir:.6f}")
    print(f"Louvain Communities (Undir):        {len(communities)} (Modularity Q: {modularity_score:.4f})")
    print(f"Infomap Communities (Dir):          {len(infomap_communities)} (Modularity Q: {infomap_modularity:.4f})")
    print("=" * 50)

    try:
        os.makedirs("data", exist_ok=True)
        stats_df = pd.DataFrame({
            "Metric": [
                "Density", 
                "Transitivity", 
                "Average Clustering", 
                "WCC Size", 
                "WCC Fraction", 
                "SCC Size", 
                "SCC Fraction", 
                "Louvain Modularity", 
                "Infomap Modularity"
            ],
            "Undirected": [
                density, 
                transitivity, 
                avg_clustering, 
                gcc_size, 
                gcc_fraction, 
                None, 
                None, 
                modularity_score, 
                None
            ],
            "Directed": [
                density_dir, 
                transitivity_dir, 
                avg_clustering_dir, 
                wcc_size_dir, 
                wcc_fraction_dir, 
                scc_size_dir, 
                scc_fraction_dir, 
                None, 
                infomap_modularity
            ]
        })
        stats_df.to_csv("data/network_global_metrics.csv", index=False)
    except Exception as e:
        print(f"Error saving global metrics: {e}")

    try: deg_assort_undir = nx.degree_assortativity_coefficient(Gu)
    except Exception: deg_assort_undir = 0.0
    try: deg_assort_dir = nx.degree_assortativity_coefficient(Gd)
    except Exception: deg_assort_dir = 0.0
    try: comm_assort_undir = nx.attribute_assortativity_coefficient(Gu, "community")
    except Exception: comm_assort_undir = 0.0

    return {
        "louvain_communities": communities, "modularity_score": modularity_score, "node_to_louvain": node_to_community,
        "infomap_communities": infomap_communities, "infomap_modularity": infomap_modularity, "node_to_infomap": node_to_infomap,
        "gcc": gcc, "gcc_size": gcc_size, "gcc_fraction": gcc_fraction, "deg_assort_undir": deg_assort_undir,
        "deg_assort_dir": deg_assort_dir, "comm_assort_undir": comm_assort_undir,
    }


def save_initial_centrality_csv(
    Gu: nx.Graph, centralities: dict, comm_data: dict, filepath: str = "data/network_centrality_metrics.csv"
) -> pd.DataFrame:

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
    unique_cids = sorted(set(node_to_community.values()))
    n = len(unique_cids)
    try: cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError: cmap = plt.cm.get_cmap(cmap_name)

    is_qualitative = cmap_name.lower().startswith(('pastel', 'paired', 'accent', 'dark2', 'set', 'tab'))
    color_map = {}
    for idx, cid in enumerate(unique_cids):
        rgba = cmap(idx % cmap.N) if is_qualitative else cmap(idx / (n - 1) if n > 1 else 0.5)
        color_map[cid] = mcolors.to_hex(rgba)
    return color_map

def _select_top_communities(graph, node_to_community: dict, df_processed: Optional[pd.DataFrame], top_k: int, sort_by: str) -> set:
    if sort_by == "post_volume" and df_processed is not None:
        author_community = {
            handle: node_to_community[handle]
            for handle in df_processed['author_handle']
            if handle in node_to_community and handle in graph.nodes()
        }
        comm_ids = df_processed['author_handle'].map(author_community).dropna()
        return set(comm_ids.value_counts().head(top_k).index.tolist())

    filtered = {n: c for n, c in node_to_community.items() if n in graph.nodes()}
    return {cid for cid, _ in Counter(filtered.values()).most_common(top_k)}


def extract_top_subgraph(
    G: nx.Graph,
    min_component_size: int = 10,
    top_k_communities: Optional[int] = None,
    node_to_community: Optional[dict] = None,
    df_processed: Optional[pd.DataFrame] = None,
    sort_by: str = "post_volume"
) -> nx.Graph:
    G_filtered = G.copy()
    
    components = list(nx.weakly_connected_components(G_filtered)) if G_filtered.is_directed() else list(nx.connected_components(G_filtered))
    for component in components:
        if len(component) <= min_component_size:
            G_filtered.remove_nodes_from(component)
            
    if top_k_communities and node_to_community:
        top_comms = _select_top_communities(G_filtered, node_to_community, df_processed, top_k_communities, sort_by)
        nodes_to_remove = [n for n in G_filtered.nodes() if node_to_community.get(n) not in top_comms]
        G_filtered.remove_nodes_from(nodes_to_remove)
        
    return G_filtered


def render_network(
    G,
    pos: dict,
    node_colors: list,
    node_sizes: list,
    labels: dict,
    title: str,
    filename: str,
    output_dir: str = "plots",
    legend_patches: Optional[list] = None,
    legend_title: Optional[str] = None,
    legend_ncol: int = 2
) -> None:
    if G.number_of_nodes() == 0:
        print(f"Warning: Grafo vuoto per {filename}. Plot saltato.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    if G.is_directed():
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="grey",
                               arrows=True, arrowstyle='-|>', arrowsize=12, connectionstyle="arc3,rad=0.1")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="grey")
        
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, 
                           alpha=0.9, edgecolors='black', linewidths=0.5)
    
    if labels:
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=9, font_weight="bold", font_color="#1e272c")
        
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left", fontsize=10, 
                  title=legend_title, title_fontsize=11, framealpha=0.85, ncol=legend_ncol)

    ax.set_title(title, pad=15)
    ax.axis("off")
    plt.tight_layout()
    save_plot_copies(filename, output_dir=output_dir)
    plt.close(fig)


def get_filtered_networks(Gu: nx.Graph, Gd: nx.DiGraph, min_component_size: int = 10) -> tuple[nx.Graph, nx.DiGraph]:
    return extract_top_subgraph(Gu, min_component_size), extract_top_subgraph(Gd, min_component_size)


def plot_network_graphs(
    Gu: nx.Graph, Gd: nx.DiGraph, df_cent: pd.DataFrame, comm_data: dict, centralities: dict,
    output_dir: str = "plots", pos: Optional[dict] = None, pos_dir: Optional[dict] = None,
    cmap_undirected: str = "tab20", cmap_directed: str = "tab20",
) -> tuple[Optional[dict], Optional[dict]]:
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    node_to_louvain = comm_data["node_to_louvain"]
    node_to_infomap = comm_data.get("node_to_infomap", {})

    nodes_in_relations = [n for n, d in Gu.degree() if d > 0]
    subG = Gu.subgraph(nodes_in_relations)
    if pos is None and nodes_in_relations:
        pos = nx.spring_layout(subG, k=0.3, iterations=60, seed=42)

    if nodes_in_relations:
        louvain_cmap = get_community_color_map(node_to_louvain, cmap_name=cmap_undirected)
        colors = [louvain_cmap.get(node_to_louvain.get(node, 0), "#bdc3c7") for node in subG.nodes()]
        sizes = [50 + (deg_cent[node] * 1200) for node in subG.nodes()]
        top_10 = df_cent.sort_values(by="degree_centrality_undirected", ascending=False).head(10)['user'].tolist()
        labels = {n: n for n in subG.nodes() if n in top_10}
        
        render_network(
            G=subG, pos=pos, node_colors=colors, node_sizes=sizes, labels=labels,
            title=f"Undirected Social Network Graph (degree centrality & Louvain partitions)\n(Modularity Q: {comm_data['modularity_score']:.4f})",
            filename="network_graph.png", output_dir=output_dir
        )


    nodes_in_relations_dir = [n for n, d in Gd.degree() if d > 0]
    subG_dir = Gd.subgraph(nodes_in_relations_dir)
    if pos_dir is None and nodes_in_relations_dir:
        pos_dir = nx.spring_layout(subG_dir, k=0.3, iterations=60, seed=42)

    if nodes_in_relations_dir:
        infomap_cmap = get_community_color_map(node_to_infomap, cmap_name=cmap_directed)
        colors_dir = [infomap_cmap.get(node_to_infomap.get(node, 0), "#bdc3c7") for node in subG_dir.nodes()]
        sizes_dir = [50 + (pagerank.get(node, 0.0) * 18000) for node in subG_dir.nodes()]
        top_10_dir = df_cent.sort_values(by="pagerank", ascending=False).head(10)['user'].tolist()
        labels_dir = {n: n for n in subG_dir.nodes() if n in top_10_dir}

        render_network(
            G=subG_dir, pos=pos_dir, node_colors=colors_dir, node_sizes=sizes_dir, labels=labels_dir,
            title=f"Directed Social Network Graph (PageRank prestige & Infomap partitions)\n(Projected Modularity Q: {comm_data.get('infomap_modularity', 0.0):.4f})",
            filename="network_graph_directed.png", output_dir=output_dir
        )

    return pos, pos_dir


def plot_filtered_network_graph(
    Gu: nx.Graph, Gd: nx.DiGraph, df_cent: pd.DataFrame, comm_data: dict, centralities: dict, df_processed: pd.DataFrame,
    min_component_size: int = 10, output_dir: str = "plots/filtered", top_k: int = 5, sort_by: str = "post_volume",
    cmap_undirected: str = "tab20", cmap_directed: str = "tab20",
) -> tuple[nx.Graph, nx.DiGraph, dict, dict]:
    Gu_plot = extract_top_subgraph(Gu, min_component_size, top_k, comm_data["node_to_louvain"], df_processed, sort_by)
    Gd_plot = extract_top_subgraph(Gd, min_component_size, top_k, comm_data["node_to_infomap"], df_processed, sort_by)

    pos = nx.spring_layout(Gu.subgraph([n for n, d in Gu.degree() if d > 0]), k=0.3, iterations=60, seed=42)
    pos_dir = nx.spring_layout(Gd.subgraph([n for n, d in Gd.degree() if d > 0]), k=0.3, iterations=60, seed=42)

    plot_network_graphs(Gu_plot, Gd_plot, df_cent, comm_data, centralities, output_dir=output_dir, pos=pos, pos_dir=pos_dir,
                        cmap_undirected=cmap_undirected, cmap_directed=cmap_directed)

    return Gu_plot, Gd_plot, pos, pos_dir


def _get_user_dominant_emotion(df_processed: pd.DataFrame, backend: str) -> dict[str, str]:
    col = "author_dominant_emotion_nrc" if backend == "nrc" else "author_dominant_emotion_bert"
    return df_processed.groupby("author_handle")[col].first().to_dict()



def plot_network_graphs_by_emotion(
    Gu: nx.Graph, Gd: nx.DiGraph, df_cent: pd.DataFrame, centralities: dict, df_processed: pd.DataFrame,
    backend: str = "nrc", output_dir: str = "plots", pos: Optional[dict] = None, pos_dir: Optional[dict] = None,
) -> None:
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    user_emotion = _get_user_dominant_emotion(df_processed, backend)
    
    legend_patches = [mpatches.Patch(color=color, label=emotion.capitalize()) for emotion, color in EMOTION_COLORS.items()]

    nodes = [n for n, d in Gu.degree() if d > 0]
    if nodes:
        subG = Gu.subgraph(nodes)
        if pos is None: pos = nx.spring_layout(subG, k=0.3, iterations=60, seed=42)
        colors = [EMOTION_COLORS.get(user_emotion.get(n, "neutral"), EMOTION_COLORS["neutral"]) for n in subG.nodes()]
        sizes = [50 + (deg_cent[n] * 1200) for n in subG.nodes()]
        top_10 = df_cent.sort_values(by="degree_centrality_undirected", ascending=False).head(10)['user'].tolist()
        
        render_network(subG, pos, colors, sizes, {n: n for n in subG.nodes() if n in top_10},
                       f"Undirected Social Network Graph — Dominant Emotion ({backend.upper()})\n(node size proportional to degree centrality)",
                       f"network_graph_emotion_{backend}.png", output_dir, legend_patches, "Dominant Emotion")

    nodes_dir = [n for n, d in Gd.degree() if d > 0]
    if nodes_dir:
        subG_dir = Gd.subgraph(nodes_dir)
        if pos_dir is None: pos_dir = nx.spring_layout(subG_dir, k=0.3, iterations=60, seed=42)
        colors_dir = [EMOTION_COLORS.get(user_emotion.get(n, "neutral"), EMOTION_COLORS["neutral"]) for n in subG_dir.nodes()]
        sizes_dir = [50 + (pagerank.get(n, 0.0) * 18000) for n in subG_dir.nodes()]
        top_10_dir = df_cent.sort_values(by="pagerank", ascending=False).head(10)['user'].tolist()
        
        render_network(subG_dir, pos_dir, colors_dir, sizes_dir, {n: n for n in subG_dir.nodes() if n in top_10_dir},
                       f"Directed Social Network Graph — Dominant Emotion ({backend.upper()})\n(node size proportional to PageRank prestige)",
                       f"network_graph_directed_emotion_{backend}.png", output_dir, legend_patches, "Dominant Emotion")


def plot_network_graphs_by_fanbase(
    Gu: nx.Graph, Gd: nx.DiGraph, df_cent: pd.DataFrame, centralities: dict,
    output_dir: str = "plots", pos: Optional[dict] = None, pos_dir: Optional[dict] = None,
) -> None:
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    FANBASE_COLORS = {'sinner': '#f39c12', 'alcaraz': '#00b4d8', 'neutral': '#cbd5e0'}
    stance_map = dict(zip(df_cent['user'], df_cent['stance_leaning']))
    
    get_color = lambda n: FANBASE_COLORS.get(stance_map.get(n, 'neutral') if pd.notna(stance_map.get(n, 'neutral')) else 'neutral', FANBASE_COLORS['neutral'])
    legend_patches = [mpatches.Patch(color='#f39c12', label='Sinner Fanbase'), mpatches.Patch(color='#00b4d8', label='Alcaraz Fanbase'), mpatches.Patch(color='#cbd5e0', label='Neutral')]

    nodes = [n for n, d in Gu.degree() if d > 0]
    if nodes:
        subG = Gu.subgraph(nodes)
        if pos is None: pos = nx.spring_layout(subG, k=0.3, iterations=60, seed=42)
        top_10 = df_cent.sort_values(by="degree_centrality_undirected", ascending=False).head(10)['user'].tolist()
        
        render_network(subG, pos, [get_color(n) for n in subG.nodes()], [50 + (deg_cent[n] * 1200) for n in subG.nodes()],
                       {n: n for n in subG.nodes() if n in top_10}, "Undirected Social Network Graph — Fanbase Leaning\n(node size proportional to degree centrality)",
                       "network_graph_fanbase.png", output_dir, legend_patches, "Fanbase Leaning", legend_ncol=1)

    # Directed
    nodes_dir = [n for n, d in Gd.degree() if d > 0]
    if nodes_dir:
        subG_dir = Gd.subgraph(nodes_dir)
        if pos_dir is None: pos_dir = nx.spring_layout(subG_dir, k=0.3, iterations=60, seed=42)
        top_10_dir = df_cent.sort_values(by="pagerank", ascending=False).head(10)['user'].tolist()
        
        render_network(subG_dir, pos_dir, [get_color(n) for n in subG_dir.nodes()], [50 + (pagerank.get(n, 0.0) * 18000) for n in subG_dir.nodes()],
                       {n: n for n in subG_dir.nodes() if n in top_10_dir}, "Directed Social Network Graph — Fanbase Leaning\n(node size proportional to PageRank prestige)",
                       "network_graph_directed_fanbase.png", output_dir, legend_patches, "Fanbase Leaning", legend_ncol=1)