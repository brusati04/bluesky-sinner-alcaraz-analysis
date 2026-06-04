import pandas as pd
import networkx as nx

def build_networks(df, did_to_handle):
    print("\n--- PHASE 2: Graph Modelling & SNA (Undirected & Directed) ---")
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

    print(f"Undirected Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Directed Network: {Gd.number_of_nodes()} nodes, {Gd.number_of_edges()} edges.")
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
            "leiden_communities": [],
            "leiden_modularity": 0.0,
            "node_to_leiden": {},
            "infomap_communities": [],
            "infomap_modularity": 0.0,
            "node_to_infomap": {},
            "lpa_communities": [],
            "lpa_modularity": 0.0,
            "node_to_lpa": {},
            "fluid_communities": [],
            "fluid_modularity": 0.0,
            "node_to_fluid": {},
            "k_fluid": 3,
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
    print(f"Louvain Community Detection (on G) found: {len(communities)} communities.")
    print(f"Graph Modularity (Q): {modularity_score:.4f}")

    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    nx.set_node_attributes(G, node_to_community, "community")
    nx.set_node_attributes(Gd, node_to_community, "community")

    # 1. Leiden Algorithm (Undirected G)
    print("\n--- Running Leiden Algorithm ---")
    try:
        import igraph as ig
        import leidenalg
        
        g_nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(g_nodes)}
        
        ig_G = ig.Graph(directed=False)
        ig_G.add_vertices(len(g_nodes))
        
        edges_ig = []
        weights_ig = []
        for u, v, d in G.edges(data=True):
            edges_ig.append((node_to_idx[u], node_to_idx[v]))
            weights_ig.append(d.get('weight', 1.0))
        ig_G.add_edges(edges_ig)
        ig_G.es['weight'] = weights_ig
        
        leiden_partition = leidenalg.find_partition(
            ig_G, leidenalg.ModularityVertexPartition, 
            weights='weight', seed=42
        )
        
        leiden_communities_list = [[] for _ in range(len(leiden_partition))]
        node_to_leiden = {}
        for comm_id, member_indices in enumerate(leiden_partition):
            for idx in member_indices:
                node = g_nodes[idx]
                leiden_communities_list[comm_id].append(node)
                node_to_leiden[node] = comm_id
        
        leiden_communities = [set(c) for c in leiden_communities_list]
        leiden_modularity = nx.community.modularity(G, leiden_communities)
        print(f"Leiden Community Detection found: {len(leiden_communities)} communities.")
        print(f"Leiden Graph Modularity (Q): {leiden_modularity:.4f}")
        nx.set_node_attributes(G, node_to_leiden, "leiden_community")
    except Exception as e:
        leiden_communities = []
        leiden_modularity = 0.0
        node_to_leiden = {}
        print(f"Error running Leiden: {e}")

    # 2. Infomap Algorithm (Directed Gd)
    print("\n--- Running Infomap Algorithm ---")
    try:
        from infomap import Infomap
        
        im = Infomap("--two-level --directed --silent")
        gd_nodes = list(Gd.nodes())
        node_to_idx_d = {node: idx for idx, node in enumerate(gd_nodes)}
        
        for u, v, d in Gd.edges(data=True):
            im.add_link(node_to_idx_d[u], node_to_idx_d[v], d.get('weight', 1.0))
            
        im.run()
        
        node_to_infomap = {}
        infomap_comm_map = {}
        for node_idx, module_id in im.modules:
            node = gd_nodes[node_idx]
            node_to_infomap[node] = module_id
            if module_id not in infomap_comm_map:
                infomap_comm_map[module_id] = []
            infomap_comm_map[module_id].append(node)
            
        infomap_communities = [set(c) for c in infomap_comm_map.values()]
        infomap_modularity = nx.community.modularity(G, infomap_communities)
        print(f"Infomap found: {len(infomap_communities)} communities.")
        print(f"Infomap Graph Modularity (Q, evaluated on G): {infomap_modularity:.4f}")
        nx.set_node_attributes(G, node_to_infomap, "infomap_community")
    except Exception as e:
        infomap_communities = []
        infomap_modularity = 0.0
        node_to_infomap = {}
        print(f"Error running Infomap: {e}")

    # 3. Label Propagation Algorithm (LPA on G)
    print("\n--- Running Label Propagation Algorithm ---")
    try:
        lpa_communities = list(nx.community.label_propagation_communities(G))
        lpa_modularity = nx.community.modularity(G, lpa_communities)
        
        node_to_lpa = {}
        for comm_id, comm in enumerate(lpa_communities):
            for node in comm:
                node_to_lpa[node] = comm_id
        print(f"Label Propagation found: {len(lpa_communities)} communities.")
        print(f"Label Propagation Modularity (Q): {lpa_modularity:.4f}")
        nx.set_node_attributes(G, node_to_lpa, "lpa_community")
    except Exception as e:
        lpa_communities = []
        lpa_modularity = 0.0
        node_to_lpa = {}
        print(f"Error running LPA: {e}")

    # Compute Giant Connected Component (GCC) on G
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc = G.subgraph(components[0])
    gcc_size = gcc.number_of_nodes()
    gcc_fraction = gcc_size / G.number_of_nodes()
    print(f"\n--- Giant Connected Component (GCC) Analysis ---")
    print(f"GCC size: {gcc_size} nodes ({gcc_fraction*100:.2f}% of N={G.number_of_nodes()})")
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Mean degree <k>: {avg_degree:.3f}")
    print(f"Phase transition threshold (Erdos-Renyi theory): <k> ~ 1.0")
    if avg_degree < 1.0:
        print("  -> Network is BELOW the critical threshold: many isolated components expected.")
    elif avg_degree < 2.0:
        print(f"  -> Network is ABOVE threshold (<k>={avg_degree:.3f} > 1.0): GCC emergence is expected.")
        print(f"  -> GCC of {gcc_fraction*100:.1f}% is consistent with early post-transition behaviour.")
    else:
        print(f"  -> Network is well connected (<k>={avg_degree:.3f}): GCC dominance expected.")

    # 4. Fluid Communities (on GCC)
    print("\n--- Running Fluid Communities on GCC ---")
    k_fluid = 3
    try:
        fluid_communities_list = list(nx.community.asyn_fluidc(gcc, k_fluid, seed=42))
        fluid_communities = [set(c) for c in fluid_communities_list]
        fluid_modularity = nx.community.modularity(gcc, fluid_communities)
        
        node_to_fluid = {}
        for comm_id, comm in enumerate(fluid_communities):
            for node in comm:
                node_to_fluid[node] = comm_id
        print(f"Fluid Communities (on GCC, k={k_fluid}) found: {len(fluid_communities)} communities.")
        print(f"Fluid Communities GCC Modularity (Q): {fluid_modularity:.4f}")
        nx.set_node_attributes(gcc, node_to_fluid, "fluid_community")
    except Exception as e:
        fluid_communities = []
        fluid_modularity = 0.0
        node_to_fluid = {}
        print(f"Error running Fluid Communities on GCC: {e}")

    # Assortativity Analysis
    print(f"\n--- Assortativity Analysis ---")
    try:
        deg_assort_undir = nx.degree_assortativity_coefficient(G)
        print(f"Undirected Graph degree assortativity (r): {deg_assort_undir:.4f}")
    except Exception as e:
        deg_assort_undir = 0.0
        print("Error calculating undirected degree assortativity:", e)

    try:
        deg_assort_dir = nx.degree_assortativity_coefficient(Gd)
        print(f"Directed Graph degree assortativity (r): {deg_assort_dir:.4f}")
    except Exception as e:
        deg_assort_dir = 0.0
        print("Error calculating directed degree assortativity:", e)

    try:
        comm_assort_undir = nx.attribute_assortativity_coefficient(G, "community")
        print(f"Undirected Graph community assortativity: {comm_assort_undir:.4f}")
    except Exception as e:
        comm_assort_undir = 0.0
        print("Error calculating undirected community assortativity:", e)

    return {
        "communities": communities,
        "modularity_score": modularity_score,
        "node_to_community": node_to_community,
        "leiden_communities": leiden_communities,
        "leiden_modularity": leiden_modularity,
        "node_to_leiden": node_to_leiden,
        "infomap_communities": infomap_communities,
        "infomap_modularity": infomap_modularity,
        "node_to_infomap": node_to_infomap,
        "lpa_communities": lpa_communities,
        "lpa_modularity": lpa_modularity,
        "node_to_lpa": node_to_lpa,
        "fluid_communities": fluid_communities,
        "fluid_modularity": fluid_modularity,
        "node_to_fluid": node_to_fluid,
        "k_fluid": k_fluid,
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
    node_to_leiden = comm_data["node_to_leiden"]
    node_to_infomap = comm_data["node_to_infomap"]
    node_to_lpa = comm_data["node_to_lpa"]
    node_to_fluid = comm_data["node_to_fluid"]
    
    for node in G.nodes():
        centrality_data.append({
            "user": node,
            "community": node_to_community.get(node, 0) if len(G.nodes()) > 1 else 0,
            "leiden_community": node_to_leiden.get(node, -1),
            "infomap_community": node_to_infomap.get(node, -1),
            "lpa_community": node_to_lpa.get(node, -1),
            "fluid_community_gcc": node_to_fluid.get(node, -1),
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
    print(f"Combined SNA metrics saved to {filepath}")
    return df_cent
