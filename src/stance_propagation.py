import pandas as pd
import networkx as nx
import ast

def run_stance_propagation(df, G, Gd, df_cent, filepath="data/network_centrality_metrics.csv"):

    
    # 1. Compute post-level stance
    post_stances = []
    for idx, row in df.iterrows():
        linked_ents = row['linked_entities']
        comp = row['sentiment_compound']
        
        # Safely extract URIs
        if isinstance(linked_ents, str):
            try:
                ents = ast.literal_eval(linked_ents)
            except Exception:
                ents = []
        else:
            ents = linked_ents if isinstance(linked_ents, list) else []
            
        uris = []
        for ent in ents:
            if isinstance(ent, dict):
                uris.append(ent.get('uri'))
            elif isinstance(ent, tuple) or isinstance(ent, list):
                uris.append(ent[0])
                
        is_sinner = "http://dbpedia.org/resource/Jannik_Sinner" in uris
        is_alcaraz = "http://dbpedia.org/resource/Carlos_Alcaraz" in uris
        
        t = str(row['text']).lower()
        if not is_sinner and ("sinner" in t or "jannik" in t):
            is_sinner = True
        if not is_alcaraz and ("alcaraz" in t or "carlos" in t):
            is_alcaraz = True
            
        # Post-level stance: sentiment * (I_Sinner - I_Alcaraz)
        s_val = 0.0
        if is_sinner and not is_alcaraz:
            s_val = float(comp)
        elif is_alcaraz and not is_sinner:
            s_val = -float(comp)
        post_stances.append(s_val)
        
    df['post_stance'] = post_stances
    
    # 2. Compute user-level initial stances
    user_initial_stances = df.groupby('author_handle')['post_stance'].mean().to_dict()
    
    # 3. Propagate stances using Laplacian/Label Spreading iterative method
    initial_stances_in_G = {node: user_initial_stances.get(node, 0.0) for node in G.nodes()}
    
    alpha_prop = 0.15
    max_iter_prop = 100
    tol_prop = 1e-5
    
    f_prop = initial_stances_in_G.copy()
    f_0_prop = f_prop.copy()
    
    for iteration in range(max_iter_prop):
        f_new_prop = {}
        diff_prop = 0.0
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                f_new_prop[node] = f_0_prop[node]
                continue
            
            # Weighted average of neighbors
            total_weight = sum(G[node][nbr].get('weight', 1) for nbr in neighbors)
            neighbor_sum = sum(G[node][nbr].get('weight', 1) * f_prop[nbr] for nbr in neighbors)
            
            propagated = neighbor_sum / total_weight
            f_new_prop[node] = alpha_prop * f_0_prop[node] + (1.0 - alpha_prop) * propagated
            diff_prop += abs(f_new_prop[node] - f_prop[node])
            
        f_prop = f_new_prop
        if diff_prop < tol_prop:
            break
            
    # Save propagated stances back to node attributes in NetworkX
    nx.set_node_attributes(G, f_prop, "stance_score")
    nx.set_node_attributes(Gd, f_prop, "stance_score")
    
    # Classify leanings
    stance_leanings = {}
    for node, score in f_prop.items():
        if score > 0.05:
            stance_leanings[node] = "sinner"
        elif score < -0.05:
            stance_leanings[node] = "alcaraz"
        else:
            stance_leanings[node] = "neutral"
            
    nx.set_node_attributes(G, stance_leanings, "stance_leaning")
    nx.set_node_attributes(Gd, stance_leanings, "stance_leaning")
    
    # Add to centrality dataframe and re-save centrality CSV
    df_cent['stance_score'] = df_cent['user'].map(f_prop)
    df_cent['stance_leaning'] = df_cent['user'].map(stance_leanings)
    df_cent.to_csv(filepath, index=False)

    
    # Print stats and compute assortativity
    sinner_count = sum(1 for l in stance_leanings.values() if l == "sinner")
    alcaraz_count = sum(1 for l in stance_leanings.values() if l == "alcaraz")
    neutral_count = sum(1 for l in stance_leanings.values() if l == "neutral")

    
    try:
        stance_assort = nx.attribute_assortativity_coefficient(G, "stance_leaning")

    except Exception as e:
        stance_assort = 0.0
        print("Error calculating stance assortativity:", e)

    return {
        "df_cent": df_cent,
        "stance_leanings": stance_leanings,
        "stance_assort": stance_assort
    }
