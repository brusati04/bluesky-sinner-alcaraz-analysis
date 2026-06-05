import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import ast
import re
import sys
from collections import Counter

# Add src/ to import path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils import parse_list_col, build_did_to_handle
from preprocessing import clean_text, preprocess

def extract_top_words(texts, top_n=10):
    words = []
    for text in texts:
        if pd.notna(text):
            words.extend(str(text).split())
    # Filter out empty or extremely short words
    words = [w for w in words if len(w) > 2]
    counter = Counter(words)
    return [word for word, count in counter.most_common(top_n)]

def parse_entities(entities_col):
    parsed = []
    for val in entities_col:
        if pd.isna(val) or val == "":
            continue
        try:
            # entities_col can be a string-encoded list of tuples/dicts
            ents = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(ents, list):
                for ent in ents:
                    if isinstance(ent, dict):
                        parsed.append(ent.get("surface_form") or ent.get("name"))
                    elif isinstance(ent, tuple) or isinstance(ent, list):
                        parsed.append(ent[0])
                    else:
                        parsed.append(str(ent))
        except Exception:
            pass
    return parsed

def profile_communities(df, cent_df, community_column, emotion_cols):
    """
    Profile each community in a specific community detection algorithm column.
    """
    # Exclude -1 (unassigned / not in GCC for fluid communities)
    unique_comms = sorted([int(c) for c in cent_df[community_column].dropna().unique() if int(c) != -1])
    
    communities = {}
    for comm_id in unique_comms:
        comm_users = cent_df[cent_df[community_column] == comm_id]['user'].tolist()
        comm_posts = df[df['author_handle'].isin(comm_users)].copy()
        
        # Sort posts by engagement & recency
        comm_posts['engagement'] = comm_posts['like_count'] + comm_posts['repost_count'] + comm_posts['reply_count']
        comm_posts = comm_posts.sort_values(by=['engagement', 'created_at'], ascending=[False, False])
        
        post_count = len(comm_posts)
        if post_count == 0:
            avg_sentiment = 0.0
            sentiment_cat = "neutral"
            top_words = []
            top_ents = []
            avg_emotions = {}
            sample_posts = []
        else:
            avg_sentiment = float(comm_posts['sentiment_compound'].mean())
            if avg_sentiment >= 0.05:
                sentiment_cat = "positive"
            elif avg_sentiment <= -0.05:
                sentiment_cat = "negative"
            else:
                sentiment_cat = "neutral"
                
            top_words = extract_top_words(comm_posts['preprocessed_text'], 10)
            
            # Entities
            ents_list = parse_entities(comm_posts['linked_entities'])
            if not ents_list:
                ents_list = parse_entities(comm_posts['entities'])
            top_ents = [ent for ent, count in Counter(ents_list).most_common(5)]
            
            # Emotions average
            avg_emotions = {}
            for col in emotion_cols:
                emotion_name = col.replace('emotion_', '')
                avg_emotions[emotion_name] = float(comm_posts[col].mean())
                
            # Limit posts in JSON to keep it compact
            sample_posts = []
            for _, p in comm_posts.head(100).iterrows():
                sample_posts.append({
                    "author": p['author_handle'],
                    "text": p['text'],
                    "created_at": p['created_at'].isoformat() if isinstance(p['created_at'], pd.Timestamp) else str(p['created_at']),
                    "sentiment": float(p['sentiment_compound']),
                    "sentiment_category": p['sentiment_category'],
                    "dominant_emotion": p['dominant_emotion'],
                    "likes": int(p['like_count']),
                    "reposts": int(p['repost_count'])
                })
                
        # Heuristic Taxonomy Classification
        all_text = " ".join(comm_posts['text'].dropna()).lower()
        links_count = sum(len(links) for links in comm_posts['links'].dropna())
        links_ratio = links_count / post_count if post_count > 0 else 0
        
        bot_keywords = ["watch", "live", "coverage", "stream", "feed", "broadcast", "score", "notizie", "cronaca", "berita", "t.co", "read more"]
        analytical_keywords = ["stats", "average", "ranking", "history", "analytics", "match-up", "points", "percentage", "sets", "data", "analysis", "tactical", "preview", "review", "rank", "draw"]
        hype_keywords = ["vamos", "goat", "incredible", "win", "champion", "king", "queen", "haha", "wow", "love", "insane", "beast", "come on", "vamosss", "bellissimo", "grandioso", "forza", "siuuu"]
        
        bot_matches = sum(1 for kw in bot_keywords if kw in all_text)
        analytical_matches = sum(1 for kw in analytical_keywords if kw in all_text)
        hype_matches = sum(1 for kw in hype_keywords if kw in all_text)
        
        # Check for automated bot indicators
        is_bot = "icdb.tv" in all_text or any(u in ["parliamodinews.bsky.social"] for u in comm_users) or links_ratio > 0.8
        
        if is_bot or (bot_matches > analytical_matches and bot_matches > hype_matches and links_ratio > 0.5):
            category = "utility/bot"
            description = "Automated feeds, broadcast link aggregators, and match trackers."
        elif analytical_matches > bot_matches and analytical_matches > hype_matches:
            category = "analytical/journalism"
            description = "Sports journalism, statistical analysis, and tactical match discussions."
        elif hype_matches > bot_matches or (hype_matches > 0 and post_count > 0):
            category = "hype/meme"
            description = "Fan celebrations, reactions, player support, memes, and emoji-heavy hype."
        else:
            category = "fan_chat"
            description = "General social interactions and organic discussions about the US Open tournament."
            
        top_users_in_comm = cent_df[cent_df[community_column] == comm_id].sort_values(by='pagerank', ascending=False).head(5)['user'].tolist()
        
        communities[str(comm_id)] = {
            "id": comm_id,
            "size": len(comm_users),
            "post_count": post_count,
            "avg_sentiment": avg_sentiment,
            "sentiment_category": sentiment_cat,
            "category": category,
            "description": description,
            "top_words": top_words,
            "top_entities": top_ents,
            "emotions": avg_emotions,
            "top_users": top_users_in_comm,
            "posts": sample_posts
        }
        
    return communities

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    df = pd.read_csv(os.path.join(base_dir, "data", "sinner_alcaraz_processed.csv"))
    cent_df = pd.read_csv(os.path.join(base_dir, "data", "network_centrality_metrics.csv"))
    
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    df = df.dropna(subset=['created_at'])
    for col in ['hashtags', 'mentions', 'links']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_col)
    
    did_to_handle = build_did_to_handle(df)
    
    # Reconstruct Directed Graph Gd to get interactions
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src_did = row['author_did']
        if pd.isna(src_did):
            continue
        source = did_to_handle.get(src_did, src_did)
        
        # A. Reply
        parent_did = row['reply_parent_did']
        if pd.notna(parent_did):
            target = did_to_handle.get(parent_did, parent_did)
            if source != target:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1, relationship="REPLY")
                    
        # B. Mentions
        for m in row['mentions']:
            m_did = m.get('did') if isinstance(m, dict) else m
            if m_did:
                target = did_to_handle.get(m_did, m_did)
                if source != target:
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1, relationship="MENTION")
                        
    node_metrics = cent_df.set_index('user').to_dict(orient='index')
    
    # Calculate user level averages
    user_post_counts = df['author_handle'].value_counts().to_dict()
    user_sentiments = df.groupby('author_handle')['sentiment_compound'].mean().to_dict()
    
    # Prepare JSON Nodes
    json_nodes = []
    for node in G.nodes():
        metrics = node_metrics.get(node, {})
        
        pagerank = float(metrics.get('pagerank', 0.0))
        in_degree = float(metrics.get('in_degree_centrality_directed', 0.0))
        out_degree = float(metrics.get('out_degree_centrality_directed', 0.0))
        closeness = float(metrics.get('closeness_centrality_directed', 0.0))
        betweenness = float(metrics.get('betweenness_centrality_directed', 0.0))
        
        # Maintain backwards compatibility for single community attribute (Louvain)
        comm = int(metrics.get('community', -1))
        comm_infomap = int(metrics.get('community_infomap', -1))
        
        json_nodes.append({
            "id": node,
            "label": node.split('.')[0] if '.' in node else node, # short label
            "community": comm,
            "communities": {
                "louvain": comm,
                "infomap": comm_infomap
            },
            "pagerank": pagerank,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "closeness": closeness,
            "betweenness": betweenness,
            "sentiment_avg": float(user_sentiments.get(node, 0.0)),
            "post_count": int(user_post_counts.get(node, 0)),
            "stance_score": float(metrics.get('stance_score', 0.0)),
            "stance_leaning": metrics.get('stance_leaning', 'neutral')
        })
        
    # Prepare JSON Edges
    json_edges = []
    for u, v, data in G.edges(data=True):
        json_edges.append({
            "from": u,
            "to": v,
            "weight": int(data.get("weight", 1)),
            "relationship": data.get("relationship", "REPLY")
        })
        
    # Get NRC emotion columns
    emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
    
    # Load global metrics to fetch modularities
    global_metrics_path = os.path.join(base_dir, "data", "network_global_metrics.csv")
    louvain_mod = 0.9678
    infomap_mod = 0.7500
    if os.path.exists(global_metrics_path):
        try:
            global_df = pd.read_csv(global_metrics_path)
            for _, row in global_df.iterrows():
                if row['Metric'] == 'Louvain Modularity':
                    louvain_mod = float(row['Value'])
                elif row['Metric'] == 'Infomap Modularity':
                    infomap_mod = float(row['Value'])
        except Exception as e:
            print(f"Error loading global metrics: {e}")
            
    # Profile all community detection algorithms
    louvain_comms = profile_communities(df, cent_df, "community", emotion_cols)
    infomap_comms = profile_communities(df, cent_df, "community_infomap", emotion_cols)
    
    algorithms = {
        "louvain": {
            "name": "Louvain",
            "modularity": louvain_mod,
            "communities": louvain_comms
        },
        "infomap": {
            "name": "Infomap",
            "modularity": infomap_mod,
            "communities": infomap_comms
        }
    }
    
    # Build complete dashboard structure
    dashboard_data = {
        "nodes": json_nodes,
        "edges": json_edges,
        "communities": louvain_comms, # For legacy fallback
        "algorithms": algorithms
    }
    
    # Save to file
    output_path = os.path.join(base_dir, "web", "data", "dashboard_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    main()
