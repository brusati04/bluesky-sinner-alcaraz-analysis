import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from utils import save_plot_copies, US_OPEN_EVENTS
from emotion_analysis import plot_emotion_distribution

def generate_visualizations(G, Gd, df_cent, comm_data, centralities, stance_results, nlp_results):
    
    # Unpack centralities
    deg_cent = centralities["deg_cent"]
    pagerank = centralities["pagerank"]
    
    # Unpack community data
    communities = comm_data["communities"]
    node_to_community = comm_data["node_to_community"]
    leiden_communities = comm_data["leiden_communities"]
    node_to_leiden = comm_data["node_to_leiden"]
    leiden_modularity = comm_data["leiden_modularity"]
    infomap_communities = comm_data["infomap_communities"]
    node_to_infomap = comm_data["node_to_infomap"]
    infomap_modularity = comm_data["infomap_modularity"]
    lpa_communities = comm_data["lpa_communities"]
    node_to_lpa = comm_data["node_to_lpa"]
    lpa_modularity = comm_data["lpa_modularity"]
    
    gcc = comm_data["gcc"]
    fluid_communities = comm_data["fluid_communities"]
    node_to_fluid = comm_data["node_to_fluid"]
    fluid_modularity = comm_data["fluid_modularity"]
    k_fluid = comm_data["k_fluid"]
    modularity_score = comm_data["modularity_score"]
    
    # Unpack stance results
    stance_leanings = stance_results["stance_leanings"]
    stance_assort = stance_results["stance_assort"]
    
    # Unpack NLP results
    df = nlp_results["df"]
    sinner_scores = nlp_results["sinner_scores"]
    alcaraz_scores = nlp_results["alcaraz_scores"]

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


    # 1.1. Leiden Network Graph
    plt.figure(figsize=(12, 12))
    node_colors_leiden = []
    if len(subG.nodes()) > 1 and len(leiden_communities) > 0:
        cmap = plt.cm.get_cmap('tab20', len(leiden_communities))
        for node in subG.nodes():
            comm_id = node_to_leiden.get(node, 0)
            node_colors_leiden.append(cmap(comm_id))
    else:
        node_colors_leiden = '#3A6073'

    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors_leiden, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Undirected Social Network Graph (degree centrality & Leiden partitions)\n(Modularity Q: {leiden_modularity:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("leiden_network_graph.png")
    plt.close()


    # 1.2. Infomap Network Graph
    plt.figure(figsize=(12, 12))
    node_colors_infomap = []
    if len(subG.nodes()) > 1 and len(infomap_communities) > 0:
        cmap = plt.cm.get_cmap('tab20', len(infomap_communities))
        for node in subG.nodes():
            comm_id = node_to_infomap.get(node, 0)
            node_colors_infomap.append(cmap(comm_id))
    else:
        node_colors_infomap = '#3A6073'

    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors_infomap, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Undirected Social Network Graph (degree centrality & Infomap partitions)\n(Modularity Q: {infomap_modularity:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("infomap_network_graph.png")
    plt.close()


    # 1.3. LPA Network Graph
    plt.figure(figsize=(12, 12))
    node_colors_lpa = []
    if len(subG.nodes()) > 1 and len(lpa_communities) > 0:
        cmap = plt.cm.get_cmap('tab20', len(lpa_communities))
        for node in subG.nodes():
            comm_id = node_to_lpa.get(node, 0)
            node_colors_lpa.append(cmap(comm_id))
    else:
        node_colors_lpa = '#3A6073'

    nx.draw_networkx_edges(subG, pos, alpha=0.15, edge_color="grey")
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors_lpa, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels_to_draw, font_size=9, font_weight="bold", font_color="#1e272c")
    plt.title(f"Undirected Social Network Graph (degree centrality & LPA partitions)\n(Modularity Q: {lpa_modularity:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("lpa_network_graph.png")
    plt.close()


    # 1.4. Fluid Communities (GCC)
    plt.figure(figsize=(12, 12))
    if len(gcc.nodes()) > 0:
        pos_gcc = nx.spring_layout(gcc, k=0.15, iterations=40, seed=42)
        node_colors_fluid = []
        if len(gcc.nodes()) > 1 and len(fluid_communities) > 0:
            cmap = plt.cm.get_cmap('tab20', len(fluid_communities))
            for node in gcc.nodes():
                comm_id = node_to_fluid.get(node, 0)
                node_colors_fluid.append(cmap(comm_id))
        else:
            node_colors_fluid = '#3A6073'

        node_sizes_fluid = [50 + (deg_cent[node] * 1200) for node in gcc.nodes()]
        nx.draw_networkx_edges(gcc, pos_gcc, alpha=0.15, edge_color="grey")
        nx.draw_networkx_nodes(gcc, pos_gcc, node_size=node_sizes_fluid, node_color=node_colors_fluid, alpha=0.9)
        labels_to_draw_fluid = {node: node for node in gcc.nodes() if node in top_10_nodes}
        nx.draw_networkx_labels(gcc, pos_gcc, labels=labels_to_draw_fluid, font_size=9, font_weight="bold", font_color="#1e272c")
    else:
        plt.text(0.5, 0.5, "GCC is empty", ha='center', va='center')

    plt.title(f"GCC Social Network Graph (degree centrality & Fluid Communities, k={k_fluid})\n(GCC Modularity Q: {fluid_modularity:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("fluid_network_graph.png")
    plt.close()


    # 1.5. Stance Propagation Graph
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
    plt.title(f"Stance Propagation Network Graph (Laplacian Smoothing)\n(Stance Assortativity: {stance_assort:.4f})", pad=15)
    plt.axis("off")
    plt.tight_layout()
    save_plot_copies("stance_network_graph.png")
    plt.close()


    # 2. Directed Social Network Graph
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


    # 3. Sentiment Category Distribution
    plt.figure(figsize=(8, 5))
    sentiment_counts = df['sentiment_category'].value_counts()
    palette = {"positive": "#2ecc71", "neutral": "#bdc3c7", "negative": "#e74c3c"}
    colors = [palette.get(cat, "#95a5a6") for cat in sentiment_counts.index]

    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, hue=sentiment_counts.index, legend=False)
    plt.title("Sentiment Distribution of Bluesky Tennis Posts")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Posts")

    total_posts = len(df)
    for i, val in enumerate(sentiment_counts.values):
        percentage = (val / total_posts) * 100
        plt.text(i, val + (total_posts * 0.01), f"{val} ({percentage:.1f}%)", ha='center', fontweight='semibold')

    plt.tight_layout()
    save_plot_copies("sentiment_distribution.png")
    plt.close()


    # Emotion Distribution Plots
    plot_emotion_distribution(df, output_dir="plots", title_suffix="")

    # 4. Sentiment Over Time (Dynamics)
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df.sort_values(by="created_at")
    df_sorted['rolling_sentiment'] = df_sorted['sentiment_compound'].rolling(
        window=min(30, len(df_sorted)), min_periods=5
    ).mean()

    ax.plot(df_sorted['created_at'], df_sorted['rolling_sentiment'],
            color="#34495e", linewidth=2.5, label="30-Post Rolling Mean")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6, label="Neutral baseline")

    ax.fill_between(df_sorted['created_at'], 0, df_sorted['rolling_sentiment'],
                    where=(df_sorted['rolling_sentiment'] >= 0),
                    color='#2ecc71', alpha=0.15, label="Positive region")
    ax.fill_between(df_sorted['created_at'], 0, df_sorted['rolling_sentiment'],
                    where=(df_sorted['rolling_sentiment'] < 0),
                    color='#e74c3c', alpha=0.15, label="Negative region")

    for event_date_str, label, color in US_OPEN_EVENTS:
        event_dt = pd.to_datetime(event_date_str, utc=True)
        if df_sorted['created_at'].min() <= event_dt <= df_sorted['created_at'].max():
            ax.axvline(x=event_dt, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
            y_pos = ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.3
            ax.text(event_dt, y_pos, label,
                    rotation=90, ha='right', va='top',
                    fontsize=7, color=color, alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_title("Sentiment Dynamics Over Time (30-Post Rolling Average)\nWith US Open 2025 Key Match Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("VADER Compound Sentiment Score")
    ax.legend(loc='lower right', fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_plot_copies("sentiment_over_time.png")
    plt.close()


    # 5. Sinner vs Alcaraz Sentiment Comparison
    fig, ax1 = plt.subplots(figsize=(9, 5))
    categories = ["Jannik Sinner", "Carlos Alcaraz"]
    avg_sinner_sent = np.mean(sinner_scores) if sinner_scores else 0.0
    avg_alcaraz_sent = np.mean(alcaraz_scores) if alcaraz_scores else 0.0
    averages = [avg_sinner_sent, avg_alcaraz_sent]
    mention_counts = [len(sinner_scores), len(alcaraz_scores)]

    x = np.arange(len(categories))
    width = 0.35

    color1 = '#3498db'
    ax1.set_xlabel('Tennis Player')
    ax1.set_ylabel('Number of Mentions', color=color1)
    bars1 = ax1.bar(x, mention_counts, width, label='Mention Volume', color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)

    ax2 = ax1.twinx()
    color2 = '#d35400'
    ax2.set_ylabel('Average VADER Compound Sentiment', color=color2)
    line2 = ax2.plot(x, averages, color=color2, marker='o', markersize=10, linewidth=3, label='Avg Sentiment')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.5, 0.5)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, height/2.0, f'{int(height)}', ha='center', va='center', color='white', fontweight='bold')

    for i, val in enumerate(averages):
        ax2.text(i, val + 0.05, f'{val:.3f}', ha='center', color=color2, fontweight='bold')

    plt.title("Rivalry Comparison: Mentions vs. Public Sentiment Profile")
    fig.tight_layout()
    save_plot_copies("rivalry_comparison.png")
    plt.close()

