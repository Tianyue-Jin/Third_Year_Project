import streamlit as st
from streamlit_shadcn_ui import tabs
import networkx as nx
import tempfile

import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform

import pyvis.network as pyvis_net
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import seaborn as sns
import random

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from cdlib.algorithms import leiden
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community.quality import modularity

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

from node2vec import Node2Vec


@st.cache_data
def generate_node2vec_embeddings(G, dimensions=64):
    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}
    return embeddings


# Function to plot in-degree distribution with log-log scaling
@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def plot_in_degree_distribution(G):
    plt.clf()  # Clear the current figure to prevent overlap

    # Aggregate in-degree counts
    in_degree_count = {}
    for node, degree in G.in_degree():
        in_degree_count[degree] = in_degree_count.get(degree, 0) + 1

    # Prepare data for plotting
    X = sorted(in_degree_count.keys())
    Y = [in_degree_count[degree] for degree in X]

    # Plot the in-degree distribution as a single line (similar to SNAP)
    plt.plot(X, Y, linestyle="-", color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("In-Degree")
    plt.ylabel("Frequency")
    plt.title("In-Degree Distribution (Log-Log Scale)")
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit


# Function to plot out-degree distribution with log-log scaling
@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def plot_out_degree_distribution(G):
    plt.clf()  # Clear the current figure to prevent overlap

    # Aggregate out-degree counts
    out_degree_count = {}
    for node, degree in G.out_degree():
        out_degree_count[degree] = out_degree_count.get(degree, 0) + 1

    # Prepare data for plotting
    X = sorted(out_degree_count.keys())
    Y = [out_degree_count[degree] for degree in X]

    # Plot the out-degree distribution as a single line (similar to SNAP)
    plt.plot(X, Y, linestyle="-", color="red")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Out-Degree")
    plt.ylabel("Frequency")
    plt.title("Out-Degree Distribution (Log-Log Scale)")
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit


@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def compute_graph_layout(G):
    """Compute and cache the layout of the graph."""
    return nx.spectral_layout(G)


@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def compute_common_neighbors(_G, node_pairs):
    """Compute the common neighbors for a given set of node pairs."""
    return [(u, v, len(list(nx.common_neighbors(_G, u, v)))) for u, v in node_pairs]


@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def compute_jaccard_coefficient(_G, node_pairs):
    """Compute the Jaccard Coefficient for a given set of node pairs."""
    jaccard_scores = nx.jaccard_coefficient(_G, node_pairs)
    return [(u, v, p) for u, v, p in jaccard_scores]


@st.cache_data(hash_funcs={nx.DiGraph: lambda _: None})
def compute_adamic_adar_index(_G, node_pairs):
    """Compute the Adamic/Adar index for a given set of node pairs."""
    adamic_adar_scores = nx.adamic_adar_index(_G, node_pairs)
    return [(u, v, p) for u, v, p in adamic_adar_scores]


def visualize_graph(G, num_nodes_to_display=200):
    # Sort nodes by degree and get a subgraph of important nodes
    important_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[
        :num_nodes_to_display
    ]
    subgraph_nodes = [node for node, _ in important_nodes]
    subgraph = G.subgraph(subgraph_nodes)

    # Generate positions using spring layout with increased separation
    positions = nx.spring_layout(subgraph, k=0.2, seed=42)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    # Fixed node size at 5
    fixed_node_size = 5

    # Create node traces
    node_x = []
    node_y = []
    node_ids = []
    node_colors = []

    for node in subgraph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))
        node_colors.append(subgraph.degree[node])  # Color nodes by degree

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",  # Only display information on hover
        marker=dict(
            showscale=True,
            colorscale="Blues",
            size=fixed_node_size,  # Set fixed size
            color=node_colors,
            colorbar=dict(
                title=dict(text="Node Degree"),
                thickness=15,
                xanchor="left",
            ),
            line_width=1,
        ),
        text=[
            f"Node {node_id}<br>Degree: {degree}"
            for node_id, degree in zip(node_ids, node_colors)
        ],  # Hover content
    )

    # Create interactive figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Interactive Network Graph (Top {num_nodes_to_display} Nodes by Degree)",  # ✅ Fix: Correct title setting
                font=dict(size=16),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Display interactive graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to display global network statistics
def display_global_statistics(G):
    undirected_G = (
        G.to_undirected()
    )  # Convert to undirected graph for triangle calculations

    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())
    max_degree = int(max(dict(G.degree()).values()))
    avg_degree = int(sum(dict(G.degree()).values()) / num_nodes)
    assortativity = nx.degree_assortativity_coefficient(G)
    num_triangles = int(sum(nx.triangles(undirected_G).values()) // 3)
    avg_triangles = num_triangles / num_edges if num_edges > 0 else 0
    max_triangles = int(
        max(nx.triangles(undirected_G).values()) if num_nodes > 0 else 0
    )
    avg_clustering_coeff = nx.average_clustering(undirected_G)
    global_clustering_coeff = nx.transitivity(undirected_G)
    max_k_core = int(max(nx.core_number(undirected_G).values()) if num_nodes > 0 else 0)
    lower_bound_max_clique = int(
        len(max(nx.find_cliques(undirected_G), key=len)) if num_nodes > 0 else 0
    )

    # Prepare the data as a list of tuples
    data = [
        ("Number of nodes", num_nodes),
        ("Number of edges", num_edges),
        ("Maximum degree", max_degree),
        ("Average degree", avg_degree),
        ("Assortativity Coefficient", assortativity),
        ("Number of triangles (3-clique)", num_triangles),
        ("Average triangles per edge", avg_triangles),
        ("Maximum number of triangles per node", max_triangles),
        ("Average local clustering coefficient", avg_clustering_coeff),
        ("Global clustering coefficient", global_clustering_coeff),
        ("Maximum k-core number", max_k_core),
        ("Lower bound on max clique size", lower_bound_max_clique),
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Metric", "Value"])

    # Apply formatting: Integers stay integers, floats are rounded to 4 decimals
    def format_values(val):
        if isinstance(val, float) and val.is_integer():
            return int(val)  # Convert integer-like floats to integers
        if isinstance(val, float):
            return f"{val:.4f}"  # Format floats to 4 decimal places
        return val

    # Format the 'Value' column
    df["Value"] = df["Value"].apply(format_values)

    # Render as HTML with exact formatting
    styled_html = df.to_html(index=False, escape=False, justify="left")

    st.write("### Global Network Statistics")
    st.write(styled_html, unsafe_allow_html=True)


def attribute_similarity(G, u, v, attribute="attribute_name"):
    attr_u = G.nodes[u].get(attribute, 0)
    attr_v = G.nodes[v].get(attribute, 0)
    return 1 if attr_u == attr_v else 0


def compute_link_prediction_features(G, data, selected_metrics):
    feature_dict = {
        "Node1": [u for u, v in data],
        "Node2": [v for u, v in data],
    }  # Track node pairs

    if "Common Neighbors" in selected_metrics:
        feature_dict["CommonNeighbors"] = [
            len(list(nx.common_neighbors(G, u, v))) for u, v in data
        ]

    if "Jaccard Coefficient" in selected_metrics:
        jaccard_scores = nx.jaccard_coefficient(G, data)
        feature_dict["JaccardCoefficient"] = [score for _, _, score in jaccard_scores]

    if "Adamic/Adar index" in selected_metrics:
        adamic_adar_scores = nx.adamic_adar_index(G, data)
        feature_dict["AdamicAdarIndex"] = [score for _, _, score in adamic_adar_scores]

    if "Preferential Attachment" in selected_metrics:
        pref_attachment_scores = nx.preferential_attachment(G, data)
        feature_dict["PreferentialAttachment"] = [
            score for _, _, score in pref_attachment_scores
        ]

    if "Resource Allocation Index" in selected_metrics:
        resource_scores = nx.resource_allocation_index(G, data)
        feature_dict["ResourceAllocationIndex"] = [
            score for _, _, score in resource_scores
        ]

    if "Shortest Path Distance" in selected_metrics:
        feature_dict["ShortestPathDistance"] = [
            nx.shortest_path_length(G, u, v) if nx.has_path(G, u, v) else float("inf")
            for u, v in data
        ]

    if "Hub Promoted Index" in selected_metrics:
        hub_promoted_scores = [
            (G.degree(u) * G.degree(v)) / max(G.degree(u), G.degree(v), 1)
            for u, v in data
        ]
        feature_dict["HubPromotedIndex"] = hub_promoted_scores

    if "Personalized PageRank" in selected_metrics:
        pr = nx.pagerank(G)  # Compute PageRank for all nodes
        feature_dict["PersonalizedPageRank"] = [
            pr.get(u, 0) + pr.get(v, 0) for u, v in data
        ]

    df = pd.DataFrame(feature_dict)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    df.fillna(0, inplace=True)  # Replace NaNs with 0

    for col in df.columns[2:]:  # Skip Node1 & Node2
        df[col] = np.log1p(df[col])  # log(1 + x) to avoid log(0) issues

    return df


# Function to determine rumour origin (automatic detection)
def identify_rumour_source(G):
    """Automatically select the node with the highest degree as the most likely rumour source."""
    if len(G.nodes) == 0:
        return None
    rumour_source = max(G.degree, key=lambda x: x[1])[0]
    return rumour_source


def adaptive_greedy_blocking(G, num_nodes):
    """Greedy method: iteratively removes nodes that maximize infection reduction."""
    blocked_nodes = set()

    for _ in range(num_nodes):
        # Evaluate impact of removing each node
        scores = {
            node: nx.betweenness_centrality(G)[node]
            for node in G.nodes()
            if node not in blocked_nodes
        }

        # Select the node with the highest impact
        if scores:
            best_node = max(scores, key=scores.get)
            blocked_nodes.add(best_node)

    return list(blocked_nodes)


def community_based_blocking(G, num_nodes):
    """Blocks nodes at critical junctions between communities."""

    if "communities" not in st.session_state:
        st.error(
            "No community detection results found. Please run community detection first."
        )
        return []

    communities = st.session_state["communities"]
    node_community_map = {
        node: i for i, community in enumerate(communities) for node in community
    }

    # Find bridge nodes connecting multiple communities
    bridge_nodes = [
        node
        for node in G.nodes()
        if len(
            set(
                node_community_map[neighbor]
                for neighbor in G.neighbors(node)
                if neighbor in node_community_map
            )
        )
        > 1
    ]

    return bridge_nodes[:num_nodes]


def randomized_rumour_blocking(G, num_nodes):
    """Implements R-GRB: Reverse BFS-based blocking, prioritizing highly connected neighbors."""
    nodes = list(G.nodes)
    blocked_nodes = set()

    while len(blocked_nodes) < num_nodes:
        # Pick a random node that hasn't been blocked yet
        random_node = random.choice(nodes)
        blocked_nodes.add(random_node)

        # Find its highest-degree neighbor and block it (reverse influence)
        neighbors = sorted(
            G.neighbors(random_node), key=lambda n: G.degree[n], reverse=True
        )
        for neighbor in neighbors:
            if len(blocked_nodes) < num_nodes:
                blocked_nodes.add(neighbor)
            else:
                break  # Stop once we reach the required number

    return list(blocked_nodes)


def hybrid_blocking(G, num_nodes):
    """Combines Degree, Betweenness, and Community-based blocking for a balanced approach."""

    num_per_strategy = max(
        1, num_nodes // 3
    )  # Evenly distribute across three strategies

    high_degree_nodes = determine_nodes_to_block(G, "Highest Degree", num_per_strategy)
    betweenness_nodes = determine_nodes_to_block(
        G, "Highest Betweenness Centrality", num_per_strategy
    )
    community_nodes = determine_nodes_to_block(
        G, "Community-Based Blocking", num_per_strategy
    )

    hybrid_set = set(high_degree_nodes + betweenness_nodes + community_nodes)
    return list(hybrid_set)[:num_nodes]  # Ensure final selection size matches num_nodes


# Function to determine nodes to block based on strategy
def determine_nodes_to_block(G, strategy, num_nodes):
    if strategy == "Random Blocking":
        return random.sample(
            list(G.nodes), min(num_nodes, len(G.nodes))
        )  # Select random unique nodes
    elif strategy == "Highest Degree":
        return [
            node
            for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[
                :num_nodes
            ]
        ]  # Extract only nodes
    elif strategy == "Highest Betweenness Centrality":
        betweenness = nx.betweenness_centrality(G)
        return sorted(betweenness, key=betweenness.get, reverse=True)[:num_nodes]
    elif strategy == "PageRank-Based Blocking":
        pagerank = nx.pagerank(G)
        return sorted(pagerank, key=pagerank.get, reverse=True)[:num_nodes]

    elif strategy == "Structural Equivalence-Based Blocking":
        # Compute Structural Equivalence Matrix
        adjacency_matrix = nx.to_numpy_array(G)
        similarity_matrix = 1 - squareform(pdist(adjacency_matrix, metric="euclidean"))
        np.fill_diagonal(similarity_matrix, 0)  # Remove self-pairs

        # Identify top structurally equivalent nodes by summing row similarities
        node_equivalence = similarity_matrix.sum(axis=1)
        sorted_nodes = np.argsort(node_equivalence)[::-1]  # Sort descending
        return list(sorted_nodes[:num_nodes])  # Select top structurally similar nodes

    elif strategy == "Structural Hole-Based Blocking":
        # Compute Effective Size of Nodes (Higher = More Structural Holes)
        effective_sizes = nx.effective_size(G)
        return sorted(effective_sizes, key=effective_sizes.get, reverse=True)[
            :num_nodes
        ]
    elif strategy == "Adaptive Greedy Blocking":
        return adaptive_greedy_blocking(G, num_nodes)
    elif strategy == "Community-Based Blocking":
        return community_based_blocking(G, num_nodes)
    elif strategy == "R-GRB (Efficient Randomized Blocking)":
        return randomized_rumour_blocking(G, num_nodes)
    elif strategy == "Hybrid Approach":
        return hybrid_blocking(G, num_nodes)


def apply_rumour_blocking(G, selected_nodes):
    """Removes blocked nodes from the graph to simulate rumour blocking."""
    if selected_nodes is None or len(selected_nodes) == 0:
        raise ValueError("No nodes selected for blocking!")

    remaining_graph = G.copy()
    remaining_graph.remove_nodes_from(selected_nodes)
    return remaining_graph


def visualize_blocked_graph(G, blocked_nodes, hide_blocked_edges=False):
    positions = nx.spring_layout(G, seed=42)

    # Separate traces for blocked and normal edges
    blocked_edge_x, blocked_edge_y = [], []
    normal_edge_x, normal_edge_y = [], []

    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]

        if hide_blocked_edges and (
            edge[0] in blocked_nodes or edge[1] in blocked_nodes
        ):
            continue  # Skip blocked edges if hiding is enabled

        # Determine if the edge is blocked
        if edge[0] in blocked_nodes or edge[1] in blocked_nodes:
            blocked_edge_x.extend([x0, x1, None])
            blocked_edge_y.extend([y0, y1, None])
        else:
            normal_edge_x.extend([x0, x1, None])
            normal_edge_y.extend([y0, y1, None])

    # Create edge traces
    normal_edge_trace = go.Scatter(
        x=normal_edge_x,
        y=normal_edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    blocked_edge_trace = go.Scatter(
        x=blocked_edge_x,
        y=blocked_edge_y,
        line=dict(width=0.5, color="red"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_x, node_y, node_colors, node_ids = [], [], [], []

    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))

        # Color nodes based on whether they are blocked or not
        node_colors.append("red" if node in blocked_nodes else "skyblue")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(color=node_colors, size=8, line_width=1),
        text=[f"Node {node_id}" for node_id in node_ids],
    )

    # Combine the traces and display
    fig = go.Figure(
        data=[normal_edge_trace, blocked_edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Interactive Graph with Fixed Blocked Edges", font=dict(size=16)
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def visualize_communities(G, communities, positions=None):
    if not positions:
        positions = nx.spring_layout(G, seed=42)  # Spring layout with a consistent seed

    # Assign colors to communities
    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
        for _ in range(len(communities))
    ]
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i]

    node_x, node_y, node_color, hover_text = [], [], [], []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(color_map[node])
        hover_text.append(
            f"Node {node}, Community {list(color_map.values()).index(color_map[node]) + 1}"
        )

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            color=node_color, size=10, line_width=0.5  # Fixed node size for consistency
        ),
        text=hover_text,
        hoverinfo="text",
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Interactive Community Detection Visualization", font=dict(size=16)
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig)


# Custom Walktrap-like Algorithm Using NetworkX
def optimal_walktrap_networkx(G, num_walks=10, walk_length=5, max_clusters=10):
    """
    Automatically determine the best number of communities using Walktrap-inspired clustering.
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Normalize adjacency matrix into a transition matrix
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid NaN errors
    transition_matrix = adj_matrix / row_sums  # Normalize rows

    # Simulate random walks
    scores = np.zeros((len(nodes), len(nodes)))

    for _ in range(num_walks):
        for i, node in enumerate(nodes):
            current_idx = node_to_idx[node]
            for _ in range(walk_length):
                if transition_matrix[current_idx].sum() == 0:
                    break  # Skip to avoid errors
                next_idx = np.random.choice(
                    len(nodes), p=transition_matrix[current_idx]
                )
                scores[i][next_idx] += 1
                current_idx = next_idx

    # Try different numbers of clusters and pick the best one based on modularity
    best_modularity = -1
    best_communities = None

    for num_clusters in range(2, min(max_clusters, len(nodes))):
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters, metric="precomputed", linkage="average"
        )
        labels = clustering.fit_predict(scores)

        # Convert labels to community format
        community_dict = {}
        for i, node in enumerate(nodes):
            label = labels[i]
            if label not in community_dict:
                community_dict[label] = set()
            community_dict[label].add(node)

        communities = list(community_dict.values())

        # Calculate modularity score
        modularity_score = modularity(G, communities)
        print(f"Clusters: {num_clusters}, Modularity: {modularity_score}")

        if modularity_score > best_modularity:
            best_modularity = modularity_score
            best_communities = communities

    print(f"Best modularity: {best_modularity}")
    return best_communities


centrality_methods = {
    "Degree Centrality": nx.degree_centrality,
    "In-Degree Centrality": nx.in_degree_centrality,
    "Out-Degree Centrality": nx.out_degree_centrality,
    "Closeness Centrality": nx.closeness_centrality,
    "Betweenness Centrality": nx.betweenness_centrality,
    "Edge Betweenness Centrality": nx.edge_betweenness_centrality,
    "Eigenvector Centrality": lambda G: nx.eigenvector_centrality(G, max_iter=1000),
    "Katz Centrality": lambda G: nx.katz_centrality(
        G, alpha=0.01, max_iter=5000, tol=1e-4
    ),
    "PageRank Centrality": lambda G: nx.pagerank(G, alpha=0.85),
}


def display_centrality_measures(G, selected_metrics):
    """Display centrality measures in a structured format with tables and charts."""

    for metric_name, metric_func in centrality_methods.items():
        if metric_name in selected_metrics:
            try:
                centrality_scores = metric_func(G)
                centrality_df = pd.DataFrame(
                    list(centrality_scores.items()), columns=["Node", "Centrality"]
                )
                centrality_df.sort_values(
                    by="Centrality", ascending=False, inplace=True
                )

                # Display top 10 nodes in a table
                st.write(f"### {metric_name}")
                st.dataframe(centrality_df.head(10).reset_index(drop=True))

                # Visualization: Bar Chart for Top 10 Nodes
                fig, ax = plt.subplots(figsize=(8, 5))
                top_10 = centrality_df.head(10)
                ax.bar(
                    top_10["Node"].astype(str),
                    top_10["Centrality"],
                    color="skyblue",
                    alpha=0.8,
                )
                ax.set_ylabel("Centrality Score")
                ax.set_xlabel("Node")
                ax.set_title(f"Top 10 Most Central Nodes in {metric_name}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not compute {metric_name}: {e}")


def reset_cache():
    """Clears Streamlit cache when the graph is changed."""
    st.cache_data.clear()  # Clear all cached computations


def generate_sample_graph(size):
    """
    Generate a sample graph dynamically based on the requested size.
    :param size: "small", "medium", or "large"
    :return: Generated NetworkX graph
    """
    if "uploaded_file" in st.session_state:
        del st.session_state["uploaded_file"]  # Remove uploaded file

    G = nx.DiGraph()  # Use directed graph by default

    if size == "small":
        G = nx.gnm_random_graph(50, 200, directed=True)  # 50 nodes, 200 edges
    elif size == "medium":
        G = nx.gnm_random_graph(500, 2000, directed=True)  # 500 nodes, 2000 edges
    elif size == "large":
        G = nx.gnm_random_graph(5000, 20000, directed=True)  # 5000 nodes, 20000 edges
    else:
        raise ValueError("Invalid size. Choose 'small', 'medium', or 'large'.")

    return G


def generate_custom_graph(num_nodes, num_edges, is_directed, is_sparse):
    """
    Generate a sample graph with user-defined settings.

    :param num_nodes: Number of nodes
    :param num_edges: Number of edges
    :param is_directed: Boolean for directed/undirected graph
    :param is_sparse: Boolean for sparse/dense graph structure
    :return: Generated NetworkX graph
    """

    if "uploaded_file" in st.session_state:
        del st.session_state["uploaded_file"]  # Remove uploaded file

    G = nx.DiGraph() if is_directed else nx.Graph()

    # Ensure the number of edges is valid
    max_possible_edges = (
        num_nodes * (num_nodes - 1)
        if is_directed
        else (num_nodes * (num_nodes - 1)) // 2
    )
    num_edges = min(num_edges, max_possible_edges)

    # Generate edges
    if is_sparse:
        # Sparse: Random G(n, m) model
        G = nx.gnm_random_graph(num_nodes, num_edges, directed=is_directed)
    else:
        # Dense: Preferential Attachment Model for scale-free properties
        G = nx.barabasi_albert_graph(
            num_nodes, min(5, num_nodes - 1)
        )  # Min degree of 5 or lower

        # Convert to directed if needed
        if is_directed:
            G = G.to_directed()

    return G


# Function to display graph metadata
def display_graph_metadata(G, source):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    graph_type = "Directed" if G.is_directed() else "Undirected"
    st.markdown(f"📄 **{source} Graph:** ({num_nodes} nodes, {num_edges} edges)")
    st.markdown(f"🔀 **Graph Type:** {graph_type}")


# ---- Initialize Session State ----
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if "graph" not in st.session_state:
    st.session_state["graph"] = None


# Define Navigation Items
main_tabs = [
    "Home",
    "Metrics",
    "Link Prediction",
    "Rumour Blocking",
    "Community Detection",
]
more_tabs = ["Compare Methods", "Finance & Business Applications"]

# Create a Navigation Bar
selected_tab = tabs(
    main_tabs + ["More ▼"],
    key="navigation",
    default=st.session_state.get("page", "Home"),
)

# Handle "More" dropdown separately
if selected_tab == "More ▼":
    selected_tab = tabs(more_tabs, key="more_navigation", default=more_tabs[0])

# If the user has not selected a page yet, set "Home" as default
if selected_tab not in main_tabs + more_tabs:
    selected_tab = "Home"

# Store the selected page in session state
st.session_state["page"] = selected_tab


# Display the correct page
if st.session_state["page"] == "Home":
    st.title("Home - Social Network Analysis Tool")
elif st.session_state["page"] == "Metrics":
    st.title("Network Metrics")
elif st.session_state["page"] == "Link Prediction":
    st.title("Link Prediction")
elif st.session_state["page"] == "Rumour Blocking":
    st.title("Rumour Blocking")
elif st.session_state["page"] == "Community Detection":
    st.title("Community Detection")
elif st.session_state["page"] == "Compare Methods":
    st.title("Comparison of Different Network Analysis Methods")
elif st.session_state["page"] == "Finance & Business Applications":
    st.title("Finance & Business Applications")


# ---- Home Page ----
page = st.session_state["page"]

# Page-specific content
if page == "Home":

    st.markdown(
        """
        ### 🏁 Quick Start Guide:
        1️⃣ **Upload a network graph (.txt or .mtx) or create a sample graph**  
        2️⃣ **Choose an analysis method** from the navigation bar  
        3️⃣ **View interactive visualizations & results**  
        4️⃣ **Compare different approaches to understand network patterns**
    """
    )

    uploaded_file = st.file_uploader(
        "Upload Graph File (TXT, MTX)", type=["txt", "mtx"]
    )

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["graph_source"] = "Uploaded"

        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Initialize the graph
        G = nx.DiGraph()  # Using DiGraph for directed graphs
        try:
            # Check the file type and load the graph accordingly
            if uploaded_file.name.endswith(".txt"):
                G = nx.read_edgelist(
                    temp_file_path, nodetype=int, create_using=nx.DiGraph()
                )
            elif uploaded_file.name.endswith(".mtx"):
                matrix = mmread(temp_file_path)
                row, col = matrix.nonzero()
                unique_nodes = set(row).union(set(col))
                for node in unique_nodes:
                    G.add_node(int(node))
                for i, j in zip(row, col):
                    if i != j:
                        G.add_edge(int(i), int(j))

            st.session_state["graph"] = G  # Store the graph in session state
            st.success(
                f"✅ `{uploaded_file.name}` uploaded successfully! Graph loaded."
            )

        except Exception as e:
            st.error(f" An error occurred while processing the graph: {e}")

    st.subheader("Try a sample network:")
    col1, col2, col3 = st.columns(3)

    if col1.button("📂 Generate Small Graph"):
        st.session_state["graph"] = generate_sample_graph("small")
        st.session_state["graph_source"] = "Generated"
        st.session_state.pop("uploaded_file", None)
        st.success("✅ Small graph generated successfully!")
        st.rerun()

    if col2.button("📂 Generate Medium Graph"):
        st.session_state["graph"] = generate_sample_graph("medium")
        st.session_state["graph_source"] = "Generated"
        st.session_state.pop("uploaded_file", None)
        st.success("✅ Medium graph generated successfully!")
        st.rerun()

    if col3.button("📂 Generate Large Graph"):
        st.session_state["graph"] = generate_sample_graph("large")
        st.session_state["graph_source"] = "Generated"
        st.session_state.pop("uploaded_file", None)
        st.success("✅ Large graph generated successfully!")
        st.rerun()

    with st.expander("🛠 Customize Sample Graph Generation"):
        col1, col2 = st.columns(2)

        # User-defined number of nodes and edges
        num_nodes = col1.number_input(
            "Number of Nodes", min_value=10, max_value=10000, value=100, step=10
        )
        num_edges = col2.number_input(
            "Number of Edges", min_value=10, max_value=100000, value=300, step=50
        )

        # User chooses directed or undirected
        is_directed = (
            st.radio("Graph Type", ["Directed", "Undirected"], index=0) == "Directed"
        )

        # User chooses sparse or dense network
        is_sparse = (
            st.radio(
                "Graph Density", ["Sparse (Random)", "Dense (Scale-Free)"], index=0
            )
            == "Sparse (Random)"
        )

        # Generate graph button
        if st.button("📂 Generate Custom Graph"):
            st.session_state["graph"] = generate_custom_graph(
                num_nodes, num_edges, is_directed, is_sparse
            )
            st.session_state["graph_source"] = "Generated"
            st.session_state.pop("uploaded_file", None)
            st.success(
                f"✅ Custom graph generated successfully! ({num_nodes} nodes, {num_edges} edges)"
            )
            st.rerun()

    # ---- Display Graph Metadata ----
    if "graph" in st.session_state and st.session_state["graph"] is not None:
        G = st.session_state["graph"]
        graph_source = st.session_state.get(
            "graph_source", "Generated"
        )  # Default to "Generated" if not set
        display_graph_metadata(G, graph_source)
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")


elif selected_tab == "Metrics":

    if "graph" in st.session_state and st.session_state["graph"] is not None:
        G = st.session_state["graph"]
        st.subheader("Select and visualize network centrality metrics.")

        # Dropdown menu to select metrics, now including in-degree and out-degree distribution
        metrics_options = [
            "Display Global Statistics",
            "Network Graph Visualization",
            "In-Degree Distribution",
            "Out-Degree Distribution",
            "Degree Centrality",
            "In-Degree Centrality",
            "Out-Degree Centrality",
            "Closeness Centrality",
            "Betweenness Centrality",
            "Edge Betweenness Centrality",
            "Eigenvector Centrality",
            "PageRank Centrality",
            "Katz Centrality",
            "Clustering Coefficients",
            "Diameter",
            "Average Shortest Path Length",
            "Triangles per Node",
            "Fragmentation",
            "Geodesic Distance",
            "Isolates",
            "Small-Worldness",
            "Structural Equivalence",
            "Structural Holes",
            "Transitivity",
        ]

        selected_metrics = st.multiselect("Select metrics to display:", metrics_options)

        # Display selected metrics
        if selected_metrics:
            st.write("## Selected Graph Metrics")
            G = st.session_state["graph"]  # Retrieve graph from session state

            if "Display Global Statistics" in selected_metrics:
                display_global_statistics(G)

            if "Network Graph Visualization" in selected_metrics:
                st.write("### Network Graph Visualization (Subsampled)")
                visualize_graph(G, num_nodes_to_display=200)

            if "In-Degree Distribution" in selected_metrics:
                plot_in_degree_distribution(G)

            if "Out-Degree Distribution" in selected_metrics:
                plot_out_degree_distribution(G)

            if any(metric in selected_metrics for metric in centrality_methods.keys()):
                display_centrality_measures(G, selected_metrics)

            if "Clustering Coefficients" in selected_metrics:
                clustering_coefficients = nx.clustering(G)

                # Convert to DataFrame and sort
                df_clustering = pd.DataFrame(
                    clustering_coefficients.items(),
                    columns=["Node", "Clustering Coefficient"],
                )
                df_clustering = df_clustering.sort_values(
                    by="Clustering Coefficient", ascending=False
                )

                # Display in Streamlit
                st.write("##### Clustering Coefficients Table")
                st.dataframe(df_clustering)

                # st.write("**Clustering Coefficients:**", clustering_coefficients)
                st.write("##### Clustering Coefficient Distribution")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    clustering_coefficients.values(), bins=10, color="blue", alpha=0.7
                )
                ax.set_xlabel("Clustering Coefficient")
                ax.set_ylabel("Number of Nodes")
                ax.set_title("Distribution of Clustering Coefficients")

                st.pyplot(fig)

                st.write("##### Clustering Coefficient vs Degree")

                node_degrees = dict(G.degree())

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(
                    x=list(node_degrees.values()),
                    y=list(clustering_coefficients.values()),
                    ax=ax,
                )
                ax.set_xlabel("Node Degree")
                ax.set_ylabel("Clustering Coefficient")
                ax.set_title("Node Degree vs Clustering Coefficient")

                st.pyplot(fig)

            if "Diameter" in selected_metrics:
                if nx.is_connected(G.to_undirected()):
                    diameter = nx.diameter(G.to_undirected())
                else:
                    diameter = "Graph not connected"
                st.write("**Diameter:**", diameter)

            if "Average Shortest Path Length" in selected_metrics:
                if nx.is_connected(G.to_undirected()):
                    avg_shortest_path_length = nx.average_shortest_path_length(
                        G.to_undirected()
                    )
                else:
                    avg_shortest_path_length = "Graph not connected"
                st.write("**Average Shortest Path Length:**", avg_shortest_path_length)

            if "Triangles per Node" in selected_metrics:
                triangle_counts = nx.triangles(G.to_undirected())
                # Convert to DataFrame and sort
                df_triangles = pd.DataFrame(
                    triangle_counts.items(), columns=["Node", "Number of Triangles"]
                )
                df_triangles = df_triangles.sort_values(
                    by="Number of Triangles", ascending=False
                )

                # Display in Streamlit
                st.write("##### Number of Triangles per Node (Sorted)")
                st.dataframe(df_triangles)

                st.write("##### Distribution of Triangle Counts")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(triangle_counts.values(), bins=10, color="blue", alpha=0.7)
                ax.set_xlabel("Number of Triangles")
                ax.set_ylabel("Number of Nodes")
                ax.set_title("Distribution of Triangles per Node")

                st.pyplot(fig)

                st.write("### Node Degree vs Number of Triangles")

                node_degrees = dict(G.degree())

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(
                    x=list(node_degrees.values()),
                    y=list(triangle_counts.values()),
                    ax=ax,
                )
                ax.set_xlabel("Node Degree")
                ax.set_ylabel("Number of Triangles")
                ax.set_title("Node Degree vs Triangle Count")

                st.pyplot(fig)

            if "Fragmentation" in selected_metrics:
                st.write("### Fragmentation")
                # Calculate reachable pairs
                connected_components = nx.connected_components(G.to_undirected())
                total_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
                reachable_pairs = sum(
                    len(c) * (len(c) - 1) for c in connected_components
                )

                # Calculate fragmentation
                fragmentation = 1 - (
                    reachable_pairs / total_pairs if total_pairs > 0 else 1
                )
                st.write(f"**Fragmentation:** {fragmentation:.4f}")

            if "Geodesic Distance" in selected_metrics:
                st.write("### Geodesic Distance")
                try:
                    # Compute shortest paths
                    path_lengths = dict(nx.shortest_path_length(G))
                    distances = [
                        length
                        for source, targets in path_lengths.items()
                        for target, length in targets.items()
                    ]
                    avg_distance = np.mean(distances)
                    max_distance = np.max(distances)
                    st.write(f"**Average Geodesic Distance:** {avg_distance:.4f}")
                    st.write(f"**Maximum Geodesic Distance:** {max_distance}")
                except nx.NetworkXError:
                    st.warning(
                        "Geodesic distance cannot be calculated for disconnected graphs."
                    )

            # Handle Isolates
            if "Isolates" in selected_metrics:
                st.write("### Isolates")
                isolates = list(nx.isolates(G))
                if isolates:
                    st.write(
                        f"**Isolates:** {len(isolates)} nodes are disconnected from the graph."
                    )
                    st.write(f"Isolated nodes: {isolates}")
                else:
                    st.write("No isolates in the graph.")

            if "Small-Worldness" in selected_metrics:
                st.write("### Small-Worldness")

                try:
                    # Compute clustering coefficient and average path length of the network
                    clustering_coeff = nx.average_clustering(G.to_undirected())
                    if nx.is_connected(G.to_undirected()):
                        avg_path_length = nx.average_shortest_path_length(
                            G.to_undirected()
                        )
                    else:
                        avg_path_length = float(
                            "inf"
                        )  # Not applicable if the graph is disconnected

                    # Generate a random graph with the same number of nodes and edges
                    random_graph = nx.gnm_random_graph(
                        G.number_of_nodes(), G.number_of_edges()
                    )
                    clustering_coeff_rand = nx.average_clustering(random_graph)
                    if nx.is_connected(random_graph):
                        avg_path_length_rand = nx.average_shortest_path_length(
                            random_graph
                        )
                    else:
                        avg_path_length_rand = float(
                            "inf"
                        )  # Handle disconnected random graph

                    # Calculate Small-Worldness
                    if avg_path_length_rand > 0 and clustering_coeff_rand > 0:
                        small_worldness = (clustering_coeff / clustering_coeff_rand) / (
                            avg_path_length / avg_path_length_rand
                        )
                        st.write(f"**Small-Worldness:** {small_worldness:.4f}")
                        st.write(
                            f"**Clustering Coefficient (Network):** {clustering_coeff:.4f}"
                        )
                        st.write(
                            f"**Clustering Coefficient (Random Graph):** {clustering_coeff_rand:.4f}"
                        )
                        st.write(
                            f"**Average Path Length (Network):** {avg_path_length:.4f}"
                        )
                        st.write(
                            f"**Average Path Length (Random Graph):** {avg_path_length_rand:.4f}"
                        )
                    else:
                        st.warning(
                            "Small-Worldness cannot be calculated due to disconnected components in the random graph."
                        )
                except Exception as e:
                    st.error(
                        f"An error occurred while calculating Small-Worldness: {e}"
                    )

            # Handle Structural Equivalence
            if "Structural Equivalence" in selected_metrics:
                st.write("### Structural Equivalence")

                # Convert the adjacency matrix
                adjacency_matrix = nx.to_numpy_array(G)

                # Calculate pairwise similarity (1 - normalized Euclidean distance)
                similarity_matrix = 1 - squareform(
                    pdist(adjacency_matrix, metric="euclidean")
                )

                # Ensure similarity values are within valid range
                similarity_matrix = np.nan_to_num(
                    similarity_matrix
                )  # Replace NaN with 0

                # Remove diagonal values (self-pairs)
                np.fill_diagonal(similarity_matrix, 0)

                # Extract non-self node pairs and their similarity
                similar_nodes = [
                    (i, j, similarity_matrix[i, j])
                    for i in range(similarity_matrix.shape[0])
                    for j in range(
                        i + 1, similarity_matrix.shape[1]
                    )  # Avoid self-pairs (i != j)
                ]

                # Convert to DataFrame
                df_similar_nodes = pd.DataFrame(
                    similar_nodes, columns=["Node 1", "Node 2", "Similarity"]
                )
                df_similar_nodes = df_similar_nodes.sort_values(
                    by="Similarity", ascending=False
                )  # Sort for better readability

                # **Display the table of node pairs with similarity scores**
                st.write(
                    "##### Most Structurally Equivalent Node Pairs (Sorted by Similarity)"
                )
                st.dataframe(df_similar_nodes.style.format({"Similarity": "{:.3f}"}))

                # Ensure similarity matrix has valid values
                if np.isnan(similarity_matrix).all() or np.max(similarity_matrix) == 0:
                    st.warning(
                        "No meaningful structural equivalence detected. The similarity matrix is empty or contains only zero values."
                    )
                else:
                    # Normalize the similarity matrix to [0,1] range for better visualization
                    norm_sim_matrix = (
                        similarity_matrix - np.min(similarity_matrix)
                    ) / (np.max(similarity_matrix) - np.min(similarity_matrix))

                    # Display Clustered Heatmap
                    st.write("##### Clustered Structural Equivalence Matrix")

                    try:
                        sns.clustermap(
                            norm_sim_matrix,
                            cmap="coolwarm",
                            method="average",  # Clustering method
                            figsize=(10, 8),
                            row_cluster=True,
                            col_cluster=True,
                        )
                        st.pyplot(plt)

                        # **Add Explanation**
                        st.markdown(
                            """
                        **Interpretation of the Heatmap:**  
                        - This heatmap represents the **structural equivalence** between nodes in the network.  
                        - The color intensity indicates **similarity**:  
                            - 🔴 **Red areas** indicate high structural similarity (nodes with similar neighbors).  
                            - 🔵 **Blue areas** indicate low similarity.  
                        - The dendrograms on the sides represent hierarchical **clustering** of nodes based on their similarity scores.
                        """
                        )

                    except ValueError as e:
                        st.error(f"Error plotting structural equivalence matrix: {e}")

            if "Structural Holes" in selected_metrics:
                st.write("### Structural Holes")

                # Compute effective size of structural holes
                effective_sizes = nx.effective_size(G)

                # Convert dictionary to DataFrame
                df_holes = pd.DataFrame.from_dict(
                    effective_sizes, orient="index", columns=["Effective Size"]
                )

                # Sort by Effective Size (Descending)
                df_holes = df_holes.sort_values(by="Effective Size", ascending=False)

                # Display Top Structural Hole Brokers
                st.write("##### Nodes Bridging Structural Holes")
                st.dataframe(df_holes)

                # Scatter plot: Effective Size vs Node Degree
                st.write("##### Structural Holes: Effective Size vs Node Degree")
                df_holes["Degree"] = df_holes.index.map(
                    lambda x: G.degree[x]
                )  # Get node degree

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df_holes, x="Degree", y="Effective Size", ax=ax)
                ax.set_title("Structural Holes: Effective Size vs Degree")
                ax.set_xlabel("Node Degree")
                ax.set_ylabel("Effective Size")
                st.pyplot(fig)

            if "Transitivity" in selected_metrics:
                st.write("### Transitivity")

                # Compute transitivity
                transitivity_value = nx.transitivity(G)

                # Display results
                st.write(
                    f"**Global Transitivity (Clustering Coefficient):** {transitivity_value:.4f}"
                )
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")


elif page == "Link Prediction":

    if st.session_state["graph"]:
        G = st.session_state["graph"]
        # Convert to undirected graph
        undirected_G = G.to_undirected()

        # Existing link prediction logic
        # st.write("Perform link prediction using various network metrics.")
        st.subheader("Link Prediction Metrics & Model Evaluation")

        # Available link prediction metrics
        link_prediction_metrics_options = [
            "Common Neighbors",
            "Jaccard Coefficient",
            "Adamic/Adar index",
            "Preferential Attachment",
            "Resource Allocation Index",
            "Shortest Path Distance",
            "Hub Promoted Index",
            "Personalized PageRank",
        ]
        selected_metrics = st.multiselect(
            "Select metrics to display:", link_prediction_metrics_options
        )

        if selected_metrics:
            # Prepare dataset for training
            existing_edges = list(undirected_G.edges())
            non_edges = list(nx.non_edges(undirected_G))
            non_edges_sample = non_edges[
                : len(existing_edges)
            ]  # Balance positive and negative samples

            # Create a dataset with labels (1 for existing edges, 0 for non-edges)
            data = existing_edges + non_edges_sample
            labels = [1] * len(existing_edges) + [0] * len(non_edges_sample)

            # Compute link prediction features based on selected metrics
            link_features_df = compute_link_prediction_features(
                undirected_G, data, selected_metrics
            )
            st.write("**Extracted Link Prediction Features:**")
            st.dataframe(link_features_df.head(10))

            if st.button("Train Link Prediction Model and Plot AUC"):
                st.write("Training link prediction model...")

                # Train/Test split
                X_train, X_test, y_train, y_test = train_test_split(
                    link_features_df.drop(["Node1", "Node2"], axis=1),
                    labels,
                    test_size=0.3,
                    random_state=42,
                )

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "XGBoost": xgb.XGBClassifier(
                        use_label_encoder=False, eval_metric="logloss"
                    ),
                }

                model_results = {}

                # Metrics for all models
                eval_metrics = []

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)[
                        :, 1
                    ]  # Probability for ROC curve

                    # Compute AUC Score
                    auc_score = roc_auc_score(y_test, y_pred_prob)
                    model_results[model_name] = auc_score

                    # Compute Evaluation Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Store Metrics in Table
                    eval_metrics.append(
                        [model_name, accuracy, precision, recall, f1, auc_score]
                    )

                # **Display AUC Scores for All Models**
                st.write("**AUC Score Comparison Across Models:**")
                auc_df = pd.DataFrame(
                    model_results.items(), columns=["Model", "AUC Score"]
                )
                st.dataframe(auc_df)

                # **Display Full Evaluation Metrics**
                st.write("### Model Performance Metrics:")
                df_metrics = pd.DataFrame(
                    eval_metrics,
                    columns=[
                        "Model",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score",
                        "AUC Score",
                    ],
                )
                st.dataframe(df_metrics)

                # **Plot ROC Curves for Each Model**
                fig, ax = plt.subplots(figsize=(6, 4))
                for model_name, model in models.items():
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    ax.plot(
                        fpr,
                        tpr,
                        label=f"{model_name} (AUC={roc_auc_score(y_test, y_pred_prob):.2f})",
                    )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="gray",
                    linestyle="--",
                    label="Random (Baseline)",
                )
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve Comparison")
                ax.legend()
                st.pyplot(fig)

    else:
        st.warning("No graph loaded. Please upload a valid graph file.")

# Rumour Blocking Page
elif page == "Rumour Blocking":

    if st.session_state["graph"]:
        G = st.session_state["graph"]
        st.subheader(
            "Interactively block nodes and visualize the affected areas of the network."
        )

        # **Automatically detect rumour source**
        rumour_source = identify_rumour_source(G)
        st.write(
            f" **Identified Rumour Source:** Node `{rumour_source}` (Highest Degree)"
        )

        # Dropdown to select blocking strategy
        blocking_strategy = st.selectbox(
            "Select a blocking strategy:",
            [
                "Random Blocking",
                "Highest Degree",
                "Highest Betweenness Centrality",
                "PageRank-Based Blocking",
                "Structural Equivalence-Based Blocking",
                "Structural Hole-Based Blocking",
                "Adaptive Greedy Blocking",
                "Community-Based Blocking",
                "R-GRB (Efficient Randomized Blocking)",
                "Hybrid Approach",
            ],
        )

        # Number of nodes to block
        num_nodes = st.slider(
            "Number of nodes to block:", min_value=1, max_value=10, value=3
        )

        # Nodes selected via algorithm
        nodes_to_block = determine_nodes_to_block(G, blocking_strategy, num_nodes)

        # Show the selected nodes
        st.write(
            f"**Nodes selected for blocking ({blocking_strategy}):** {nodes_to_block}"
        )

        # Toggle to hide blocked edges
        hide_edges = st.checkbox("Hide edges connected to blocked nodes", value=False)

        if st.button("Apply Rumour Blocking"):
            if not nodes_to_block:
                st.warning("Please select at least one node to block.")
            else:
                # Create a new blocked graph (instead of using visualization function)
                blocked_nodes = determine_nodes_to_block(
                    G, blocking_strategy, num_nodes
                )

                blocked_G = apply_rumour_blocking(G, blocked_nodes)

                # Visualize the graph with blocked nodes and edges highlighted
                fig = visualize_blocked_graph(
                    G, nodes_to_block, hide_blocked_edges=hide_edges
                )
                st.plotly_chart(fig, use_container_width=True)

                # ------------------  EVALUATION SECTION ------------------
                st.subheader("Evaluation Metrics After Rumour Blocking")
                blocked_G_undirected = blocked_G.to_undirected()

                # Simulating Rumour Spread
                infection_results = {}

                # Function to run the epidemic model and store results
                def run_epidemic_model(
                    model_class, model_name, G, blocked_G, config_params
                ):
                    """Runs an epidemic model and stores infection counts."""
                    model = model_class(G)
                    config = mc.Configuration()
                    for param, value in config_params.items():
                        config.add_model_parameter(param, value)
                    model.set_initial_status(config)
                    iterations_before = model.iteration_bunch(10)
                    infected_before = sum(
                        iter["node_count"][1] for iter in iterations_before
                    )
                    peak_before = max(
                        iter["node_count"][1] for iter in iterations_before
                    )

                    model = model_class(blocked_G)
                    model.set_initial_status(config)
                    iterations_after = model.iteration_bunch(10)
                    infected_after = sum(
                        iter["node_count"][1] for iter in iterations_after
                    )
                    peak_after = max(iter["node_count"][1] for iter in iterations_after)

                    infection_results[model_name] = {
                        "infected_before": infected_before,
                        "infected_after": infected_after,
                        "peak_before": peak_before,
                        "peak_after": peak_after,
                        "iterations_before": iterations_before,
                        "iterations_after": iterations_after,
                    }

                # Run models and store results
                run_epidemic_model(
                    ep.IndependentCascadesModel,
                    "IC Model",
                    G,
                    blocked_G,
                    {"fraction_infected": 0.05},
                )
                run_epidemic_model(ep.SIModel, "SI Model", G, blocked_G, {"beta": 0.01})
                run_epidemic_model(
                    ep.SIRModel,
                    "SIR Model",
                    G,
                    blocked_G,
                    {"beta": 0.01, "gamma": 0.005},
                )
                run_epidemic_model(
                    ep.ThresholdModel,
                    "LT Model",
                    G,
                    blocked_G,
                    {"fraction_infected": 0.05},
                )

                # Storing Evaluation Metrics
                metrics = []
                # 1. Number of Connected Components (Should Increase)
                before_components = nx.number_connected_components(G.to_undirected())
                after_components = nx.number_connected_components(blocked_G_undirected)
                metrics.append(
                    [
                        "Connected Components",
                        before_components,
                        after_components,
                        "Higher means more fragmentation",
                    ]
                )

                # 2.Average Shortest Path Length (Largest Component Only)
                # Convert the graph to undirected
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                subgraph_before = G.subgraph(largest_cc).to_undirected()

                # Compute average shortest path length only if the graph is connected
                avg_path_before = (
                    nx.average_shortest_path_length(subgraph_before)
                    if nx.is_connected(subgraph_before)
                    else np.nan
                )

                largest_cc_after = max(
                    nx.connected_components(blocked_G.to_undirected()), key=len
                )
                subgraph_after = blocked_G.subgraph(largest_cc_after).to_undirected()
                avg_path_after = (
                    nx.average_shortest_path_length(subgraph_after)
                    if nx.is_connected(subgraph_after)
                    else np.nan
                )

                metrics.append(
                    [
                        "Avg. Shortest Path (Largest Component)",
                        round(avg_path_before, 2),
                        round(avg_path_after, 2),
                        "Higher means rumour spreads slower",
                    ]
                )

                # Size of Largest Connected Component
                largest_cc_size_before = len(
                    max(nx.connected_components(G.to_undirected()), key=len)
                )
                largest_cc_size_after = len(
                    max(nx.connected_components(blocked_G.to_undirected()), key=len)
                )
                metrics.append(
                    [
                        "Largest Connected Component Size",
                        largest_cc_size_before,
                        largest_cc_size_after,
                        "Lower means rumour containment was effective",
                    ]
                )

                # 3. Eigenvector Centrality Reduction(Rumour Influence) (Should Decrease)
                before_centrality = sum(
                    nx.eigenvector_centrality_numpy(G, max_iter=1000).values()
                )
                after_centrality = sum(
                    nx.eigenvector_centrality_numpy(blocked_G, max_iter=1000).values()
                )
                metrics.append(
                    [
                        "Eigenvector Centrality",
                        round(before_centrality, 2),
                        round(after_centrality, 2),
                        "Lower means less influence",
                    ]
                )

                # 4. Mean Infection Time
                mean_infection_before = sum(
                    sum(iter["status"].values())
                    for model in infection_results.values()
                    for iter in model["iterations_before"]
                ) / (len(infection_results) * 10)
                mean_infection_after = sum(
                    sum(iter["status"].values())
                    for model in infection_results.values()
                    for iter in model["iterations_after"]
                ) / (len(infection_results) * 10)
                metrics.append(
                    [
                        "Mean Infection Time",
                        round(mean_infection_before, 2),
                        round(mean_infection_after, 2),
                        "Higher means blocking was effective",
                    ]
                )

                # 5️. Peak Infection Reduction (Across Models)
                peak_before_total = sum(
                    model["peak_before"] for model in infection_results.values()
                )
                peak_after_total = sum(
                    model["peak_after"] for model in infection_results.values()
                )
                metrics.append(
                    [
                        "Peak Infection",
                        peak_before_total,
                        peak_after_total,
                        "Lower means rumour spread was reduced",
                    ]
                )

                # 6️. Final Infection Spread Percentage (Averaged Across Models)
                total_nodes = len(G.nodes)
                final_spread_before = (
                    sum(
                        model["infected_before"] / total_nodes
                        for model in infection_results.values()
                    )
                    * 100
                    / len(infection_results)
                )
                final_spread_after = (
                    sum(
                        model["infected_after"] / total_nodes
                        for model in infection_results.values()
                    )
                    * 100
                    / len(infection_results)
                )
                metrics.append(
                    [
                        "Final Infection Spread (%)",
                        round(final_spread_before, 2),
                        round(final_spread_after, 2),
                        "Lower is better",
                    ]
                )

                # --------------------------------------
                # ** Display Results in Table**
                df_metrics = pd.DataFrame(
                    metrics,
                    columns=[
                        "Metric",
                        "Before Blocking",
                        "After Blocking",
                        "Interpretation",
                    ],
                )
                st.dataframe(df_metrics)

                # **Visualization: Bar Chart for Infected Nodes Across Models**
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = pd.DataFrame.from_dict(infection_results, orient="index")[
                    ["infected_before", "infected_after"]
                ]

                # **Plot Bar Chart**
                df_plot.plot(kind="bar", ax=ax, color=["red", "green"], alpha=0.7)

                # **Labels and Title**
                ax.set_ylabel("Number of Infected Nodes")
                ax.set_title("Rumour Spread Before vs After Blocking (Across Models)")
                ax.legend(["Before Blocking", "After Blocking"])

                # **Add Value Labels Above Bars**
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        fmt="%d",
                        label_type="edge",
                        fontsize=10,
                        fontweight="bold",
                        padding=1,
                    )

                # **Fix: Ensure Model Names Appear Horizontally**
                ax.set_xticklabels(df_plot.index, rotation=0)

                # **Display the Plot**
                st.pyplot(fig)
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")

# Community Detection Page
elif page == "Community Detection":
    if st.session_state["graph"]:
        G = st.session_state["graph"]
        st.subheader("Detect communities in the graph using various algorithms.")

        algorithm = st.selectbox(
            "Select a community detection algorithm:",
            [
                "Louvain Method",
                "Girvan-Newman Algorithm",
                "Leiden Algorithm",
                "Label Propagation Algorithm",
                "Spectral Clustering",
                "Edge Betweenness",
                "Walktrap Algorithm",
            ],
        )

        if st.button("Run Community Detection"):
            communities = None  # Reset communities before running new detection
            st.session_state["community_warning_displayed"] = False
            st.session_state["hub_index_error_displayed"] = False

            # ---- Run the selected community detection algorithm ----
            if algorithm == "Louvain Method":
                communities = list(nx.community.louvain_communities(G))

            elif algorithm == "Girvan-Newman Algorithm":
                comp = nx.community.girvan_newman(G)
                communities = tuple(sorted(c) for c in next(comp))

            elif algorithm == "Leiden Algorithm":
                # Apply Leiden directly on the NetworkX graph
                partition = leiden(G)
                # Convert Leiden result to a list of sets (each set is a community)
                communities = [set(comm) for comm in partition.communities]

            elif algorithm == "Label Propagation Algorithm":
                communities = list(asyn_lpa_communities(G))

            elif algorithm == "Spectral Clustering":
                adjacency_matrix = nx.to_numpy_array(G)
                spectral = SpectralClustering(
                    n_clusters=2, affinity="precomputed", assign_labels="discretize"
                )
                labels = spectral.fit_predict(adjacency_matrix)

                # Convert dictionary {node: label} into list of sets
                community_dict = {}
                for i, node in enumerate(G.nodes()):
                    label = labels[i]
                    if label not in community_dict:
                        community_dict[label] = set()
                    community_dict[label].add(node)

                # Convert to list of sets (expected format for visualization)
                communities = list(community_dict.values())

            elif algorithm == "Edge Betweenness":
                comp = nx.community.girvan_newman(G)
                communities = tuple(sorted(c) for c in next(comp))

            elif algorithm == "Walktrap Algorithm":
                communities = optimal_walktrap_networkx(
                    G, max_clusters=10
                )  # Adjust number of clusters as needed

            # ---- Store Results in Session State ----
            if communities:
                # Ensure correct dictionary format: {node: community_id}
                community_dict = {
                    node: i
                    for i, community in enumerate(communities)
                    for node in community
                }
                st.session_state["communities"] = communities
                st.session_state["community_dict"] = community_dict
                st.success(f"**Number of communities detected:** `{len(communities)}`")

            else:
                st.error("No valid communities detected. Try a different algorithm.")

            # Visualization
            visualize_communities(G, communities)

            # ---- Show Community Detection Results ----
            if "communities" in st.session_state:
                st.subheader("📌 Community Detection Results")
                st.write(
                    f"🔹 **Total Communities Found:** `{len(st.session_state['communities'])}`"
                )

                # Show a few sample communities
                for i, comm in enumerate(
                    st.session_state["communities"][:10]
                ):  # Limit display
                    st.write(
                        f"🟢 **Community {i+1}** ({len(comm)} nodes):",
                        list(comm)[:5],
                        " ...",
                    )

            # ------------------  EVALUATION SECTION ------------------
            st.subheader("Community Detection Evaluation Metrics")
            evaluation_metrics = []

            # **1. Modularity Score (Higher = better community structure)**
            modularity = nx.community.quality.modularity(G, communities)
            evaluation_metrics.append(
                [
                    "Modularity Score",
                    round(modularity, 4),
                    "Higher means better-defined communities",
                ]
            )

            # **2. Number of Communities**
            evaluation_metrics.append(
                [
                    "Number of Communities",
                    len(communities),
                    "More communities indicate finer granularity",
                ]
            )

            # **3. Average Community Size**
            avg_size = sum(len(c) for c in communities) / len(communities)
            evaluation_metrics.append(
                [
                    "Average Community Size",
                    round(avg_size, 4),
                    "Larger means broader community structure",
                ]
            )

            # **4. Community Size Variation (Lower = more evenly sized communities)**
            community_sizes = [len(c) for c in communities]
            std_dev_size = np.std(community_sizes)
            evaluation_metrics.append(
                [
                    "Community Size Variation",
                    round(std_dev_size, 4),
                    "Lower means more evenly sized communities",
                ]
            )

            # **5. Conductance (Lower = better community separation)**
            def compute_conductance(G, community):
                cut_size = sum(
                    1 for _ in nx.edge_boundary(G, community)
                )  # Edges between C and rest of G
                total_degree = sum(
                    dict(G.degree(community)).values()
                )  # Total degree of nodes in C
                return (
                    cut_size / total_degree if total_degree > 0 else np.nan
                )  # Avoid division by zero

            conductance_values = [
                compute_conductance(G, c) for c in communities if len(c) > 1
            ]
            avg_conductance = (
                np.nanmean(conductance_values) if conductance_values else np.nan
            )
            evaluation_metrics.append(
                [
                    "Average Conductance",
                    round(avg_conductance, 4),
                    "Lower means better-separated communities",
                ]
            )

            # **6. Average Community Density (Higher = denser communities)**
            densities = [nx.density(G.subgraph(c)) for c in communities if len(c) > 1]
            avg_density = np.nanmean(densities) if densities else np.nan
            evaluation_metrics.append(
                [
                    "Average Community Density",
                    round(avg_density, 4),
                    "Higher means denser communities",
                ]
            )

            # **7. Average Clustering Coefficient (Higher = more tightly clustered communities)**
            avg_clustering = np.mean(
                [
                    nx.average_clustering(G.subgraph(c))
                    for c in communities
                    if len(c) > 1
                ]
            )
            evaluation_metrics.append(
                [
                    "Average Clustering Coefficient",
                    round(avg_clustering, 4),
                    "Higher means nodes are tightly clustered",
                ]
            )

            # **Convert to DataFrame for Display**
            df_metrics = pd.DataFrame(
                evaluation_metrics, columns=["Metric", "Value", "Interpretation"]
            )

            # **Better Styling for Table**
            st.dataframe(
                df_metrics.style.format({"Value": "{:.4f}"}).set_properties(
                    **{"text-align": "left"}
                )
            )
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")


elif page == "Compare Methods":
    # st.title("Comparison of Different Network Analysis Methods")
    if st.session_state["graph"]:
        G = st.session_state["graph"]
        comparison_category = st.selectbox(
            "Select Comparison Category",
            ["Link Prediction", "Rumour Blocking", "Community Detection"],
        )

        if comparison_category == "Link Prediction":
            st.subheader("Link Prediction Comparison")

            selected_metrics = st.multiselect(
                "Select link prediction metrics:",
                [
                    "Common Neighbors",
                    "Jaccard Coefficient",
                    "Adamic/Adar index",
                    "Preferential Attachment",
                    "Resource Allocation Index",
                    "Shortest Path Distance",
                    "Hub Promoted Index",
                    "Personalized PageRank",
                ],
            )

            if selected_metrics:
                results = []

                # Convert the graph to undirected to avoid "not implemented for directed type" errors
                G_undirected = G.to_undirected()

                # Models for evaluation
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "XGBoost": xgb.XGBClassifier(
                        use_label_encoder=False, eval_metric="logloss"
                    ),
                }

                if "community_warning_displayed" not in st.session_state:
                    st.session_state["community_warning_displayed"] = False
                adamic_adar_warning_displayed = (
                    False  # Avoid multiple Adamic/Adar errors
                )

                for metric in selected_metrics:
                    # Prepare dataset
                    existing_edges = list(G_undirected.edges())
                    non_edges = list(nx.non_edges(G_undirected))[: len(existing_edges)]
                    data = existing_edges + non_edges
                    labels = [1] * len(existing_edges) + [0] * len(non_edges)

                    # Compute feature for metric
                    features = []
                    for u, v in data:
                        if metric == "Common Neighbors":
                            features.append(
                                len(list(nx.common_neighbors(G_undirected, u, v)))
                            )

                        elif metric == "Jaccard Coefficient":
                            features.append(
                                next(nx.jaccard_coefficient(G_undirected, [(u, v)]))[2]
                            )

                        elif metric == "Adamic/Adar index":
                            try:
                                adamic_adar_scores = nx.adamic_adar_index(
                                    G_undirected, data
                                )  # Ensure data is correctly formatted
                                features = [
                                    score if score is not None else 0
                                    for _, _, score in adamic_adar_scores
                                ]

                                if all(
                                    v == 0 for v in features
                                ):  # If all values are 0, avoid computation
                                    if not st.session_state.get(
                                        "adamic_warning", False
                                    ):
                                        st.error(
                                            "Adamic/Adar index produced no valid values. Try selecting different metrics."
                                        )
                                        st.session_state["adamic_warning"] = True
                                    continue  # Skip this metric

                            except Exception as e:
                                if not st.session_state.get("adamic_warning", False):
                                    st.error(f"Error computing Adamic/Adar index: {e}")
                                    st.session_state["adamic_warning"] = True
                                continue

                        elif metric == "Preferential Attachment":
                            features.append(
                                next(
                                    nx.preferential_attachment(G_undirected, [(u, v)])
                                )[2]
                            )

                        elif metric == "Resource Allocation Index":
                            features.append(
                                next(
                                    nx.resource_allocation_index(G_undirected, [(u, v)])
                                )[2]
                            )

                        elif metric == "Shortest Path Distance":
                            try:
                                sp_length = nx.shortest_path_length(
                                    G_undirected, source=u, target=v
                                )
                                features.append(sp_length)
                            except:
                                features.append(None)  # If undefined, handle gracefully

                        elif metric == "Hub Promoted Index":
                            # Ensure community information exists
                            if (
                                "community_dict" not in st.session_state
                                or not st.session_state["community_dict"]
                            ):
                                if not st.session_state.get(
                                    "community_warning_displayed", False
                                ):  # Only show the error once
                                    st.error(
                                        "No community information found. Please run community detection first."
                                    )
                                    st.session_state["community_warning_displayed"] = (
                                        True  # Prevent multiple messages
                                    )
                                continue

                            # Compute the metric properly
                            try:
                                # Ensure we pass the correctly formatted dictionary
                                hub_index_scores = nx.cn_soundarajan_hopcroft(
                                    G_undirected,
                                    data,
                                    community=st.session_state["community_dict"],
                                )
                                features.append(
                                    [
                                        score if score is not None else 0
                                        for _, _, score in hub_index_scores
                                    ]
                                )
                            except Exception as e:
                                if not st.session_state.get(
                                    "hub_index_error_displayed", False
                                ):  # Show error only once
                                    st.error(f"Error computing Hub Promoted Index: {e}")
                                    st.session_state["hub_index_error_displayed"] = True
                                continue

                        elif metric == "Personalized PageRank":
                            try:
                                pr = nx.pagerank(G_undirected, alpha=0.85)
                                features.append(pr[u] + pr[v])
                            except:
                                features.append(None)  # If undefined, handle gracefully

                    X = pd.DataFrame(features, columns=[metric])
                    y = pd.Series(labels)

                    if X.isna().all().all() or X.empty:
                        st.error(
                            f"Error: {metric} produced no valid values. Try selecting different metrics."
                        )
                        continue  # Skip this metric if it's empty

                    # Proceed only if X has valid features
                    X.fillna(0, inplace=True)  # Replace NaN with 0 if necessary

                    # Train/Test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                    # Store AUC scores for all models
                    auc_scores = {}

                    for model_name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred_prob = model.predict_proba(X_test)[:, 1]

                        auc_score = roc_auc_score(y_test, y_pred_prob)
                        auc_scores[model_name] = auc_score

                    results.append(
                        [
                            metric,
                            auc_scores["Logistic Regression"],
                            auc_scores["Random Forest"],
                            auc_scores["XGBoost"],
                        ]
                    )

                # Convert to DataFrame
                df_results = pd.DataFrame(
                    results,
                    columns=[
                        "Metric",
                        "Logistic Regression",
                        "Random Forest",
                        "XGBoost",
                    ],
                )
                df_results.set_index("Metric", inplace=True)

                # **Display Results in Table**
                st.write("### AUC Score Comparison Across Models")
                st.dataframe(df_results.style.format("{:.4f}"))

                # **Bar Plot (Grouped by Model)**
                if not df_results.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df_results.plot(kind="bar", ax=ax)

                    ax.set_ylabel("AUC Score")
                    ax.set_xlabel("Metric")  # Explicitly label x-axis
                    ax.set_title("Comparison of Link Prediction Metrics Across Models")
                    ax.legend(title="Model")
                    ax.set_xticklabels(
                        df_results.index, rotation=30, ha="right", fontsize=10
                    )

                    # Show plot
                    st.pyplot(fig)
                else:
                    st.warning(
                        "No valid data to plot. Ensure at least one metric returns valid results."
                    )

                # **Heatmap for Better Visualization**
                if not df_results.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        df_results,
                        annot=True,
                        cmap="coolwarm",
                        fmt=".3f",
                        linewidths=0.5,
                        ax=ax,
                    )
                    ax.set_title("Heatmap of AUC Scores Across Metrics and Models")
                    st.pyplot(fig)
                else:
                    st.warning(
                        "No valid data to plot. Ensure at least one metric returns valid results."
                    )
            else:
                st.warning("Please select at least one strategy to compare.")

        elif comparison_category == "Rumour Blocking":
            st.subheader("Rumour Blocking Comparison")

            # Allow users to compare multiple rumour blocking strategies
            selected_strategies = st.multiselect(
                "Select strategies to compare:",
                [
                    "Random Blocking",
                    "Highest Degree",
                    "Highest Betweenness Centrality",
                    "PageRank-Based Blocking",
                    "Structural Equivalence-Based Blocking",
                    "Structural Hole-Based Blocking",
                    "Adaptive Greedy Blocking",
                    "Community-Based Blocking",
                    "R-GRB (Efficient Randomized Blocking)",
                    "Hybrid Approach",
                ],
            )

            if selected_strategies:
                num_nodes = st.slider(
                    "Number of nodes to block:", min_value=1, max_value=10, value=3
                )

                # Store comparison results
                comparison_results = {}

                # Run each strategy
                for strategy in selected_strategies:
                    nodes_to_block = determine_nodes_to_block(G, strategy, num_nodes)
                    blocked_G = G.copy()
                    blocked_G.remove_nodes_from(nodes_to_block)

                    # Simulate rumour spread
                    model = ep.IndependentCascadesModel(blocked_G)
                    config = mc.Configuration()
                    config.add_model_parameter("fraction_infected", 0.05)
                    model.set_initial_status(config)
                    iterations = model.iteration_bunch(10)
                    infected_after = sum(iter["node_count"][1] for iter in iterations)

                    comparison_results[strategy] = infected_after

                # Convert dictionary to DataFrame (No Index Column)
                df_comparison = pd.DataFrame.from_dict(
                    comparison_results,
                    orient="index",
                    columns=["Infected Nodes After Blocking"],
                ).reset_index()  # Reset index to turn it into a column

                # Rename the "index" column properly to "Strategy"
                df_comparison.rename(columns={"index": "Strategy"}, inplace=True)

                # Sort the table in ascending order based on infections after blocking
                df_comparison = df_comparison.sort_values(
                    by="Infected Nodes After Blocking", ascending=False
                )

                # Dynamically adjust height based on the number of rows (but limit max height)
                table_height = min((len(df_comparison) * 35) + 50, 400)

                # Fix: Use `index=False` inside `st.dataframe()` to remove the leftmost index
                st.dataframe(
                    df_comparison, height=table_height, width=600, hide_index=True
                )

                # Visualization: Bar Chart with Strategy Names
                fig, ax = plt.subplots(figsize=(8, 5))

                # Set x-axis labels as strategy names instead of numerical indices
                df_comparison.plot(
                    kind="bar",
                    x="Strategy",  # Ensure Strategy names are used
                    y="Infected Nodes After Blocking",
                    color="green",
                    alpha=0.7,
                    legend=False,
                    ax=ax,
                )

                ax.set_ylabel("Infected Nodes After Blocking")
                ax.set_xlabel("")  # Remove redundant x-axis label
                ax.set_title("Comparison of Rumour Blocking Strategies")

                # Adjust x-axis labels
                ax.set_xticklabels(
                    df_comparison["Strategy"], rotation=30, ha="right"
                )  # Rotate for better visibility

                # Add value labels above bars with padding to prevent overlapping
                for i, v in enumerate(df_comparison["Infected Nodes After Blocking"]):
                    ax.text(
                        i,
                        v
                        + max(df_comparison["Infected Nodes After Blocking"])
                        * 0.01,  # Adjust vertical placement
                        str(int(v)),
                        ha="center",
                        fontsize=10,
                        fontweight="bold",
                    )

                # Show Plot in Streamlit
                st.pyplot(fig)
            else:
                st.warning("Please select at least one strategy to compare.")

        elif comparison_category == "Community Detection":
            st.subheader("Community Detection Comparison")

            available_algorithms = [
                "Louvain",
                "Girvan-Newman",
                "Leiden",
                "Label Propagation",
                "Spectral Clustering",
                "Edge Betweenness",
                "Walktrap",
            ]

            # Let user select which algorithms to compare
            selected_algorithms = st.multiselect(
                "Select algorithms to compare:", available_algorithms
            )

            if selected_algorithms:
                community_results = {}

                for algorithm in selected_algorithms:
                    if algorithm == "Louvain":
                        communities = list(nx.community.louvain_communities(G))
                    elif algorithm == "Girvan-Newman":
                        comp = nx.community.girvan_newman(G)
                        communities = tuple(sorted(c) for c in next(comp))
                    elif algorithm == "Leiden":
                        partition = leiden(G)
                        communities = [set(comm) for comm in partition.communities]

                    elif algorithm == "Label Propagation":
                        communities = list(asyn_lpa_communities(G))

                    elif algorithm == "Spectral Clustering":
                        adjacency_matrix = nx.to_numpy_array(G)
                        spectral = SpectralClustering(
                            n_clusters=2,
                            affinity="precomputed",
                            assign_labels="discretize",
                        )
                        labels = spectral.fit_predict(adjacency_matrix)

                        community_dict = {}
                        for i, node in enumerate(G.nodes()):
                            label = labels[i]
                            if label not in community_dict:
                                community_dict[label] = set()
                            community_dict[label].add(node)

                        communities = list(community_dict.values())

                    elif algorithm == "Edge Betweenness":
                        comp = nx.community.girvan_newman(G)
                        communities = tuple(sorted(c) for c in next(comp))

                    elif algorithm == "Walktrap":
                        communities = optimal_walktrap_networkx(G, max_clusters=10)

                    # Calculate modularity score for each algorithm
                    modularity_score = nx.community.quality.modularity(G, communities)
                    community_results[algorithm] = modularity_score

                # Convert results to DataFrame
                df_community = pd.DataFrame.from_dict(
                    community_results, orient="index", columns=["Modularity Score"]
                )

                # Sort results
                df_community.sort_values(
                    by="Modularity Score", ascending=False, inplace=True
                )

                # **Display Table with Proper Formatting**
                styled_df = df_community.style.set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("text-align", "center"),
                                ("font-weight", "bold"),
                                ("padding", "8px"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [
                                ("text-align", "center"),
                                ("padding", "8px 20px"),
                                ("white-space", "nowrap"),
                            ],
                        },
                    ]
                ).format(
                    precision=4
                )  # Ensure uniform decimal places

                st.dataframe(styled_df, height=len(df_community) * 35 + 35, width=500)

                # **Visualization: Bar Chart**
                fig, ax = plt.subplots(figsize=(8, 5))
                df_community.plot(kind="bar", color="blue", alpha=0.7, ax=ax)

                # **Visualization: Bar Chart**
                fig, ax = plt.subplots(figsize=(8, 5))
                df_community.plot(
                    kind="bar", color="blue", alpha=0.7, ax=ax, legend=False
                )

                # Compute a dynamic offset for label placement
                y_max = df_community[
                    "Modularity Score"
                ].max()  # Get the maximum modularity score
                offset = y_max * 0.01  # Adjust text positioning dynamically

                # Add Value Labels Above Bars with Dynamic Positioning
                for i, v in enumerate(df_community["Modularity Score"]):
                    ax.text(
                        i,
                        v + offset,  # Dynamically place text slightly above the bar
                        f"{v:.4f}",
                        ha="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                    )

                ax.set_ylabel("Modularity Score")
                ax.set_title("Comparison of Community Detection Algorithms")

                # Adjust x-axis labels for readability
                ax.set_xticklabels(df_community.index, rotation=30, ha="right")

                st.pyplot(fig)

            else:
                st.warning("Please select at least one algorithm to compare.")
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")

elif page == "Finance & Business Applications":
    # st.title("Finance & Business Network Analysis")

    st.write(
        """
    Social Network Analysis (SNA) has significant applications in finance and business.
    By analyzing relationships and interactions, we can identify influential entities,
    predict future links, and prevent fraud or misinformation.
    """
    )

    st.subheader("Key Network Metrics & Financial Applications")

    metrics_explanation = {
        "Degree Centrality": "Identifies key financial influencers (e.g., big investors, high-volume traders).",
        "Betweenness Centrality": "Detects brokers or intermediaries in financial transactions.",
        "Closeness Centrality": "Finds entities that can spread financial information the fastest.",
        "Eigenvector Centrality": "Measures the importance of financial entities, similar to Google's PageRank.",
        "Community Detection": "Groups related entities (e.g., market sectors, investor groups).",
        "Link Prediction": "Predicts future financial relationships (e.g., mergers, collaborations).",
        "Rumour Blocking": "Helps prevent financial misinformation and stock market manipulations.",
    }

    for metric, explanation in metrics_explanation.items():
        st.markdown(f"**{metric}:** {explanation}")

    # Upload Financial Network Data
    G_financial = st.session_state["graph"]

    # Visualization
    if "graph" in st.session_state and st.session_state["graph"] is not None:
        st.subheader("Visualizing the Financial Network")
        visualize_graph(G_financial, num_nodes_to_display=200)

        st.subheader("Select Financial Network Metrics to Analyze")

        selected_financial_metrics = st.multiselect(
            "Choose metrics:", list(metrics_explanation.keys())
        )

        if selected_financial_metrics:
            st.write("## Financial Network Metrics Analysis")

            if "Degree Centrality" in selected_financial_metrics:
                display_centrality_measures(G_financial, ["Degree Centrality"])

            if "Betweenness Centrality" in selected_financial_metrics:
                display_centrality_measures(G_financial, ["Betweenness Centrality"])

            if "Closeness Centrality" in selected_financial_metrics:
                display_centrality_measures(G_financial, ["Closeness Centrality"])

            if "Eigenvector Centrality" in selected_financial_metrics:
                display_centrality_measures(G_financial, ["Eigenvector Centrality"])

            if "Community Detection" in selected_financial_metrics:
                communities = list(nx.community.louvain_communities(G_financial))
                visualize_communities(G_financial, communities)

            if "Link Prediction" in selected_financial_metrics:
                st.write("Predicting potential future financial connections...")

                G_financial = G_financial.to_undirected()

                # Get possible non-edges (node pairs that are not connected)
                data = list(nx.non_edges(G_financial))[
                    :1000
                ]  # Limit to 1000 pairs for efficiency

                # Compute link prediction features
                df_link_prediction = compute_link_prediction_features(
                    G_financial,
                    data,
                    [
                        "Common Neighbors",
                        "Jaccard Coefficient",
                        "Adamic/Adar index",
                        "Preferential Attachment",
                        "Resource Allocation Index",
                        "Shortest Path Distance",
                        "Hub Promoted Index",
                        "Personalized PageRank",
                    ],
                )

                # Display results
                st.dataframe(df_link_prediction.head(10))

            if "Rumour Blocking" in selected_financial_metrics:
                st.write("Exploring financial rumour control...")
                rumour_source = identify_rumour_source(G_financial)
                st.write(
                    f"**Potential Financial Rumour Source:** Node `{rumour_source}`"
                )

                blocking_strategy = st.selectbox(
                    "Select a blocking strategy:",
                    [
                        "Random Blocking",
                        "Highest Degree",
                        "Highest Betweenness Centrality",
                        "PageRank-Based Blocking",
                        "Structural Equivalence-Based Blocking",
                        "Structural Hole-Based Blocking",
                        "Adaptive Greedy Blocking",
                        "Community-Based Blocking",
                        "R-GRB (Efficient Randomized Blocking)",
                        "Hybrid Approach",
                    ],
                )
                num_nodes = st.slider(
                    "Number of nodes to block:", min_value=1, max_value=10, value=3
                )
                if st.button("Apply Financial Rumour Blocking"):
                    nodes_to_block = determine_nodes_to_block(
                        G_financial, blocking_strategy, num_nodes
                    )

                    if not nodes_to_block:
                        st.warning(
                            "No nodes selected for blocking. Try selecting a different strategy or increasing the number of nodes."
                        )
                    else:
                        blocked_G_financial = apply_rumour_blocking(
                            G_financial, nodes_to_block
                        )
                        st.session_state["financial_blocked_graph"] = (
                            blocked_G_financial
                        )

                        # Display visualization
                        fig = visualize_blocked_graph(
                            G_financial, nodes_to_block, hide_blocked_edges=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.success(
                            f"Successfully applied {blocking_strategy} strategy to block {num_nodes} nodes."
                        )
    else:
        st.warning("No financial network data uploaded. Please upload a dataset.")
else:
    st.warning("Please upload a graph file to proceed.")
