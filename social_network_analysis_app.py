import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform
import tempfile
import networkx as nx
import pyvis.network as pyvis_net
from streamlit.components.v1 import iframe
import os
import base64
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import random

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


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
    plt.plot(X, Y, linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency')
    plt.title('In-Degree Distribution (Log-Log Scale)')
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
    plt.plot(X, Y, linestyle='-', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Out-Degree')
    plt.ylabel('Frequency')
    plt.title('Out-Degree Distribution (Log-Log Scale)')
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit

# Function to visualize the graph
# def visualize_graph(G):
#     plt.clf()  # Clear the current figure to prevent overlap
#     plt.figure(figsize=(10, 8))
#     nx.draw(G, with_labels=True, node_size=50, font_size=8, edge_color="gray", node_color="blue", alpha=0.7)
#     plt.title('Network Graph Visualization')
#     st.pyplot(plt.gcf())  # Show the current figure in Streamlit
# Function to visualize the graph using Plotly

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
    important_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes_to_display]
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
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
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
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',  # Only display information on hover
        marker=dict(
            showscale=True,
            colorscale='Blues',
            size=fixed_node_size,  # Set fixed size
            color=node_colors,
            colorbar=dict(
                title="Node Degree",
                thickness=15,
                xanchor='left',
                titleside='right'
            ),
            line_width=1
        ),
        text=[f"Node {node_id}<br>Degree: {degree}" for node_id, degree in zip(node_ids, node_colors)]  # Hover content
    )

    # Create interactive figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Interactive Network Graph (Top {num_nodes_to_display} Nodes by Degree)",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
    )

    # Display interactive graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)



# Function to display global network statistics
def display_global_statistics(G):
    undirected_G = G.to_undirected()  # Convert to undirected graph for triangle calculations

    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())
    max_degree = int(max(dict(G.degree()).values()))
    avg_degree = int(sum(dict(G.degree()).values()) / num_nodes)
    assortativity = nx.degree_assortativity_coefficient(G)
    num_triangles = int(sum(nx.triangles(undirected_G).values()) // 3)
    avg_triangles = num_triangles / num_edges if num_edges > 0 else 0
    max_triangles = int(max(nx.triangles(undirected_G).values()) if num_nodes > 0 else 0)
    avg_clustering_coeff = nx.average_clustering(undirected_G)
    global_clustering_coeff = nx.transitivity(undirected_G)
    max_k_core = int(max(nx.core_number(undirected_G).values()) if num_nodes > 0 else 0)
    lower_bound_max_clique = int(len(max(nx.find_cliques(undirected_G), key=len)) if num_nodes > 0 else 0)

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

# def embedding_similarity(u, v):
#     node_embeddings = compute_node_embeddings(G)
#     return cosine_similarity(node_embeddings[u].reshape(1, -1), node_embeddings[v].reshape(1, -1))[0][0]



def link_prediction_features(G, metrics):
    """Generate link prediction features based on selected metrics."""
    if G.is_directed():
        st.warning("The uploaded graph is directed. Converting it to an undirected graph for link prediction analysis.")
        undirected_G = G.to_undirected()
    else:
        undirected_G = G

    non_edges = list(nx.non_edges(undirected_G))
    results = []

    if 'Common Neighbors' in metrics:
        common_neighbors = compute_common_neighbors(undirected_G, non_edges)
        df_cn = pd.DataFrame(common_neighbors, columns=['Node1', 'Node2', 'CommonNeighbors'])
        results.append(df_cn)

    if 'Jaccard Coefficient' in metrics:
        jaccard_coefficient = compute_jaccard_coefficient(undirected_G, non_edges)
        df_jc = pd.DataFrame(jaccard_coefficient, columns=['Node1', 'Node2', 'JaccardCoefficient'])
        results.append(df_jc)

    if 'Adamic/Adar index' in metrics:
        adamic_adar = compute_adamic_adar_index(undirected_G, non_edges)
        df_aa = pd.DataFrame(adamic_adar, columns=['Node1', 'Node2', 'AdamicAdarIndex'])
        results.append(df_aa)

    if 'Preferential Attachment' in metrics:
        preferential_attachment = list(nx.preferential_attachment(undirected_G, non_edges))
        df_pa = pd.DataFrame(preferential_attachment, columns=['Node1', 'Node2', 'PreferentialAttachment'])
        results.append(df_pa)
    # if 'Resource Allocation Index' in metrics:
    #     rai = list(nx.resource_allocation_index(G, non_edges))
    #     df_rai = pd.DataFrame(rai, columns=['Node1', 'Node2', 'ResourceAllocationIndex'])
    #     results.append(df_rai)
    # if 'Attribute Similarity' in metrics:
    #     attr_sim = [(u, v, attribute_similarity(G, u, v, attribute="attribute_name")) for u, v in data]
    #     df_attr = pd.DataFrame(attr_sim, columns=['Node1', 'Node2', 'AttributeSimilarity'])
    #     # results.append(df_attr)

    # if 'Embedding-Based Similarity' in metrics:
    #     embedding_sim = [(u, v, embedding_similarity(u, v)) for u, v in data]
    #     df_embed = pd.DataFrame(embedding_sim, columns=['Node1', 'Node2', 'EmbeddingSimilarity'])
    #     results.append(df_embed)

    # if 'Temporal Interaction Frequency' in metrics:
    #     temp_freq = [(u, v, interaction_frequency(G, u, v, interaction_data={})) for u, v in data]
    #     df_temp = pd.DataFrame(temp_freq, columns=['Node1', 'Node2', 'TemporalFrequency'])
    #     results.append(df_temp)
    if results:
        return pd.concat(results, axis=1).loc[:, ~pd.concat(results, axis=1).columns.duplicated()]
    else:
        return pd.DataFrame()

def compute_link_prediction_features(G, data, selected_metrics):
    feature_dict = {'Node1': [u for u, v in data], 'Node2': [v for u, v in data]}  # Track node pairs

    if 'Common Neighbors' in selected_metrics:
        feature_dict['CommonNeighbors'] = [len(list(nx.common_neighbors(G, u, v))) for u, v in data]

    if 'Jaccard Coefficient' in selected_metrics:
        jaccard_scores = nx.jaccard_coefficient(G, data)
        feature_dict['JaccardCoefficient'] = [score for _, _, score in jaccard_scores]

    if 'Adamic/Adar index' in selected_metrics:
        adamic_adar_scores = nx.adamic_adar_index(G, data)
        feature_dict['AdamicAdarIndex'] = [score for _, _, score in adamic_adar_scores]

    if 'Preferential Attachment' in selected_metrics:
        pref_attachment_scores = nx.preferential_attachment(G, data)
        feature_dict['PreferentialAttachment'] = [score for _, _, score in pref_attachment_scores]

    return pd.DataFrame(feature_dict)

def apply_rumor_blocking(G, selected_nodes):
    # Remove selected nodes to simulate rumor blocking
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

        if hide_blocked_edges and (edge[0] in blocked_nodes or edge[1] in blocked_nodes):
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
        x=normal_edge_x, y=normal_edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    blocked_edge_trace = go.Scatter(
        x=blocked_edge_x, y=blocked_edge_y,
        line=dict(width=0.5, color='red'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace
    node_x, node_y, node_colors, node_ids = [], [], [], []

    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))

        # Color nodes based on whether they are blocked or not
        node_colors.append('red' if node in blocked_nodes else 'skyblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(color=node_colors, size=8, line_width=1),
        text=[f"Node {node_id}" for node_id in node_ids]
    )

    # Combine the traces and display
    fig = go.Figure(data=[normal_edge_trace, blocked_edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Graph with Fixed Blocked Edges",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig


def visualize_communities(G, communities, positions=None):
    if not positions:
        positions = nx.spring_layout(G, seed=42)  # Spring layout with a consistent seed

    # Assign colors to communities
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(communities))]
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
        hover_text.append(f"Node {node}, Community {list(color_map.values()).index(color_map[node]) + 1}")

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            color=node_color,
            size=10,  # Fixed node size for consistency
            line_width=0.5
        ),
        text=hover_text,
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Community Detection Visualization",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    st.plotly_chart(fig)


# Function to determine which nodes to block based on the strategy
def determine_nodes_to_block(G, strategy, k):
    if strategy == "Highest Degree":
        return [node for node, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:k]]
    elif strategy == "Highest Betweenness Centrality":
        centrality = nx.betweenness_centrality(G)
        return [node for node, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:k]]
    elif strategy == "PageRank-Based Blocking":
        pagerank = nx.pagerank(G)
        return [node for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:k]]
    elif strategy == "Random Blocking":
        return random.sample(list(G.nodes()), min(k, len(G.nodes())))
    return []



# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if "graph" not in st.session_state:
    st.session_state["graph"] = None

# Define Navigation Items
nav_items = ["Home", "Link Prediction", "Rumor Blocking", "Community Detection", "Compare Methods"]

# Create Columns for Navigation (5 equal parts)
cols = st.columns(len(nav_items))

# Set a uniform button width using Markdown and CSS
button_style = """
    <style>
        div.stButton > button {
            width: 100%;
            height: 50px;
            font-weight: bold;
            font-size: 14px;
            border-radius: 8px;
            transition: background-color 0.3s, color 0.3s;
            white-space: normal !important; /* Allows text wrapping */
            word-wrap: break-word !important;
            line-height: 1.2em; /* Adjust line height for readability */
            padding: 8px 5px; /* Reduce padding to avoid text clipping */
        }
        .selected {
            background-color: #FF4B4B !important;
            color: white !important;
            border: 2px solid #FF4B4B !important;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Create Buttons in Each Column
for i, item in enumerate(nav_items):
    with cols[i]:
        if st.button(item, key=item):
            st.session_state["page"] = item  # Update session state for navigation

# Highlight the active page using Markdown
selected_page = st.session_state["page"]
st.markdown(f'<style>div.stButton > button[data-testid="stButton-{selected_page}"] {{ background-color: #FF4B4B; color: white; }}</style>', unsafe_allow_html=True)



# Display file uploader globally
uploaded_file = st.file_uploader("Upload your graph file", type=["txt", "mtx"])
if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file

    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Initialize the graph
    G = nx.DiGraph()  # DiGraph for directed, Graph for undirected
    try:
        # Check the file type and load the graph accordingly
        if uploaded_file.name.endswith('.txt'):
            G = nx.read_edgelist(temp_file_path, nodetype=int, create_using=nx.DiGraph())
        elif uploaded_file.name.endswith('.mtx'):
            matrix = mmread(temp_file_path)
            row, col = matrix.nonzero()
            unique_nodes = set(row).union(set(col))
            for node in unique_nodes:
                G.add_node(int(node))
            for i, j in zip(row, col):
                if i != j:
                    G.add_edge(int(i), int(j))

        st.success("Graph successfully loaded!")
        st.session_state["graph"] = G  # Store the graph in session state

    except Exception as e:
        st.error(f"An error occurred while processing the graph: {e}")

# Current page selection
page = st.session_state["page"]

# Page-specific content
if page == "Home":
    st.title("Social Network Analysis Tool")
    st.write("Upload your graph to explore network statistics.")
    # st.write("Session State Debug:", st.session_state)

    # Ensure both file is uploaded and graph is successfully stored
    if "graph" in st.session_state and st.session_state["graph"] is not None:
        st.write("Graph is loaded and available for analysis.")

        # Dropdown menu to select metrics, now including in-degree and out-degree distribution
        metrics_options = [
            'Display Global Statistics',
            "Network Graph Visualization",
            "In-Degree Distribution",
            "Out-Degree Distribution",
            "Degree Centrality",
            "In Degree Centrality",
            "Out Degree Centrality",
            "Closeness Centrality",
            "Current Flow Closeness(Information) Centrality",
            "Betweenness Centrality",
            "Edge Betweenness Centrality",
            "Communicability Betweenness Centrality",
            "Eigenvector Centrality",
            "PageRank Centrality",
            "Katz Centrality"
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
            "Transitivity"

        ]
        selected_metrics = st.multiselect("Select metrics to display:", metrics_options)

        # Display selected metrics
        if selected_metrics:
            st.write("## Selected Graph Metrics")
            if "In-Degree Distribution" in selected_metrics:
                plot_in_degree_distribution(G)

            if "Out-Degree Distribution" in selected_metrics:
                plot_out_degree_distribution(G)

            if "Degree Centrality" in selected_metrics:
                degree_centrality = nx.degree_centrality(G)
                st.write("**Degree Centrality:**", degree_centrality)

            if "In-Degree Centrality" in selected_metrics:

                in_degree_centrality = nx.in_degree_centrality(G)
                st.write("**In-Degree Centrality:**", in_degree_centrality)

            if "Out-Degree Centrality" in selected_metrics:
                out_degree_centrality = nx.out_degree_centrality(G)
                st.write("**Out-Degree Centrality:**", out_degree_centrality)

            if "Closeness Centrality" in selected_metrics:
                closeness_centrality = nx.closeness_centrality(G)
                st.write("**Closeness Centrality:**", closeness_centrality)

            if "Current Flow Closeness(Information) Centrality" in selected_metrics:
                current_flow_closeness_centrality = nx.current_flow_closeness_centrality(G)
                st.write("**Current Flow Closeness Centrality:**", current_flow_closeness_centrality)

            if "Communicability Betweenness Centrality" in selected_metrics:
                communicability_betweenness_centrality = nx.communicability_betweenness_centrality(G)
                st.write("**Betweenness Centrality:**", communicability_betweenness_centrality)

            if "Edge Betweenness Centrality" in selected_metrics:
                edge_betweenness_centrality = nx.edge_betweenness_centrality(G)
                st.write("**Edge Betweenness Centrality:**", edge_betweenness_centrality)

            if "Betweenness Centrality" in selected_metrics:
                betweenness_centrality = nx.betweenness_centrality(G)
                st.write("**Betweenness Centrality:**", betweenness_centrality)

            if "Katz Centrality" in selected_metrics:
                katz = nx.katz_centrality(G)
                st.write("**Katz Centrality:**", katz)

            if "PageRank Centrality" in selected_metrics:
                pagerank = nx.pagerank(G, alpha=0.85)
                st.write("**PageRank Centrality:**", pagerank)

            if "Clustering Coefficients" in selected_metrics:
                clustering_coefficients = nx.clustering(G)
                st.write("**Clustering Coefficients:**", clustering_coefficients)

            if "Diameter" in selected_metrics:
                if nx.is_connected(G.to_undirected()):
                    diameter = nx.diameter(G.to_undirected())
                else:
                    diameter = "Graph not connected"
                st.write("**Diameter:**", diameter)

            if "Average Shortest Path Length" in selected_metrics:
                if nx.is_connected(G.to_undirected()):
                    avg_shortest_path_length = nx.average_shortest_path_length(G.to_undirected())
                else:
                    avg_shortest_path_length = "Graph not connected"
                st.write("**Average Shortest Path Length:**", avg_shortest_path_length)

            if "Triangles per Node" in selected_metrics:
                triangles = nx.triangles(G.to_undirected())
                st.write("**Number of Triangles per Node:**", triangles)

            if "Eigenvector Centrality" in selected_metrics:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                st.write("**Eigenvector Centrality:**", eigenvector_centrality)

            if "Display Global Statistics" in selected_metrics:
                display_global_statistics(G)

            if "Network Graph Visualization" in selected_metrics:
                st.write("### Network Graph Visualization (Subsampled)")
                visualize_graph(G, num_nodes_to_display=200)

            if "Fragmentation" in selected_metrics:
                st.write("### Fragmentation")
                # Calculate reachable pairs
                connected_components = nx.connected_components(G.to_undirected())
                total_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
                reachable_pairs = sum(len(c) * (len(c) - 1) for c in connected_components)

                # Calculate fragmentation
                fragmentation = 1 - (reachable_pairs / total_pairs if total_pairs > 0 else 1)
                st.write(f"**Fragmentation:** {fragmentation:.4f}")

            if "Geodesic Distance" in selected_metrics:
                st.write("### Geodesic Distance")
                try:
                    # Compute shortest paths
                    path_lengths = dict(nx.shortest_path_length(G))
                    distances = [
                        length for source, targets in path_lengths.items() for target, length in targets.items()
                    ]
                    avg_distance = np.mean(distances)
                    max_distance = np.max(distances)
                    st.write(f"**Average Geodesic Distance:** {avg_distance:.4f}")
                    st.write(f"**Maximum Geodesic Distance:** {max_distance}")
                except nx.NetworkXError:
                    st.warning("Geodesic distance cannot be calculated for disconnected graphs.")

            # Handle Isolates
            if "Isolates" in selected_metrics:
                st.write("### Isolates")
                isolates = list(nx.isolates(G))
                if isolates:
                    st.write(f"**Isolates:** {len(isolates)} nodes are disconnected from the graph.")
                    st.write(f"Isolated nodes: {isolates}")
                else:
                    st.write("No isolates in the graph.")

            if "Small-Worldness" in selected_metrics:
                st.write("### Small-Worldness")

                try:
                    # Compute clustering coefficient and average path length of the network
                    clustering_coeff = nx.average_clustering(G.to_undirected())
                    if nx.is_connected(G.to_undirected()):
                        avg_path_length = nx.average_shortest_path_length(G.to_undirected())
                    else:
                        avg_path_length = float('inf')  # Not applicable if the graph is disconnected

                    # Generate a random graph with the same number of nodes and edges
                    random_graph = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
                    clustering_coeff_rand = nx.average_clustering(random_graph)
                    if nx.is_connected(random_graph):
                        avg_path_length_rand = nx.average_shortest_path_length(random_graph)
                    else:
                        avg_path_length_rand = float('inf')  # Handle disconnected random graph

                    # Calculate Small-Worldness
                    if avg_path_length_rand > 0 and clustering_coeff_rand > 0:
                        small_worldness = (clustering_coeff / clustering_coeff_rand) / (avg_path_length / avg_path_length_rand)
                        st.write(f"**Small-Worldness:** {small_worldness:.4f}")
                        st.write(f"**Clustering Coefficient (Network):** {clustering_coeff:.4f}")
                        st.write(f"**Clustering Coefficient (Random Graph):** {clustering_coeff_rand:.4f}")
                        st.write(f"**Average Path Length (Network):** {avg_path_length:.4f}")
                        st.write(f"**Average Path Length (Random Graph):** {avg_path_length_rand:.4f}")
                    else:
                        st.warning("Small-Worldness cannot be calculated due to disconnected components in the random graph.")
                except Exception as e:
                    st.error(f"An error occurred while calculating Small-Worldness: {e}")

            # Handle Structural Equivalence
            if "Structural Equivalence" in selected_metrics:
                st.write("### Structural Equivalence")

                # Convert the adjacency matrix
                adjacency_matrix = nx.to_numpy_array(G)

                # Calculate pairwise similarity (1 - normalized Euclidean distance)
                similarity_matrix = 1 - squareform(pdist(adjacency_matrix, metric='euclidean'))

                # Display results
                st.write("**Pairwise Structural Equivalence Similarity Matrix:**")
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_matrix, cmap="viridis")
                plt.title("Structural Equivalence Similarity Matrix")
                st.pyplot(plt)

            if "Structural Holes" in selected_metrics:
                st.write("### Structural Holes")
                effective_sizes = nx.effective_size(G)
                # Display results
                st.write("**Effective Sizes (Structural Holes):**")
                st.write(pd.DataFrame.from_dict(effective_sizes, orient='index', columns=['Effective Size']))

            if "Transitivity" in selected_metrics:
                st.write("### Transitivity")

                # Compute transitivity
                transitivity_value = nx.transitivity(G)

                # Display results
                st.write(f"**Global Transitivity (Clustering Coefficient):** {transitivity_value:.4f}")

        # Endogeneity section
        # st.write("### Endogeneity Analysis")
        # display_endogeneity_properties(G)

    else:
        st.warning("No graph loaded. Please upload a valid graph file.")


elif page == "Link Prediction":
    st.title("Link Prediction")

    if st.session_state["graph"]:
        G = st.session_state["graph"]
        # Convert to undirected graph
        undirected_G = G.to_undirected()

        # Existing link prediction logic
        # st.write("Perform link prediction using various network metrics.")
        st.subheader("Link Prediction Metrics & Model Evaluation")

        # Available link prediction metrics
        link_prediction_metrics_options = [
            'Common Neighbors',
            'Jaccard Coefficient',
            'Adamic/Adar index',
            'Preferential Attachment',
        ]
        selected_metrics = st.multiselect("Select metrics to display:", link_prediction_metrics_options)

        if selected_metrics:
            # Prepare dataset for training
            existing_edges = list(undirected_G.edges())
            non_edges = list(nx.non_edges(undirected_G))
            non_edges_sample = non_edges[:len(existing_edges)]  # Balance positive and negative samples

            # Create a dataset with labels (1 for existing edges, 0 for non-edges)
            data = existing_edges + non_edges_sample
            labels = [1] * len(existing_edges) + [0] * len(non_edges_sample)

            # Compute link prediction features based on selected metrics
            link_features_df = compute_link_prediction_features(undirected_G, data, selected_metrics)
            st.write("**Extracted Link Prediction Features:**")
            st.dataframe(link_features_df.head(10))

            if st.button("Train Link Prediction Model and Plot AUC"):
                st.write("Training link prediction model...")

                # Train/Test split
                X_train, X_test, y_train, y_test = train_test_split(link_features_df.drop(['Node1', 'Node2'], axis=1), labels, test_size=0.3, random_state=42)

                # Train logistic regression model
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve


                # Compute Evaluation Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)



                # **Baseline Performance: Random Prediction**
                random_probs = np.random.rand(len(y_test))
                auc_random = auc(*roc_curve(y_test, random_probs)[:2])

                # **Precision@K & Recall@K**
                precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
                k = 10  # Example: Evaluate top 10 link predictions
                precision_at_k = precisions[k]
                recall_at_k = recalls[k]

                # **Display Metrics in a Table**
                metrics = [
                    ["Accuracy", f"{accuracy:.2f}", "Higher is better"],
                    ["Precision", f"{precision:.2f}", "Higher means better positive link detection"],
                    ["Recall", f"{recall:.2f}", "Higher means more links were detected"],
                    ["F1-Score", f"{f1:.2f}", "Balance between Precision & Recall"],
                    ["AUC Score", f"{auc(roc_curve(y_test, y_pred_prob)[0], roc_curve(y_test, y_pred_prob)[1]):.2f}", "Higher means better model"],
                    ["Baseline AUC (Random)", f"{auc_random:.2f}", "Reference for comparison"],
                    [f"Precision at {k}", f"{precision_at_k:.2f}", "Precision for Top-10 predictions"],
                    [f"Recall at {k}", f"{recall_at_k:.2f}", "Recall for Top-10 predictions"],
                ]
                df_metrics = pd.DataFrame(metrics, columns=["Metric", "Value", "Interpretation"])
                st.dataframe(df_metrics)

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                auc_score = auc(fpr, tpr)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle="--", label="Random (Baseline)")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Receiver Operating Characteristic (ROC) Curve")
                ax.legend()
                st.pyplot(fig)

                # **Plot Precision-Recall Curve**
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(recalls, precisions, color='red', label="Precision-Recall Curve")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                ax.legend()
                st.pyplot(fig)
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")

# Rumor Blocking Page
elif page == "Rumor Blocking":
    st.title("Rumor Blocking")

    if st.session_state["graph"]:
        G = st.session_state["graph"]
        st.write("Interactively block nodes and visualize the affected areas of the network.")

        # Dropdown to select blocking strategy
        blocking_strategy = st.selectbox(
            "Select a blocking strategy:",
            ["Highest Degree", "Highest Betweenness Centrality", "PageRank-Based Blocking", "Random Blocking"]
        )

        # Number of nodes to block
        num_nodes = st.slider("Number of nodes to block:", min_value=1, max_value=10, value=3)



        # Nodes selected via algorithm
        nodes_to_block = determine_nodes_to_block(G, blocking_strategy, num_nodes)

        # Show the selected nodes
        st.write(f"**Nodes selected for blocking ({blocking_strategy}):** {nodes_to_block}")

        # Toggle to hide blocked edges
        hide_edges = st.checkbox("Hide edges connected to blocked nodes", value=False)

        if st.button("Apply Rumor Blocking"):
            if not nodes_to_block:
                st.warning("Please select at least one node to block.")
            else:
                # Create a new blocked graph (instead of using visualization function)
                blocked_G = G.copy()
                blocked_G.remove_nodes_from(nodes_to_block)  # Remove blocked nodes

                # Visualize the graph with blocked nodes and edges highlighted
                fig = visualize_blocked_graph(G, nodes_to_block, hide_blocked_edges=hide_edges)
                st.plotly_chart(fig, use_container_width=True)

                # ------------------  EVALUATION SECTION ------------------
                st.subheader("Evaluation Metrics After Rumor Blocking")
                blocked_G_undirected = blocked_G.to_undirected()

                # Simulating Rumor Spread
                infection_results = {}
                # Function to run the epidemic model and store results
                def run_epidemic_model(model_class, model_name, G, blocked_G, config_params):
                    """Runs an epidemic model and stores infection counts."""
                    model = model_class(G)
                    config = mc.Configuration()
                    for param, value in config_params.items():
                        config.add_model_parameter(param, value)
                    model.set_initial_status(config)
                    iterations_before = model.iteration_bunch(10)
                    infected_before = sum(iter['node_count'][1] for iter in iterations_before)
                    peak_before = max(iter['node_count'][1] for iter in iterations_before)

                    model = model_class(blocked_G)
                    model.set_initial_status(config)
                    iterations_after = model.iteration_bunch(10)
                    infected_after = sum(iter['node_count'][1] for iter in iterations_after)
                    peak_after = max(iter['node_count'][1] for iter in iterations_after)

                    infection_results[model_name] = {
                        "infected_before": infected_before,
                        "infected_after": infected_after,
                        "peak_before": peak_before,
                        "peak_after": peak_after,
                        "iterations_before": iterations_before,
                        "iterations_after": iterations_after
                    }

                # Run models and store results
                run_epidemic_model(ep.IndependentCascadesModel, "IC Model", G, blocked_G, {"fraction_infected": 0.05})
                run_epidemic_model(ep.SIModel, "SI Model", G, blocked_G, {"beta": 0.01})
                run_epidemic_model(ep.SIRModel, "SIR Model", G, blocked_G, {"beta": 0.01, "gamma": 0.005})
                run_epidemic_model(ep.ThresholdModel, "LT Model", G, blocked_G, {"fraction_infected": 0.05})

                # Storing Evaluation Metrics
                metrics = []
                # 1. Number of Connected Components (Should Increase)
                before_components = nx.number_connected_components(G.to_undirected())
                after_components = nx.number_connected_components(blocked_G_undirected)
                metrics.append(["Connected Components", before_components, after_components, "Higher means more fragmentation"])

                # 2.Average Shortest Path Length (Largest Component Only)
                # Convert the graph to undirected
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                subgraph_before = G.subgraph(largest_cc).to_undirected()

                # Compute average shortest path length only if the graph is connected
                avg_path_before = nx.average_shortest_path_length(subgraph_before) if nx.is_connected(subgraph_before) else np.nan

                largest_cc_after = max(nx.connected_components(blocked_G.to_undirected()), key=len)
                subgraph_after = blocked_G.subgraph(largest_cc_after).to_undirected()
                avg_path_after = nx.average_shortest_path_length(subgraph_after) if nx.is_connected(subgraph_after) else np.nan

                metrics.append(["Avg. Shortest Path (Largest Component)", round(avg_path_before, 2), round(avg_path_after, 2), "Higher means rumor spreads slower"])

                # Size of Largest Connected Component
                largest_cc_size_before = len(max(nx.connected_components(G.to_undirected()), key=len))
                largest_cc_size_after = len(max(nx.connected_components(blocked_G.to_undirected()), key=len))
                metrics.append(["Largest Connected Component Size", largest_cc_size_before, largest_cc_size_after, "Lower means rumor containment was effective"])

                #3. Eigenvector Centrality Reduction(Rumor Influence) (Should Decrease)
                before_centrality = sum(nx.eigenvector_centrality_numpy(G, max_iter=1000).values())
                after_centrality = sum(nx.eigenvector_centrality_numpy(blocked_G, max_iter=1000).values())
                metrics.append(["Eigenvector Centrality", round(before_centrality, 2), round(after_centrality, 2), "Lower means less influence"])

                # 4. Mean Infection Time
                mean_infection_before = sum(sum(iter['status'].values()) for model in infection_results.values() for iter in model["iterations_before"]) / (len(infection_results) * 10)
                mean_infection_after = sum(sum(iter['status'].values()) for model in infection_results.values() for iter in model["iterations_after"]) / (len(infection_results) * 10)
                metrics.append(["Mean Infection Time", round(mean_infection_before, 2), round(mean_infection_after, 2), "Higher means blocking was effective"])

                # 5️. Peak Infection Reduction (Across Models)
                peak_before_total = sum(model["peak_before"] for model in infection_results.values())
                peak_after_total = sum(model["peak_after"] for model in infection_results.values())
                metrics.append(["Peak Infection", peak_before_total, peak_after_total, "Lower means rumor spread was reduced"])

                # 6️. Final Infection Spread Percentage (Averaged Across Models)
                total_nodes = len(G.nodes)
                final_spread_before = sum(model["infected_before"] / total_nodes for model in infection_results.values()) * 100 / len(infection_results)
                final_spread_after = sum(model["infected_after"] / total_nodes for model in infection_results.values()) * 100 / len(infection_results)
                metrics.append(["Final Infection Spread (%)", round(final_spread_before, 2), round(final_spread_after, 2), "Lower is better"])

                # --------------------------------------
                # ** Display Results in Table**
                df_metrics = pd.DataFrame(metrics, columns=["Metric", "Before Blocking", "After Blocking", "Interpretation"])
                st.dataframe(df_metrics)

                # **Visualization: Bar Chart for Infected Nodes Across Models**
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = pd.DataFrame.from_dict(infection_results, orient="index")[["infected_before", "infected_after"]]

                # **Plot Bar Chart**
                df_plot.plot(kind="bar", ax=ax, color=["red", "green"], alpha=0.7)

                # **Labels and Title**
                ax.set_ylabel("Number of Infected Nodes")
                ax.set_title("Rumor Spread Before vs After Blocking (Across Models)")
                ax.legend(["Before Blocking", "After Blocking"])

                # **Add Value Labels Above Bars**
                for container in ax.containers:
                    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10, fontweight='bold', padding=1)

                # **Fix: Ensure Model Names Appear Horizontally**
                ax.set_xticklabels(df_plot.index, rotation=0)

                # **Display the Plot**
                st.pyplot(fig)
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")

# Community Detection Page
elif page == "Community Detection":
    st.title("Community Detection")
    if st.session_state["graph"]:
        G = st.session_state["graph"]
        st.write("Detect communities in the graph using various algorithms.")

        algorithm = st.selectbox(
            "Select a community detection algorithm:",
            ("Louvain Method", "Girvan-Newman Algorithm")
        )

        if st.button("Run Community Detection"):
            # Detect communities based on the selected algorithm
            if algorithm == "Louvain Method":
                communities = list(nx.community.louvain_communities(G))
            elif algorithm == "Girvan-Newman Algorithm":
                comp = nx.community.girvan_newman(G)
                communities = tuple(sorted(c) for c in next(comp))

            # Visualization
            visualize_communities(G, communities)

            # ------------------  EVALUATION SECTION ------------------
            st.subheader("Community Detection Evaluation Metrics")
            evaluation_metrics = []

            # **1. Modularity Score (Higher = better community structure)**
            modularity = nx.community.quality.modularity(G, communities)
            evaluation_metrics.append(["Modularity Score", round(modularity, 4), "Higher means better-defined communities"])

            # **2. Number of Communities**
            evaluation_metrics.append(["Number of Communities", len(communities), "More communities indicate finer granularity"])

            # **3. Average Community Size**
            avg_size = sum(len(c) for c in communities) / len(communities)
            evaluation_metrics.append(["Average Community Size", round(avg_size, 4), "Larger means broader community structure"])

            # **4. Community Size Variation (Lower = more evenly sized communities)**
            community_sizes = [len(c) for c in communities]
            std_dev_size = np.std(community_sizes)
            evaluation_metrics.append(["Community Size Variation", round(std_dev_size, 4), "Lower means more evenly sized communities"])

            # **5. Conductance (Lower = better community separation)**
            def compute_conductance(G, community):
                cut_size = sum(1 for _ in nx.edge_boundary(G, community))  # Edges between C and rest of G
                total_degree = sum(dict(G.degree(community)).values())  # Total degree of nodes in C
                return cut_size / total_degree if total_degree > 0 else np.nan  # Avoid division by zero

            conductance_values = [compute_conductance(G, c) for c in communities if len(c) > 1]
            avg_conductance = np.nanmean(conductance_values) if conductance_values else np.nan
            evaluation_metrics.append(["Average Conductance", round(avg_conductance, 4), "Lower means better-separated communities"])

            # **6. Average Community Density (Higher = denser communities)**
            densities = [nx.density(G.subgraph(c)) for c in communities if len(c) > 1]
            avg_density = np.nanmean(densities) if densities else np.nan
            evaluation_metrics.append(["Average Community Density", round(avg_density, 4), "Higher means denser communities"])

            # **7. Average Clustering Coefficient (Higher = more tightly clustered communities)**
            avg_clustering = np.mean([nx.average_clustering(G.subgraph(c)) for c in communities if len(c) > 1])
            evaluation_metrics.append(["Average Clustering Coefficient", round(avg_clustering, 4), "Higher means nodes are tightly clustered"])

            # **Convert to DataFrame for Display**
            df_metrics = pd.DataFrame(evaluation_metrics, columns=["Metric", "Value", "Interpretation"])

            # **Better Styling for Table**
            st.dataframe(df_metrics.style.format({"Value": "{:.4f}"}).set_properties(**{"text-align": "left"}))
    else:
        st.warning("No graph loaded. Please upload a valid graph file.")



elif page == "Compare Methods":
    st.title("Comparison of Different Network Analysis Methods")
    if st.session_state["graph"]:
        G = st.session_state["graph"]
        comparison_category = st.selectbox("Select Comparison Category", ["Link Prediction", "Rumor Blocking", "Community Detection"])

        if comparison_category == "Link Prediction":
            st.subheader("Link Prediction Comparison")

            selected_metrics = st.multiselect("Select link prediction metrics:", ["Common Neighbors", "Jaccard Coefficient", "Adamic/Adar", "Preferential Attachment"])

            if selected_metrics:
                auc_scores = {}

                # Convert the graph to undirected to avoid "not implemented for directed type" errors
                G_undirected = G.to_undirected()

                for metric in selected_metrics:
                    # Prepare dataset
                    existing_edges = list(G_undirected.edges())
                    non_edges = list(nx.non_edges(G_undirected))[:len(existing_edges)]
                    data = existing_edges + non_edges
                    labels = [1] * len(existing_edges) + [0] * len(non_edges)

                    # Compute feature for metric
                    features = []
                    for u, v in data:
                        if metric == "Common Neighbors":
                            features.append(len(list(nx.common_neighbors(G_undirected, u, v))))
                        elif metric == "Jaccard Coefficient":
                            features.append(next(nx.jaccard_coefficient(G_undirected, [(u, v)]))[2])
                        elif metric == "Adamic/Adar":
                            features.append(next(nx.adamic_adar_index(G_undirected, [(u, v)]))[2])
                        elif metric == "Preferential Attachment":
                            features.append(next(nx.preferential_attachment(G_undirected, [(u, v)]))[2])

                    X = pd.DataFrame(features, columns=[metric])
                    y = pd.Series(labels)

                    # Train/Test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Train logistic regression model
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred_prob = model.predict_proba(X_test)[:, 1]

                    # Compute AUC
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    auc_score = auc(fpr, tpr)
                    auc_scores[metric] = auc_score

                # Convert AUC Scores to DataFrame
                df_auc = pd.DataFrame.from_dict(auc_scores, orient="index", columns=["AUC Score"])
                df_auc.sort_values(by="AUC Score", ascending=False, inplace=True)

                # Convert AUC Scores to DataFrame
                df_auc = pd.DataFrame.from_dict(auc_scores, orient="index", columns=["AUC Score"])
                df_auc.sort_values(by="AUC Score", ascending=False, inplace=True)

                # **Force Column Width & Styling**
                styled_df = df_auc.style \
                    .set_table_styles([
                        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold"), ("padding", "10px"), ("font-size", "14px")]},
                        {"selector": "td", "props": [("text-align", "center"), ("padding", "10px 20px"), ("white-space", "nowrap"), ("font-size", "14px")]}
                    ]) \
                    .format(precision=4)  # Rounds numbers to 4 decimal places

                # **Force Proper Height and Width**
                st.dataframe(styled_df, height=min(300, len(df_auc) * 35 + 40), width=550)

                # **Visualization**
                fig, ax = plt.subplots(figsize=(10, 5))
                df_auc.plot(kind="bar", color="red", alpha=0.7, ax=ax)

                # Labels and Title
                ax.set_ylabel("AUC Score")
                ax.set_title("Comparison of Link Prediction Metrics")
                ax.set_xticklabels(df_auc.index, rotation=30)

                # **Add value labels above bars**
                for i, v in enumerate(df_auc["AUC Score"]):
                    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold", color="black")

                # Show Plot
                st.pyplot(fig)

            else:
                st.warning("Please select at least one strategy to compare.")

        elif comparison_category == "Rumor Blocking":
            st.subheader("Rumor Blocking Comparison")

            # Allow users to compare multiple rumor blocking strategies
            selected_strategies = st.multiselect(
                "Select strategies to compare:",
                ["Highest Degree", "Highest Betweenness Centrality", "PageRank-Based", "Random Blocking"]
            )

            if selected_strategies:
                num_nodes = st.slider("Number of nodes to block:", min_value=1, max_value=10, value=3)

                # Store comparison results
                comparison_results = {}

                # Run each strategy
                for strategy in selected_strategies:
                    nodes_to_block = determine_nodes_to_block(G, strategy, num_nodes)
                    blocked_G = G.copy()
                    blocked_G.remove_nodes_from(nodes_to_block)

                    # Simulate rumor spread
                    model = ep.IndependentCascadesModel(blocked_G)
                    config = mc.Configuration()
                    config.add_model_parameter("fraction_infected", 0.05)
                    model.set_initial_status(config)
                    iterations = model.iteration_bunch(10)
                    infected_after = sum(iter['node_count'][1] for iter in iterations)

                    comparison_results[strategy] = infected_after

                # Convert dictionary to DataFrame
                df_comparison = pd.DataFrame.from_dict(
                    comparison_results,
                    orient='index',
                    columns=["Infected Nodes After Blocking"]
                ).reset_index()

                # Rename the "index" column properly to "Strategy"
                df_comparison.rename(columns={"index": "Strategy"}, inplace=True)

                # Sort the table based on "Infected Nodes After Blocking" in ascending order
                df_comparison = df_comparison.sort_values(by="Infected Nodes After Blocking", ascending=True).reset_index(drop=True)  # Remove old index

                # Style the table for readability
                styled_df = df_comparison.style.set_properties(**{
                    "white-space": "nowrap",
                    "text-align": "left",
                    "width": "250px"
                })

                # Display the table with proper height based on the number of strategies
                st.dataframe(styled_df, height=(len(df_comparison) * 35) + 50, width=600)


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
                    ax=ax
                )

                ax.set_ylabel("Infected Nodes After Blocking")
                ax.set_xlabel("")  # Remove redundant x-axis label
                ax.set_title("Comparison of Rumor Blocking Strategies")

                # Adjust x-axis labels
                ax.set_xticklabels(df_comparison["Strategy"], rotation=30, ha="right")  # Rotate for better visibility

                # Add value labels above bars with padding to prevent overlapping
                for i, v in enumerate(df_comparison["Infected Nodes After Blocking"]):
                    ax.text(
                        i, v + max(df_comparison["Infected Nodes After Blocking"]) * 0.01,  # Adjust vertical placement
                        str(int(v)),
                        ha="center",
                        fontsize=10,
                        fontweight="bold"
                    )

                # Show Plot in Streamlit
                st.pyplot(fig)
            else:
                st.warning("Please select at least one strategy to compare.")


        elif comparison_category == "Community Detection":
            st.subheader("Community Detection Comparison")

            selected_algorithms = st.multiselect("Select algorithms to compare:", ["Louvain", "Girvan-Newman"])

            if selected_algorithms:
                community_results = {}

                for algorithm in selected_algorithms:
                    if algorithm == "Louvain":
                        communities = list(nx.community.louvain_communities(G))
                    elif algorithm == "Girvan-Newman":
                        comp = nx.community.girvan_newman(G)
                        communities = tuple(sorted(c) for c in next(comp))

                    modularity_score = nx.community.quality.modularity(G, communities)
                    community_results[algorithm] = modularity_score

                # Convert results to DataFrame
                df_community = pd.DataFrame.from_dict(
                    community_results,
                    orient="index",
                    columns=["Modularity Score"]
                )

                # Sort results
                df_community.sort_values(by="Modularity Score", ascending=False, inplace=True)

                # **Display Table with Proper Formatting**
                styled_df = df_community.style \
                    .set_table_styles([
                        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold"), ("padding", "8px")]},
                        {"selector": "td", "props": [("text-align", "center"), ("padding", "8px 20px"), ("white-space", "nowrap")]}
                    ]) \
                    .format(precision=4)  # Ensure uniform decimal places

                st.dataframe(styled_df, height=len(df_community) * 35 + 35, width=500)

                # **Visualization: Bar Chart**
                fig, ax = plt.subplots(figsize=(8, 5))
                df_community.plot(kind="bar", color="blue", alpha=0.7, ax=ax)

                # **Visualization: Bar Chart**
                fig, ax = plt.subplots(figsize=(8, 5))
                df_community.plot(kind="bar", color="blue", alpha=0.7, ax=ax, legend=False)

                # Compute a dynamic offset for label placement
                y_max = df_community["Modularity Score"].max()  # Get the maximum modularity score
                offset = y_max * 0.01  # Adjust text positioning dynamically

                # Add Value Labels Above Bars with Dynamic Positioning
                for i, v in enumerate(df_community["Modularity Score"]):
                    ax.text(
                        i, v + offset,  # Dynamically place text slightly above the bar
                        f"{v:.4f}",
                        ha="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black"
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
else:
    st.warning("Please upload a graph file to proceed.")
