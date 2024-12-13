import causal
import networkx as nx
import alg2
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn as nn
from graphxai.utils.explanation import Explanation
import matplotlib.pyplot as plt
import torch
from imp_syn import from_networkx_to_torch, GIN
from graphxai.datasets import AlkaneCarbonyl, Benzene, FluorideCarbonyl
# data_path_benzene = 'datasets/real_world/benzene/benzene.npz'
# data_path_alkane_carbonyl = 'Ddatasets/real_world/alkane_carbonyl/alkane_carbonyl.npz'
# for i in range(750, num_iterations):
data_path_FluorideCarbonyl = 'datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
# dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# dataset_AlkaneCarbonyl = AlkaneCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_alkane_carbonyl)
dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)
def preprocess_benzene(dataset,i):
    data, exp = dataset[i]
    explanation = exp[0]
    with plt.ioff():
        G, pos = explanation.visualize_graph(show=False)
    role_id = [int(value) for value in explanation.node_imp.tolist()]
    has_one = 1 in role_id  # Check if 1 is in role_id
    print(role_id)
    data = from_networkx_to_torch(G, role_id)
    node_imp = torch.tensor(role_id, dtype=torch.float)
    N = data.x.size(0)  # Number of nodes
    edge_imp = torch.zeros((N, N), dtype=torch.float)
    edge_imp[data.edge_index[0], data.edge_index[1]] = 1.
    if data.x is None:  # If no node features are set
        feature_imp = torch.tensor([1.0])
    else:
        num_features = data.x.size(1)
        feature_imp = torch.ones(num_features, dtype=torch.float)
    gt_exp = Explanation(feature_imp=feature_imp, node_imp=node_imp, edge_imp=edge_imp, graph=data)
    gt_agg = gt_exp.graph
    return data, G, role_id, node_imp, gt_exp, gt_agg, has_one
def vis_benzene(dataset):
    num_iterations = 500
    for i in range(num_iterations):
        plt.figure(figsize=(10, 12), dpi=300)
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess_benzene(dataset, i)
        if has_one:  # Proceed only if there is a 1 in role_id
            plt.figure(figsize=(10, 12), dpi=300)
            nx.draw(G, node_size=30, with_labels=False, node_color=["red" if role_id[node] == 1 else "blue" for node in G.nodes()])
            print(i)
            plt.show()

def vis_FluorideCarbonyl(dataset):
    num_iterations = 500
    for i in range(num_iterations):
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess_benzene(dataset, i)
        if has_one:  # Proceed only if there is a 1 in role_id
            plt.figure(figsize=(10, 12), dpi=300)
            pos = nx.spring_layout(G, seed=1234)
            node_colors_new_v = ["yellow" if role_id[node] == 1 else "blue" for node in G.nodes]
            edge_colors = ["red" if role_id[u]==1 and role_id[v]==1 else "grey" for u, v in G.edges]
            nx.draw(G, pos, with_labels=False, node_size=400, font_size=10, font_weight='bold',
                    node_color=node_colors_new_v, edge_color=edge_colors)
            plt.show()
def vis_graph(unique_labels, graph_id, causal_graphs, df, title):
    color_map = {0: 'blue', 1: 'red', 2: 'black', 3: 'yellow'}
    default_color = 'gray'
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # Adjust the size as needed
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing
    for idx, (graph_id, cg) in enumerate(causal_graphs.items()):
        if idx >= 4:  # Stop after plotting the first nine graphs
            break
        G = nx.Graph()
        G.add_nodes_from(cg.v)
        G.add_edges_from(cg.p)
        pos = nx.spring_layout(G)
        graph_data = df[df['graph_id'] == graph_id]
        node_labels = graph_data.set_index('from')['node_label'].to_dict()
        node_colors = [color_map.get(node_labels.get(node), default_color) for node in G.nodes]
        ax = axes[idx]
        nx.draw(G, pos, with_labels=False, node_size=200, font_weight='bold', node_color=node_colors, edge_color="grey",ax=ax)
        nx.draw(G, pos, with_labels=False, node_size=200, font_size=10, font_weight='bold', node_color=node_colors,edge_color="grey", ax=ax)
        ax.set_title(f'Graph {graph_id}')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
def vis_NCI1():
    NCI1 = 'data/NCI1/'
    NCI1_df = pd.read_csv(NCI1 + 'NCI1_A.txt', sep=',', header=None, names=['from', 'to'])
    NCI1_graph_indicator = pd.read_csv(NCI1 + 'NCI1_graph_indicator.txt', header=None, names=['graph_id'])
    NCI1_node_labels = pd.read_csv(NCI1 + 'NCI1_node_labels.txt', header=None, names=['node_label'])
    NCI1_graph_labels = pd.read_csv(NCI1 + 'NCI1_graph_labels.txt', header=None, names=['graph_label'])
    filtered_graph_ids = NCI1_graph_labels[NCI1_graph_labels['graph_label'] == 0].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    NCI1_df['graph_id'] = NCI1_graph_indicator['graph_id']
    NCI1_df['node_label'] = NCI1_node_labels['node_label']
    grouped = NCI1_df.groupby('graph_id')
    NCI1_causal_graphs = {}
    for graph_id, group in grouped:
        if graph_id in filtered_graph_ids:
            V = set(group['from']).union(set(group['to']))
            edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
            NCI1_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = NCI1_node_labels['node_label'].unique()
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        NCI1_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    filtered_causal_graphs = {graph_id: NCI1_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in NCI1_causal_graphs}
    vis_graph(unique_labels = unique_labels, graph_id = filtered_graph_ids, causal_graphs = filtered_causal_graphs, df = NCI1_df, title =None)
def vis_Mutagenicity():
    Mutagenicity = 'data/Mutagenicity/'
    Mutagenicity_df = pd.read_csv(Mutagenicity + 'Mutagenicity_A.txt', sep=',', header=None, names=['from', 'to'])
    Mutagenicity_graph_indicator = pd.read_csv(Mutagenicity + 'Mutagenicity_graph_indicator.txt', header=None, names=['graph_id'])
    Mutagenicity_node_labels = pd.read_csv(Mutagenicity + 'Mutagenicity_node_labels.txt', header=None, names=['node_label'])
    Mutagenicity_graph_labels = pd.read_csv(Mutagenicity + 'Mutagenicity_graph_labels.txt', header=None, names=['graph_label'])
    filtered_graph_ids = Mutagenicity_graph_labels[Mutagenicity_graph_labels['graph_label'] == 0].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    Mutagenicity_df['graph_id'] = Mutagenicity_graph_indicator['graph_id']
    Mutagenicity_df['node_label'] = Mutagenicity_node_labels['node_label']
    grouped = Mutagenicity_df.groupby('graph_id')
    Mutagenicity_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        Mutagenicity_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = Mutagenicity_node_labels['node_label'].unique()
    filtered_causal_graphs = {graph_id: Mutagenicity_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in Mutagenicity_causal_graphs}
    vis_graph(unique_labels=unique_labels, graph_id=filtered_graph_ids, causal_graphs=filtered_causal_graphs, df=Mutagenicity_df, title =None)
def vis_ENZYMES():
    ENZYMES = 'data/ENZYMES/'
    ENZYMES_df = pd.read_csv(ENZYMES + 'ENZYMES_A.txt', sep=',', header=None, names=['from', 'to'])
    ENZYMES_graph_indicator = pd.read_csv(ENZYMES + 'ENZYMES_graph_indicator.txt', header=None, names=['graph_id'])
    ENZYMES_node_labels = pd.read_csv(ENZYMES + 'ENZYMES_node_labels.txt', header=None,names=['node_label'])
    ENZYMES_graph_labels = pd.read_csv(ENZYMES + 'ENZYMES_graph_labels.txt', header=None,names=['graph_label'])
    filtered_graph_ids = ENZYMES_graph_labels[ENZYMES_graph_labels['graph_label'] == 0].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    ENZYMES_df['graph_id'] = ENZYMES_graph_indicator['graph_id']
    ENZYMES_df['node_label'] = ENZYMES_node_labels['node_label']
    grouped = ENZYMES_df.groupby('graph_id')
    ENZYMES_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        ENZYMES_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = ENZYMES_node_labels['node_label'].unique()
    filtered_causal_graphs = {graph_id: ENZYMES_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in ENZYMES_causal_graphs}
    vis_graph(unique_labels=unique_labels, graph_id=filtered_graph_ids, causal_graphs=filtered_causal_graphs,df=ENZYMES_df, title=None)
def vis_AIDS():
    AIDS = 'data/AIDS/'
    AIDS_df = pd.read_csv(AIDS + 'AIDS_A.txt', sep=',', header=None, names=['from', 'to'])
    AIDS_graph_indicator = pd.read_csv(AIDS + 'AIDS_graph_indicator.txt', header=None, names=['graph_id'])
    AIDS_node_labels = pd.read_csv(AIDS + 'AIDS_node_labels.txt', header=None, names=['node_label'])
    AIDS_graph_labels = pd.read_csv(AIDS + 'AIDS_graph_labels.txt', header=None, names=['graph_label'])
    filtered_graph_ids = AIDS_graph_labels[AIDS_graph_labels['graph_label'] == 0].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    AIDS_df['graph_id'] = AIDS_graph_indicator['graph_id']
    AIDS_df['node_label'] = AIDS_node_labels['node_label']
    grouped = AIDS_df.groupby('graph_id')
    AIDS_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        AIDS_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = AIDS_node_labels['node_label'].unique()
    filtered_causal_graphs = {graph_id: AIDS_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in AIDS_causal_graphs}
    vis_graph(unique_labels=unique_labels, graph_id=filtered_graph_ids, causal_graphs=filtered_causal_graphs,df=AIDS_df, title=None)
def vis_PROTEINS():
    PROTEINS = 'data/PROTEINS/'
    PROTEINS_df = pd.read_csv(PROTEINS + 'PROTEINS_A.txt', sep=',', header=None, names=['from', 'to'])
    PROTEINS_graph_indicator = pd.read_csv(PROTEINS + 'PROTEINS_graph_indicator.txt', header=None, names=['graph_id'])
    PROTEINS_node_labels = pd.read_csv(PROTEINS + 'PROTEINS_node_labels.txt', header=None, names=['node_label'])
    PROTEINS_graph_labels = pd.read_csv(PROTEINS + 'PROTEINS_graph_labels.txt', header=None, names=['graph_label'])
    filtered_graph_ids = PROTEINS_graph_labels[PROTEINS_graph_labels['graph_label'] == 1].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    PROTEINS_df['graph_id'] = PROTEINS_graph_indicator['graph_id']
    PROTEINS_df['node_label'] = PROTEINS_node_labels['node_label']
    grouped = PROTEINS_df.groupby('graph_id')
    PROTEINS_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        PROTEINS_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = PROTEINS_node_labels['node_label'].unique()
    filtered_causal_graphs = {graph_id: PROTEINS_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in PROTEINS_causal_graphs}
    vis_graph(unique_labels=unique_labels, graph_id=filtered_graph_ids, causal_graphs=filtered_causal_graphs,df=PROTEINS_df, title=None)
def vis_REDDIT():
    REDDIT = 'data/REDDIT-BINARY/'
    hot_id_file_path = REDDIT + 'hot_id.txt'

    # Reading graph data
    REDDIT_df = pd.read_csv(REDDIT + 'REDDIT-BINARY_A.txt', sep=',', header=None, names=['from', 'to'])
    REDDIT_graph_indicator = pd.read_csv(REDDIT + 'REDDIT-BINARY_graph_indicator.txt', header=None, names=['graph_id'])
    REDDIT_graph_labels = pd.read_csv(REDDIT + 'REDDIT-BINARY_graph_labels.txt', header=None, names=['graph_label'])

    # Check the contents of REDDIT_graph_labels
    print("Sample of graph labels:", REDDIT_graph_labels['graph_label'].value_counts())

    # Mapping nodes to their graph IDs
    node_to_graph_id = pd.Series(REDDIT_graph_indicator['graph_id'].values, index=REDDIT_graph_indicator.index + 1)

    # Assigning graph IDs to edges in REDDIT_df
    REDDIT_df['graph_id'] = REDDIT_df['from'].map(node_to_graph_id)

    # Reading and processing hot_id.txt for node labels
    with open(hot_id_file_path, 'r') as file:
        hot_ids = file.readlines()
    gt_node_ids = [int(id.strip()) for id in hot_ids]

    # Generate node labels
    total_nodes = max(REDDIT_df['from'].max(), REDDIT_df['to'].max()) + 1
    node_labels = [0] * total_nodes
    for node_id in gt_node_ids:
        if node_id < total_nodes:
            node_labels[node_id] = 1

    # Create a node label mapping
    node_label_mapping = {node_id: label for node_id, label in enumerate(node_labels)}

    # Apply node labels to the DataFrame
    REDDIT_df['from_label'] = REDDIT_df['from'].map(node_label_mapping)
    REDDIT_df['to_label'] = REDDIT_df['to'].map(node_label_mapping)

    label_to_filter = 1  # or -1

    # Filter graphs based on the chosen label
    filtered_graph_ids = set(REDDIT_graph_labels[REDDIT_graph_labels['graph_label'] == label_to_filter].index + 1)
    filtered_graph_ids = filtered_graph_ids.intersection(set(REDDIT_df['graph_id'].unique()))

    # Grouping and creating causal graphs
    grouped = REDDIT_df.groupby('graph_id')
    REDDIT_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        REDDIT_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)

    # Visualizing graphs
    unique_labels = set(node_labels)
    filtered_causal_graphs = {graph_id: REDDIT_causal_graphs[graph_id] for graph_id in filtered_graph_ids if
                              graph_id in REDDIT_causal_graphs}

    def vis_graph(unique_labels, graph_id, causal_graphs, df, title):
        color_map = {0: 'blue', 1: 'red', 2: 'black', 3: 'yellow'}
        default_color = 'gray'
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # Adjust the size as needed
        axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

        for idx, (graph_id, cg) in enumerate(causal_graphs.items()):
            if idx >= 4:  # Stop after plotting the first four graphs
                break

            G = nx.Graph()
            G.add_nodes_from(cg.v)
            G.add_edges_from(cg.p)

            if len(G.nodes) == 0 or len(G.edges) == 0:  # Check for empty graphs
                continue

            pos = nx.spring_layout(G, k=0.15, iterations=20)  # Adjust layout parameters as needed
            graph_data = df[df['graph_id'] == graph_id]

            # Consolidate 'from_label' and 'to_label' into a single label per node
            node_labels = {}
            for node in G.nodes:
                from_label = graph_data.loc[graph_data['from'] == node, 'from_label'].max()
                to_label = graph_data.loc[graph_data['to'] == node, 'to_label'].max()
                node_labels[node] = max(from_label, to_label)

            node_colors = [color_map.get(node_labels.get(node), default_color) for node in G.nodes]

            ax = axes[idx]
            nx.draw(G, pos, with_labels=True, node_size=300, font_weight='bold', node_color=node_colors,
                    edge_color="grey", ax=ax)
            ax.set_title(f'Graph {graph_id}')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    if not filtered_graph_ids:
        print("filtered_graph_ids is empty.")
        print("REDDIT_graph_labels['graph_label'] == 0:", REDDIT_graph_labels[REDDIT_graph_labels['graph_label'] == 0])
        print("REDDIT_df['graph_id'].unique():", REDDIT_df['graph_id'].unique())
    if filtered_graph_ids:
        vis_graph(unique_labels=unique_labels, graph_id=list(filtered_graph_ids), causal_graphs=filtered_causal_graphs,
              df=REDDIT_df, title='Graph Visualization')
