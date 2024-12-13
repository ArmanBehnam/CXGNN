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
# data_path_alkane_carbonyl = 'datasets/real_world/alkane_carbonyl/alkane_carbonyl.npz'
# for i in range(750, num_iterations):
# data_path_FluorideCarbonyl = 'datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
# dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# dataset_AlkaneCarbonyl = AlkaneCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_alkane_carbonyl)
# dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)
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
def vis_graph(unique_labels, graph_id, causal_graphs, df, title):
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
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
        # node_label_map = dict(zip(graph_data['from'], graph_data['node_label']))
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
    NCI1 = 'D:/University/Research/data/NCI1/'
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

def vis_REDDIT_BINARY():
    REDDIT_BINARY = 'data/REDDIT-BINARY/'
    REDDIT_BINARY_df = pd.read_csv(REDDIT_BINARY + 'REDDIT-BINARY_A.txt', sep=',', header=None, names=['from', 'to'])
    REDDIT_BINARY_graph_indicator = pd.read_csv(REDDIT_BINARY + 'REDDIT-BINARY_graph_indicator.txt', header=None, names=['graph_id'])
    # REDDIT_BINARY_node_labels = pd.read_csv(REDDIT_BINARY + 'REDDIT-BINARY_node_labels.txt', header=None, names=['node_label'])
    REDDIT_BINARY_graph_labels = pd.read_csv(REDDIT_BINARY + 'REDDIT-BINARY_graph_labels.txt', header=None, names=['graph_label'])
    filtered_graph_ids = REDDIT_BINARY_graph_labels[REDDIT_BINARY_graph_labels['graph_label'] == 0].index + 1
    filtered_graph_ids = filtered_graph_ids.tolist()
    REDDIT_BINARY_df['graph_id'] = REDDIT_BINARY_graph_indicator['graph_id']
    REDDIT_BINARY_df['node_label'] = REDDIT_BINARY_node_labels['node_label']
    grouped = REDDIT_BINARY_df.groupby('graph_id')
    REDDIT_BINARY_causal_graphs = {}
    for graph_id, group in grouped:
        V = set(group['from']).union(set(group['to']))
        edges = list(zip(group['from'], group['to'])) + list(zip(group['to'], group['from']))
        REDDIT_BINARY_causal_graphs[graph_id] = causal.CausalGraph(V=V, path=edges)
    unique_labels = REDDIT_BINARY_node_labels['node_label'].unique()
    filtered_causal_graphs = {graph_id: REDDIT_BINARY_causal_graphs[graph_id] for graph_id in filtered_graph_ids if graph_id in REDDIT_BINARY_causal_graphs}
    vis_graph(unique_labels=unique_labels, graph_id=filtered_graph_ids, causal_graphs=filtered_causal_graphs,df=REDDIT_BINARY_df, title=None)

def get_node_labels_by_graph(path):
    # Load data files
    edges = np.loadtxt(path + 'REDDIT-BINARY_A.txt', delimiter=',').astype(np.int32)
    graph_indicator = np.loadtxt(path + 'REDDIT-BINARY_graph_indicator.txt', delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(path + 'REDDIT-BINARY_graph_labels.txt', delimiter=',').astype(np.int32)

    # Initialize a dictionary to hold node labels for each graph
    graph_node_labels = {}

    for i, graph_id in enumerate(graph_indicator):
        if graph_id not in graph_node_labels:
            graph_node_labels[graph_id] = []

        if graph_labels[graph_id - 1] == 1:  # Assuming label 1 is the ground truth
            graph_node_labels[graph_id].append(1)  # Node is part of the ground truth
        else:
            graph_node_labels[graph_id].append(0)  # Node is not part of the ground truth

    return graph_node_labels
REDDIT_BINARY = 'data/REDDIT-BINARY/'
node_labels_by_graph = get_node_labels_by_graph(REDDIT_BINARY)
for graph_id, labels in node_labels_by_graph.items():
    print(f"Node labels for graph {graph_id}: {labels}")
