import numpy as np
import math
import causal
import networkx as nx
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn as nn
from graphxai.utils.explanation import Explanation
from graphxai.explainers import PGMExplainer, GuidedBP, IntegratedGradExplainer, GradExplainer, GNNExplainer
import matplotlib.pyplot as plt
from alg2 import alg_2
def join_graph(G1, G2, n_pert_edges):
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F
def ba(start, width, role_start=0, m=5):
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles
def house(start, role_start=0):
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start, role_start, role_start]
    return graph, roles
def grid(start, dim=2, role_start=0):
    grid_G = nx.grid_graph([dim, dim])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start)
    roles = [role_start for i in grid_G.nodes()]
    return grid_G, roles
def tree(start, height, r=2, role_start=0):
    graph = nx.balanced_tree(r, height)
    roles = [0] * graph.number_of_nodes()
    return graph, roles
def cycle(start, len_cycle, role_start=0):
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    return graph, roles
def build_graph(width_basis,basis_type,Ground_truth,start=0,rdm_basis_plugins=False,add_random_edges=0,m=None):
    num_shapes = 1
    if Ground_truth == "grid":
        list_shapes = [[Ground_truth, 3]] * num_shapes
    elif Ground_truth == "house":
        list_shapes = [[Ground_truth]] * num_shapes  # You can customize the shapes here
    else:
        list_shapes = [[Ground_truth, 6]] * num_shapes
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)
    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])
    return basis, role_id, plugins
def perturb(graph_list, p):
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list
def generate_random_graph(basis_type=None, Ground_truth=None):
    num_shapes = 1
    # width_basis = np.random.randint(2, 5)  # Random width for the basis
    # m = np.random.randint(2, 4)  # Random 'm' for BA graph
    width_basis = 3
    m = 2
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    return G, role_id, name
def from_networkx_to_torch(graph, role_id):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor(role_id, dtype=torch.float).view(-1, 1)  # Using role_id as node features
    return Data(x=x, edge_index=edge_index)
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
def implement(G, model, node_imp, predicted_classes, data, role_id, num_epochs = None):
    null_batch = torch.zeros(1).long()
    forward_kwargs = {'batch': null_batch}  # Input to explainers forward methods
    forward_kwargs.pop('unwanted_argument', None)
    # Plot ground-truth explanation ------------------------
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(40, 25))
    node_colors = ['blue' if role_id[node] == 0 else 'green' if role_id[node] == 1 else 'black' if role_id[node] == 2 else
        'orange' if role_id[node] == 3 else 'purple' if role_id[node] == 4 else 'cyan' if role_id[node] == 5 else
        'pink' if role_id[node] == 6 else 'brown' if role_id[node] == 7 else 'red' for node in G.nodes]
    pos = nx.spring_layout(G, seed=1234)
    nx.draw(G, pos, with_labels=True, node_size=400, font_size=10, font_weight='bold', node_color=node_colors, edge_color="grey", ax=ax1)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, edgecolors='black')
    nodes.set_edgecolor('black')
    ax1.set_title("Ground truth")
    # ------------------------------------------------------

    # Call Explainer: PGMExplainer--------------------------------------
    PGME_exp = PGMExplainer(model, explain_graph=True)
    PGME = PGME_exp.get_explanation_graph(x=data.x, edge_index=data.edge_index, top_k_nodes=5)
    PGME.visualize_graph(ax=ax2)
    ax2.set_title('PGMExplainer')
    PGME_predictions = (PGME.node_imp > 0.5).float()
    PGME_intersection = torch.logical_and(node_imp, PGME_predictions)
    print(f"node_imp size: {node_imp.size()}")
    print(f"PGME_predictions size: {PGME_predictions.size()}")

    PGME_recall = torch.sum(PGME_intersection).item() / len(PGME_intersection)
    PGME_find = int((node_imp == PGME_predictions).all().item())
    PGME_validity = int(all(idx in (node_imp == 1).nonzero(as_tuple=False).view(-1).tolist() for idx in (PGME_predictions == 1).nonzero(as_tuple=False).view(-1).tolist()))
    PGME_acc = ((torch.sum((node_imp == 1) & (PGME_predictions == 1)).item()) / (torch.sum(node_imp == 1).item())) * 100
    print(f"PGMExplainer Explanation recall is: {PGME_recall * 100:.2f}%")
    print("PGMExplainer Explanation found the ground truth? ", PGME_find)
    print("Is PGMExplainer Explanation final subgraph valid? ", PGME_validity)
    print(f"How well does the ground truth is found? {PGME_acc:.2f}%", '\n')
    # ------------------------------------------------------

    # Call Explainer: GuidedBP--------------------------------------
    if 'batch' in forward_kwargs:
        forward_kwargs.pop('batch')
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        guidedbp_predicted_classes = preds.argmax(dim=1)
    print("Model Predictions:", guidedbp_predicted_classes)
    guided_bp = GuidedBP(model=model)
    guided_bp.debug_gradients(data.x, data.edge_index)  # Check if gradients are non-zero
    guidedbp = guided_bp.get_explanation_graph(x=data.x, edge_index=data.edge_index,y=torch.tensor(role_id, dtype=torch.long))
    print("Whole Graph Explanation:", guidedbp)
    threshold = 0.1  # Modify as needed
    guidedbp_predictions = (guidedbp.node_imp > threshold).float()
    print("GuidedBP Predictions:", guidedbp_predictions)
    guidedbp_intersection = torch.logical_and(node_imp, guidedbp_predictions)
    guidedbp_recall = torch.sum(guidedbp_intersection).item() / len(guidedbp_intersection)
    guidedbp.visualize_graph(ax=ax3)
    ax3.set_title('GuidedBP')
    role_id_tensor = torch.tensor(role_id, dtype=torch.float)
    guidedbp_find = int((role_id_tensor == guidedbp_predictions).all().item())
    predicted_important = (guidedbp_predictions == 1)
    true_important = (role_id_tensor == 1)
    valid_predictions = predicted_important & true_important
    guidedbp_validity = int(valid_predictions.sum().item() == predicted_important.sum().item()) if predicted_important.any() else 1
    true_positives = (guidedbp_predictions == 1) & (role_id_tensor == 1)
    guidedbp_acc = (true_positives.sum().float() / role_id_tensor.sum().item() * 100) if role_id_tensor.sum().item() > 0 else 0
    print(f"GuidedBP Explanation recall is: {guidedbp_recall * 100:.2f}%")
    print("GuidedBP Explanation found the ground truth? ", guidedbp_find)
    print("Is GuidedBP Explanation final subgraph valid? ", guidedbp_validity)
    print(f"How well does the ground truth is found? {guidedbp_acc:.2f}%", '\n')
    # ------------------------------------------------------

    # Call Explainer: Integrated Gradients--------------------------------------
    if 'batch' in forward_kwargs:
        forward_kwargs.pop('batch')
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        integrated_grads_predicted_classes = preds.argmax(dim=1)
    print("Model Predictions:", integrated_grads_predicted_classes)
    integrated_grads = IntegratedGradExplainer(model=model, criterion=torch.nn.CrossEntropyLoss())
    integrated_grads_explanation = integrated_grads.get_explanation_graph(x=data.x, edge_index=data.edge_index, y=torch.tensor(role_id, dtype=torch.long))
    print("Whole Graph Explanation:", integrated_grads_explanation)
    threshold = 0.1  # Modify as needed
    integrated_grads_predictions = (integrated_grads_explanation.node_imp > threshold).float()
    print("Integrated Gradients Predictions:", integrated_grads_predictions)
    integrated_grads_intersection = torch.logical_and(node_imp, integrated_grads_predictions)
    integrated_grads_recall = torch.sum(integrated_grads_intersection).item() / len(integrated_grads_intersection)
    role_id_tensor = torch.tensor(role_id, dtype=torch.float)
    integrated_grads_find = int((role_id_tensor == integrated_grads_predictions).all().item())
    print("Integrated Gradients Explanation found the ground truth? ", integrated_grads_find)
    predicted_important = (integrated_grads_predictions == 1)
    true_important = (role_id_tensor == 1)
    valid_predictions = predicted_important & true_important
    integrated_grads_validity = int(valid_predictions.sum().item() == predicted_important.sum().item()) if predicted_important.any() else 1
    true_positives = (integrated_grads_predictions == 1) & (role_id_tensor == 1)
    integrated_grads_acc = (true_positives.sum().float() / role_id_tensor.sum().item() * 100) if role_id_tensor.sum().item() > 0 else 0
    print(f"Integrated Gradients Explanation recall is: {integrated_grads_recall * 100:.2f}%")
    print("Is Integrated Gradients Explanation final subgraph valid? ", integrated_grads_validity)
    print(f"How well does the ground truth is found? {integrated_grads_acc:.2f}%", '\n')
    # ------------------------------------------------------

    # Call Explainer: GradExplainer--------------------------------------
    if 'batch' in forward_kwargs:
        forward_kwargs.pop('batch')
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        grads_predicted_classes = preds.argmax(dim=1)
    print("Model Predictions:", grads_predicted_classes)
    grads = GradExplainer(model=model, criterion=torch.nn.CrossEntropyLoss())
    grads_explanation = grads.get_explanation_graph(x=data.x, edge_index=data.edge_index,label=torch.tensor(role_id, dtype=torch.long))
    print("Whole Graph Explanation:", grads_explanation)
    threshold = 0.1  # Modify as needed
    grads_predictions = (grads_explanation.node_imp > threshold).float()
    print("Integrated Gradients Predictions:", grads_predictions)
    grads_intersection = torch.logical_and(node_imp, grads_predictions)
    grads_recall = torch.sum(integrated_grads_intersection).item() / len(grads_intersection)
    role_id_tensor = torch.tensor(role_id, dtype=torch.float)
    grads_find = int((role_id_tensor == grads_predictions).all().item())
    print("GradExplainer Explanation found the ground truth? ", grads_find)
    predicted_important = (grads_predictions == 1)
    true_important = (role_id_tensor == 1)
    valid_predictions = predicted_important & true_important
    grads_validity = int(valid_predictions.sum().item() == predicted_important.sum().item()) if predicted_important.any() else 1
    true_positives = (integrated_grads_predictions == 1) & (role_id_tensor == 1)
    grads_acc = (true_positives.sum().float() / role_id_tensor.sum().item() * 100) if role_id_tensor.sum().item() > 0 else 0
    print(f"GradExplainer Gradients Explanation recall is: {grads_recall * 100:.2f}%")
    print("Is GradExplainer Gradients Explanation final subgraph valid? ", grads_validity)
    print(f"How well does the ground truth is found? {grads_acc:.2f}%", '\n')
    # ------------------------------------------------------

    # Call Explainer: GNNExplainer--------------------------------------
    GNN_exp = GNNExplainer(model)
    gnnex = GNN_exp.get_explanation_graph(x=data.x, edge_index=data.edge_index)
    gnnex.visualize_graph(ax=ax6)
    ax6.set_title('GNNexplainer')
    gnnex_predictions = (gnnex.node_imp > 0.5).float()
    gnnex_intersection = torch.logical_and(node_imp, gnnex_predictions)
    gnnex_recall = torch.sum(gnnex_intersection).item() / len(gnnex_intersection)
    gnnex_find = int((node_imp == gnnex_predictions).all().item())
    gnnex_validity = int(all(idx in (node_imp == 1).nonzero(as_tuple=False).view(-1).tolist() for idx in (gnnex_predictions == 1).nonzero(as_tuple=False).view(-1).tolist()))
    gnnex_acc = ((torch.sum((node_imp == 1) & (gnnex_predictions == 1)).item()) / (torch.sum(node_imp == 1).item())) * 100
    print(f"GNNExplainer Explanation recall is: {gnnex_recall * 100:.2f}%")
    print("GNNExplainer Explanation found the ground truth? ", gnnex_find)
    print("Is GNNExplainer Explanation final subgraph valid? ", gnnex_validity)
    print(f"How well does the ground truth is found? {gnnex_acc:.2f}%", '\n')
    # ------------------------------------------------------

    # Call Explainer: My Explanation--------------------------------------
    data1 = pd.DataFrame({'node_label': predicted_classes.tolist()})
    cg = causal.CausalGraph(V=G.nodes, path=G.edges)
    relative_positives = (node_imp == 1).nonzero(as_tuple=True)[0]
    relative_positives = relative_positives.tolist()
    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = alg_2(Graph=cg,num_epochs=num_epochs,data=data1,role_id=role_id)
    node_colors_new_v = ["#FFE134" if node in best_new_v else "#400040" for node in G.nodes]
    edge_colors = ["red" if u in best_new_v and v in best_new_v else "grey" for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_size=400, font_size=10, font_weight='bold', node_color=node_colors_new_v, edge_color=edge_colors, ax=ax7)
    ax7.set_title('My Explanation')
    print('The ground truth is: ', relative_positives)
    print('Our method finding is: ', best_new_v)
    my_predictions = torch.zeros_like(node_imp)
    my_predictions[list(best_new_v) if isinstance(best_new_v, set) else best_new_v] = 1
    relative_positives_tensor = torch.zeros_like(node_imp)
    relative_positives_tensor[relative_positives] = 1
    predicted_tensor = torch.zeros_like(node_imp)
    predicted_tensor[list(best_new_v)] = 1
    my_recall = torch.sum(predicted_tensor[relative_positives]).item() / len(relative_positives)
    my_acc = (torch.sum((node_imp == 1) & (my_predictions == 1)).item()) / (torch.sum(node_imp == 1).item()) * 100
    my_gt_find = int(set(best_new_v) == set(relative_positives))
    my_validity = int(all(item in relative_positives for item in best_new_v))
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return ({'PGME_find': PGME_find, 'guidedbp_find': guidedbp_find, 'ig_find': integrated_grads_find, 'grad_find': grads_find,
             'gnnex_find': gnnex_find, 'my_find': my_gt_find},
            {'PGME_acc': PGME_acc, 'guidedbp_acc': guidedbp_acc, 'ig_acc': integrated_grads_acc, 'grad_acc': grads_acc,
             'gnnex_acc': gnnex_acc, 'my_acc': my_acc},
            {'PGME_recall': PGME_recall * 100, 'guidedbp_recall': guidedbp_recall * 100,
             'ig_recall': integrated_grads_recall * 100, 'grad_recall': grads_recall * 100,
             'gnnex_recall': gnnex_recall * 100, 'my_recall': my_recall * 100},
            {'PGME_validity': PGME_validity, 'guidedbp_validity': guidedbp_validity,
             'ig_validity': integrated_grads_validity, 'grad_validity': grads_validity,
             'gnnex_validity': gnnex_validity, 'my_validity': my_validity})
def implement_CXGNN(G, node_imp, predicted_classes, data, role_id, num_epochs = None):
    null_batch = torch.zeros(1).long()
    forward_kwargs = {'batch': null_batch}  # Input to explainers forward methods
    forward_kwargs.pop('unwanted_argument', None)
    data1 = pd.DataFrame({'node_label': predicted_classes.tolist()})
    cg = causal.CausalGraph(V=G.nodes, path=G.edges)
    relative_positives = (node_imp == 1).nonzero(as_tuple=True)[0]
    relative_positives = relative_positives.tolist()
    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = alg_2(Graph=cg,num_epochs=num_epochs,data=data1,role_id=role_id)
    print('The ground truth is: ', relative_positives)
    print('Our method finding is: ', best_new_v)
    my_predictions = torch.zeros_like(node_imp)
    my_predictions[list(best_new_v) if isinstance(best_new_v, set) else best_new_v] = 1
    relative_positives_tensor = torch.zeros_like(node_imp)
    relative_positives_tensor[relative_positives] = 1
    predicted_tensor = torch.zeros_like(node_imp)
    predicted_tensor[list(best_new_v)] = 1
    my_recall = torch.sum(predicted_tensor[relative_positives]).item() / len(best_new_v)
    my_acc = (torch.sum((node_imp == 1) & (my_predictions == 1)).item()) / (torch.sum(node_imp == 1).item()) * 100
    my_gt_find = int(set(best_new_v) == set(relative_positives))
    my_validity = int(all(item in relative_positives for item in best_new_v))
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return ({'my_find': my_gt_find},{'my_acc': my_acc},{'my_recall': my_recall * 100},{ 'my_validity': my_validity})
groundtruth_find_df = pd.DataFrame(columns=['PGME_find', 'guidedbp_find', 'ig_find', 'grad_find', 'gnnex_find', 'my_find'])
graph_explanation_accuracy_df = pd.DataFrame(columns=['PGME_acc', 'guidedbp_acc', 'ig_acc', 'grad_acc', 'gnnex_acc', 'my_acc'])
graph_explanation_recall_df = pd.DataFrame(columns=['PGME_recall', 'guidedbp_recall', 'ig_recall', 'grad_recall', 'gnnex_recall', 'my_recall'])
validities_df = pd.DataFrame(columns=['PGME_validity', 'guidedbp_validity', 'ig_validity', 'grad_validity', 'gnnex_validity', 'my_validity'])
def implement_GuidedBP(G, model, node_imp, predicted_classes, data, role_id, num_epochs = 20):
    null_batch = torch.zeros(1).long()
    forward_kwargs = {'batch': null_batch}  # Input to explainers forward methods
    forward_kwargs.pop('unwanted_argument', None)
    # Call Explainer: GuidedBP--------------------------------------
    if 'batch' in forward_kwargs:
        forward_kwargs.pop('batch')
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        guidedbp_predicted_classes = preds.argmax(dim=1)
    print("Model Predictions:", guidedbp_predicted_classes)
    guided_bp = GuidedBP(model=model)
    guided_bp.debug_gradients(data.x, data.edge_index)  # Check if gradients are non-zero
    guidedbp = guided_bp.get_explanation_graph(x=data.x, edge_index=data.edge_index,
                                               y=torch.tensor(role_id, dtype=torch.long))
    print("Whole Graph Explanation:", guidedbp)
    threshold = 0.1  # Modify as needed
    guidedbp_predictions = (guidedbp.node_imp > threshold).float()
    has_one = 1 in guidedbp_predictions  # Check if 1 is in role_id
    print("GuidedBP Predictions:", guidedbp_predictions)
    guidedbp_intersection = torch.logical_and(node_imp, guidedbp_predictions)
    guidedbp_recall = torch.sum(guidedbp_intersection).item() / len(guidedbp_intersection)
    role_id_tensor = torch.tensor(role_id, dtype=torch.float)
    guidedbp_find = int((role_id_tensor == guidedbp_predictions).all().item())
    predicted_important = (guidedbp_predictions == 1)
    true_important = (role_id_tensor == 1)
    valid_predictions = predicted_important & true_important
    guidedbp_validity = int(valid_predictions.sum().item() == predicted_important.sum().item()) if predicted_important.any() else 0
    true_positives = (guidedbp_predictions == 1) & (role_id_tensor == 1)
    guidedbp_acc = (true_positives.sum().float() / role_id_tensor.sum().item() * 100) if role_id_tensor.sum().item() > 0 else 0
    print(f"GuidedBP Explanation recall is: {guidedbp_recall * 100:.2f}%")
    print("GuidedBP Explanation found the ground truth? ", guidedbp_find)
    print("Is GuidedBP Explanation final subgraph valid? ", guidedbp_validity)
    print(f"How well does the ground truth is found? {guidedbp_acc:.2f}%", '\n')
    # ------------------------------------------------------
    return ({'guidedbp_find': guidedbp_find},
            {'guidedbp_acc': guidedbp_acc},
            {'guidedbp_recall': guidedbp_recall * 100},
            {'guidedbp_validity': guidedbp_validity})
def ba_house(basis_type="ba", Ground_truth="house", width_basis = None, m = 3, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement(G, model, node_imp, predicted_classes, data, role_id, num_epochs = num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")
def ba_grid(basis_type="ba", Ground_truth="grid", width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement(G, model,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")
def ba_cycle(basis_type="ba", Ground_truth="cycle", width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement(G, model,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")
def tree_house(basis_type="tree", Ground_truth="house", width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        # G, role_id, name = generate_random_graph(basis_type="ba", Ground_truth="house")
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement(G, model,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")
def tree_grid(basis_type="tree", Ground_truth="grid", width_basis = 3, m = 3, num_epochs = None, num_iterations = 500):

    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        # G, role_id, name = generate_random_graph(basis_type="ba", Ground_truth="house")
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement_CXGNN(G,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")
def tree_cycle(basis_type="tree", Ground_truth="cycle", width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        # G, role_id, name = generate_random_graph(basis_type="ba", Ground_truth="house")
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        data = from_networkx_to_torch(G, role_id)
        model = GIN(1, 32, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            predicted_classes = out.argmax(dim=1)
            role_id_tensor = torch.tensor(role_id, dtype=torch.long)
            loss = criterion(out, role_id_tensor)
            loss.backward()
        optimizer.step()
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
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement(G, model,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"The implement took {minutes} minutes and {seconds:.2f} seconds to execute.")
    avg_nodes = total_nodes / num_iterations
    avg_edges = total_edges / num_iterations
    print(f"Average number of nodes: {avg_nodes}")
    print(f"Average number of edges: {avg_edges}")
    print("Graph explanation recall:")
    for column in graph_explanation_recall_df.columns:
        print(f"{column}: {graph_explanation_recall_df.mean()[column]:.2f}")
    print("Graph explanation accuracy:")
    for column in graph_explanation_accuracy_df.columns:
        print(f"{column}: {graph_explanation_accuracy_df.mean()[column]:.2f}")
    print("Groundtruth match accuracy:")
    for column in groundtruth_find_df.columns:
        print(f"{column}: {groundtruth_find_df.mean()[column] * 100:.2f}")
    print("Groundtruth Validity:")
    for column in validities_df.columns:
        print(f"{column}: {validities_df.mean()[column] * 100:.2f}")


