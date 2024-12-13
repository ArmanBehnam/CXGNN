import numpy as np
import math
import causal
import networkx as nx
import time
import alg2
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn as nn
from imp_syn import from_networkx_to_torch, GIN
from graphxai.utils.explanation import Explanation
from graphxai.datasets import AlkaneCarbonyl, Benzene, FluorideCarbonyl
from graphxai.explainers import PGMExplainer, GuidedBP, IntegratedGradExplainer, GradExplainer, GNNExplainer
import matplotlib.pyplot as plt
data_path_benzene = 'datasets/real_world/benzene/benzene.npz'
data_path_FluorideCarbonyl = 'datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
def preprocess(dataset,i):
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
    best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = alg2.alg_2(Graph=cg, num_epochs=num_epochs, data=data1, role_id=role_id)
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
# groundtruth_find_df = pd.DataFrame(columns=['PGME_find', 'guidedbp_find', 'ig_find', 'grad_find', 'gnnex_find', 'my_find'])
# graph_explanation_accuracy_df = pd.DataFrame(columns=['PGME_acc', 'guidedbp_acc', 'ig_acc', 'grad_acc', 'gnnex_acc', 'my_acc'])
# graph_explanation_recall_df = pd.DataFrame(columns=['PGME_recall', 'guidedbp_recall', 'ig_recall', 'grad_recall', 'gnnex_recall', 'my_recall'])
# validities_df = pd.DataFrame(columns=['PGME_validity', 'guidedbp_validity', 'ig_validity', 'grad_validity', 'gnnex_validity', 'my_validity'])
groundtruth_find_df = pd.DataFrame(columns=['my_find'])
graph_explanation_accuracy_df = pd.DataFrame(columns=['my_acc'])
graph_explanation_recall_df = pd.DataFrame(columns=['my_recall'])
validities_df = pd.DataFrame(columns=['my_validity'])
def implement_CXGNN(G, node_imp, predicted_classes, data, role_id, num_epochs = None):
    null_batch = torch.zeros(1).long()
    forward_kwargs = {'batch': null_batch}  # Input to explainers forward methods
    forward_kwargs.pop('unwanted_argument', None)
    data1 = pd.DataFrame({'node_label': predicted_classes.tolist()})
    cg = causal.CausalGraph(V=G.nodes, path=G.edges)
    relative_positives = (node_imp == 1).nonzero(as_tuple=True)[0]
    relative_positives = relative_positives.tolist()
    models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = alg2.alg_2(Graph=cg,num_epochs=num_epochs,data=data1,role_id=role_id)
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
    pos = nx.spring_layout(G, seed=1234)
    plt.figure()
    node_colors_new_v = ["#FFE134" if node in best_new_v else "#400040" for node in G.nodes]
    edge_colors = ["red" if u in best_new_v and v in best_new_v else "grey" for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_size=400, font_size=10, font_weight='bold',
            node_color=node_colors_new_v, edge_color=edge_colors)
    plt.show()
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return ({'my_find': my_gt_find},{'my_acc': my_acc},{'my_recall': my_recall * 100},{ 'my_validity': my_validity})

def benzene(dataset = None, num_epochs = None, num_iterations = None):
    total_nodes = 0
    total_edges: int = 0
    start_time = time.time()
    for i in range(num_iterations):
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        if has_one:  # Proceed only if there is a 1 in role_id
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
            # gt_exp = Explanation(feature_imp=feature_imp, node_imp=node_imp, edge_imp=edge_imp, graph=data)
            groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement_CXGNN(G,node_imp, predicted_classes, role_id, num_epochs)

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
def fluoride_carbonyl(dataset = None, num_epochs = None, num_iterations = None):
    total_nodes = 0
    total_edges = 0
    start_time = time.time()
    for i in range(num_iterations):
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()
        if has_one:  # Proceed only if there is a 1 in role_id
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
            groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = implement_CXGNN(G,node_imp,predicted_classes,data,role_id,num_epochs)
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

# fluoride_carbonyl(dataset = dataset_Benzene, num_epochs = 1, num_iterations = 200)

