from imp_syn import build_graph, perturb, from_networkx_to_torch, GIN
import causal
import time
import os
from PIL import Image
import pandas as pd
import torch
from graphxai.utils.explanation import Explanation
import matplotlib.pyplot as plt
from alg2 import alg_2
import random
from matplotlib.lines import Line2D
from graphxai.datasets import AlkaneCarbonyl, Benzene, FluorideCarbonyl
from imp_real import preprocess, implement_CXGNN
def plot_CXGNN(G, node_imp, predicted_classes, role_id, num_epochs = None, iter_num = 0, base_path = None):
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
    successful_path = os.path.join(base_path, 'successful')
    unsuccessful_path = os.path.join(base_path, 'unsuccessful')
    os.makedirs(successful_path, exist_ok=True)
    os.makedirs(unsuccessful_path, exist_ok=True)
    loss_data_ground_truth_nodes = {}  # Fill with actual loss data for ground truth nodes
    loss_data_non_ground_truth_nodes = {}
    plt.figure(figsize=(12, 8))
    vis_loss_1 = []
    vis_loss_2 = []
    # fig, axs = plt.subplots(2, 5, figsize=(20, 10))  # 5x2 grid of subplots
    gt_nodes = sorted(relative_positives, key=lambda node: models[node]['expected_p'], reverse=True)[:5]
    non_gt_nodes = random.sample(list(set(G.nodes) - set(relative_positives)),min(5, len(set(G.nodes) - set(relative_positives))))
    if my_gt_find == 1:
        print('yes')
        for node in gt_nodes:
            loss_data_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='green')
            vis_loss_1.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        # non_gt_nodes = set(G.nodes) - set(relative_positives)
        # selected_non_gt_nodes = random.sample(list(non_gt_nodes), min(5, len(list(non_gt_nodes))))
        for node in non_gt_nodes:
            loss_data_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='red')
            vis_loss_2.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves Successful explanation')
        legend_handles = [
            Line2D([0], [0], color='green', lw=4, label='Ground Truth Nodes'),
            Line2D([0], [0], color='red', lw=4, label='Non-Ground Truth Nodes')
        ]
        # Add the legend to the plot
        plt.legend(handles=legend_handles)
        plt.savefig(os.path.join(successful_path, f'Loss_Curves_Successful_explanation_iter_{iter_num}.png'))

        # plt.savefig(f'{successful_path}Loss_Curves_Successful_explanation_iter_{iter_num}.png')
        # plt.show()
    else:
        print('No')
        for node in gt_nodes:
            loss_data_non_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_non_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='green')
            vis_loss_1.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        for node in non_gt_nodes:
            loss_data_non_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_non_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='red')
            vis_loss_2.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves Unsuccessful explanation')
        legend_handles = [
            Line2D([0], [0], color='green', lw=4, label='Ground Truth Nodes'),
            Line2D([0], [0], color='red', lw=4, label='Non-Ground Truth Nodes')
        ]
        # Add the legend to the plot
        plt.legend(handles=legend_handles)
        plt.savefig(os.path.join(unsuccessful_path, f'Loss_Curves_Unsuccessful_explanation_iter_{iter_num}.png'))
        # plt.savefig(f'{unsuccessful_path}Loss_Curves_Unuccessful_explanation_iter_{iter_num}.png')
    my_validity = int(all(item in relative_positives for item in best_new_v))
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return ({'my_find': my_gt_find},{'my_acc': my_acc},{'my_recall': my_recall * 100},{ 'my_validity': my_validity})
def plot_loss_1(G, model, node_imp, predicted_classes, data, role_id, num_epochs = None, iter_num = 0, base_path = None):
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
    successful_path = os.path.join(base_path, 'successful')
    unsuccessful_path = os.path.join(base_path, 'unsuccessful')
    os.makedirs(successful_path, exist_ok=True)
    os.makedirs(unsuccessful_path, exist_ok=True)
    loss_data_ground_truth_nodes = {}  # Fill with actual loss data for ground truth nodes
    loss_data_non_ground_truth_nodes = {}
    plt.figure(figsize=(12, 8))
    vis_loss_1 = []
    vis_loss_2 = []
    # fig, axs = plt.subplots(2, 5, figsize=(20, 10))  # 5x2 grid of subplots
    gt_nodes = sorted(relative_positives, key=lambda node: models[node]['expected_p'], reverse=True)[:5]
    non_gt_nodes = random.sample(list(set(G.nodes) - set(relative_positives)),min(5, len(set(G.nodes) - set(relative_positives))))
    if my_gt_find == 1:
        print('yes')
        for node in gt_nodes:
            loss_data_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='green')
            vis_loss_1.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        for node in non_gt_nodes:
            loss_data_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='red',linestyle='--', dashes=[5, 10])

            vis_loss_2.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.title('Loss Curves Successful explanation')
        legend_handles = [
            Line2D([0], [0], color='red', lw=4, label='Non-Ground Truth Nodes', linestyle='--'),
            Line2D([0], [0], color='green', lw=4, label='Ground Truth Nodes')
        ]
        plt.legend(handles=legend_handles)
        plt.savefig(os.path.join(successful_path, f'Loss_Curves_Successful_explanation_iter_{iter_num}.png'))
        # plt.show()
    else:
        print('No')
        for node in gt_nodes:
            loss_data_non_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_non_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='green')
            vis_loss_1.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        for node in non_gt_nodes:
            loss_data_non_ground_truth_nodes[node] = models[node]['loss_history']
            plt.plot(loss_data_non_ground_truth_nodes[node], label=f'Node {node} (Ground Truth)',color='red',linestyle='--', dashes=[5, 10])
            vis_loss_2.append(models[node]['loss_history'])
            print(node, models[node]['loss_history'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.title('Loss Curves Unsuccessful explanation')
        legend_handles = [
            Line2D([0], [0], color='green', lw=4, label='Ground Truth Nodes'),
            Line2D([0], [0], color='red', lw=4, label='Non-Ground Truth Nodes')
        ]
        # Add the legend to the plot
        plt.legend(handles=legend_handles)
        plt.savefig(os.path.join(unsuccessful_path, f'Loss_Curves_Unsuccessful_explanation_iter_{iter_num}.png'))
    # plt.show()
    my_validity = int(all(item in relative_positives for item in best_new_v))
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return ({'my_find': my_gt_find},{'my_acc': my_acc},{'my_recall': my_recall * 100},{ 'my_validity': my_validity})
def plot_loss_2(G, node_imp, predicted_classes, role_id, num_epochs = None, iter_num = 0, base_path = None):
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
    data_path = os.path.join(base_path)
    expected_p_nodes = {}
    if my_gt_find == 0:
        for node in set(G.nodes) :
            expected_p_nodes[node] = models[node]['expected_p']
        plt.figure(figsize=(12, 8))
        sorted_items = sorted(expected_p_nodes.items(), key=lambda x: x[1], reverse=True)
        sorted_nodes, sorted_values = zip(*sorted_items)
        colors = ['green' if node in relative_positives else 'red' for node in sorted_nodes]
        # plt.bar(expected_p_nodes.keys(), expected_p_nodes.values(), color=colors)
        # plt.bar(range(len(sorted_nodes)), sorted_values, color=colors)  # Use sorted values
        plt.bar(range(len(sorted_nodes)), sorted_values, color=colors)
        plt.xticks(range(len(sorted_nodes)), sorted_nodes)
        plt.xlabel('Nodes')
        plt.ylabel('Expected Value')
        plt.savefig(os.path.join(data_path, f'Node_experssivity_distribution_iter_{iter_num}.png'))
        # plt.show()
    my_validity = int(all(item in relative_positives for item in best_new_v))
    print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    print("My Explanation found the ground truth? ", my_gt_find)
    print("Is our final subgraph valid? ", my_validity)
    print("How well the ground truth is found? {:.2f}".format(my_acc), '\n')
    # ------------------------------------------------------
    # plt.show()
    return (expected_p_nodes,best_node, {'my_find': my_gt_find},{'my_acc': my_acc},{'my_recall': my_recall * 100},{ 'my_validity': my_validity})

groundtruth_find_df = pd.DataFrame(columns=['PGME_find', 'guidedbp_find', 'ig_find', 'grad_find', 'gnnex_find', 'my_find'])
graph_explanation_accuracy_df = pd.DataFrame(columns=['PGME_acc', 'guidedbp_acc', 'ig_acc', 'grad_acc', 'gnnex_acc', 'my_acc'])
graph_explanation_recall_df = pd.DataFrame(columns=['PGME_recall', 'guidedbp_recall', 'ig_recall', 'grad_recall', 'gnnex_recall', 'my_recall'])
validities_df = pd.DataFrame(columns=['PGME_validity', 'guidedbp_validity', 'ig_validity', 'grad_validity', 'gnnex_validity', 'my_validity'])
def loss_fig_1(basis_type=None, Ground_truth=None, width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
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
        base_path = 'D:\\University\\Research\\loss_1'
        groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = plot_loss_1(G, model,
                                                                                                       node_imp,
                                                                                                       predicted_classes,
                                                                                                       data, role_id,
                                                                                                       num_epochs=num_epochs, iter_num=i, base_path = base_path)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
    successful_path = r'D:\University\Research\loss_1\successful'
    unsuccessful_path = r'D:\University\Research\loss_1\unsuccessful'
    successful_images = [os.path.join(successful_path, f) for f in os.listdir(successful_path) if f.endswith('.png')]
    unsuccessful_images = [os.path.join(unsuccessful_path, f) for f in os.listdir(unsuccessful_path) if f.endswith('.png')]
    successful_images.sort()
    unsuccessful_images.sort()
    successful_images = successful_images[:2]
    unsuccessful_images = unsuccessful_images[:2]
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))  # Adjust the size as needed
    def plot_images_in_subplot(image_paths, start_row):
        for idx, image_path in enumerate(image_paths):
            col = idx % 2
            image = Image.open(image_path)
            axs[start_row, col].imshow(image)
            axs[start_row, col].axis('off')  # Hide the axis
    plot_images_in_subplot(successful_images, 0)
    plot_images_in_subplot(unsuccessful_images, 1)
    plt.tight_layout()
    plt.savefig(r'D:\University\Research\loss_1\combined_figure.png', dpi=500)  # Save the combined figure
    plt.show()
def loss_fig_2(basis_type=None, Ground_truth=None, width_basis = None, m = None, num_epochs = None, num_iterations = None):
    num_shapes = 1
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    total_nodes = 0
    total_edges = 0
    total_node = []
    total_exp = []
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
        image_paths = 'D:\\University\\Research\\loss_2\\'
        exp, node, groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = plot_loss_2(G,
                                                                                                       node_imp,
                                                                                                       predicted_classes, role_id,
                                                                                                       num_epochs=num_epochs, iter_num=i, base_path = image_paths)
        graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
        groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
        graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
        validities_df.loc[len(validities_df)] = validities
        total_node.append(node)
        total_exp.append(exp)
        plt.show()

# loss_fig_2(basis_type="tree", Ground_truth="house", width_basis = 3, m = 3, num_epochs = 5, num_iterations = 100)

data_path_benzene = 'datasets/real_world/benzene/benzene.npz'
# data_path_FluorideCarbonyl = 'datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)
def loss_fig_3(dataset = None, num_epochs = None, num_iterations = None):
    for i in range(num_iterations):
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
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
            base_path = 'D:\\University\\Research\\loss_3\\'
            groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = plot_loss_1(G, model,
                                                                                                             node_imp,
                                                                                                             predicted_classes,
                                                                                                             data,
                                                                                                             role_id,
                                                                                                             num_epochs=num_epochs,
                                                                                                             iter_num=i,
                                                                                                             base_path=base_path)

            graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
            groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
            graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
            validities_df.loc[len(validities_df)] = validities
    successful_path = r'D:\University\Research\loss_3\successful'
    unsuccessful_path = r'D:\University\Research\loss_3\unsuccessful'
    successful_images = [os.path.join(successful_path, f) for f in os.listdir(successful_path) if f.endswith('.png')]
    unsuccessful_images = [os.path.join(unsuccessful_path, f) for f in os.listdir(unsuccessful_path) if f.endswith('.png')]
    successful_images.sort()
    unsuccessful_images.sort()
    successful_images = successful_images[:2]
    unsuccessful_images = unsuccessful_images[:2]
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))  # Adjust the size as needed
    def plot_images_in_subplot(image_paths, start_row):
        for idx, image_path in enumerate(image_paths):
            col = idx % 2
            image = Image.open(image_path)
            axs[start_row, col].imshow(image)
            axs[start_row, col].axis('off')  # Hide the axis
    plot_images_in_subplot(successful_images, 0)
    plot_images_in_subplot(unsuccessful_images, 1)
    plt.tight_layout()
    plt.show()
def loss_fig_4(dataset = None, num_epochs = None, num_iterations = None):
    for i in range(num_iterations):
        data, G, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
        if has_one:  # Proceed only if there is a 1 in role_id
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
            image_paths = 'D:\\University\\Research\\loss_4\\'
            exp, node, groundtruth_find, graph_explanation_accuracy, graph_explanation_recall, validities = plot_loss_2(G = G,
                                                                                                       node_imp = node_imp,predicted_classes = predicted_classes,role_id=role_id,
                                                                                                       num_epochs=num_epochs, iter_num=i, base_path = image_paths)
            graph_explanation_accuracy_df.loc[len(graph_explanation_accuracy)] = graph_explanation_accuracy
            groundtruth_find_df.loc[len(groundtruth_find_df)] = groundtruth_find
            graph_explanation_recall_df.loc[len(graph_explanation_recall_df)] = graph_explanation_recall
            validities_df.loc[len(validities_df)] = validities
            plt.show()

# loss_fig_3(dataset = dataset_FluorideCarbonyl, num_epochs = 20, num_iterations = 100)
loss_fig_3(dataset = dataset_Benzene, num_epochs = 20, num_iterations = 100)
