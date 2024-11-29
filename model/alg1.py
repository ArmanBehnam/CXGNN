import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
class NNModel(nn.Module):
    def __init__(self, input_size, output_size, h_size, h_layers):
        super(NNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size
        self.h_layers = h_layers
        layers = [nn.Linear(self.input_size, self.h_size), nn.ReLU()]
        for _ in range(h_layers - 1):
            layers += [nn.Linear(self.h_size, self.h_size), nn.ReLU()]
        layers.append(nn.Linear(self.h_size, self.output_size))
        self.nn = nn.Sequential(*layers)
        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
    def forward(self, u):
        return torch.sigmoid(self.nn(u))
class NCM:
    def __init__(self, graph, target_node, learning_rate, h_size, h_layers, data):
        self.graph = graph
        self.h_size = h_size
        self.h_layers = h_layers
        self.learning_rate = learning_rate
        self.target_node = target_node
        self.states = {graph.target_node: torch.tensor([graph.labels[target_node]], dtype=torch.float32)}
        self.u_i = {v: torch.tensor([data.loc[v, 'node_label']], dtype=torch.float32) for v in graph.one_hop_neighbors | graph.two_hop_neighbors}
        self.u_ij = {v: torch.tensor([data.loc[v, 'node_label']], dtype=torch.float32) for v in graph.one_hop_neighbors | graph.two_hop_neighbors}
        self.u = torch.cat(list(self.states.values()) + list(self.u_i.values()) + list(self.u_ij.values()), dim=0)
        self.model = NNModel(input_size=len(self.u), output_size=1, h_size=h_size, h_layers=h_layers)
        self.ratio = len(self.graph.p) / len(self.graph.v)

    def add_gaussian_noise(self, tensor, mean=0.0, std=0.01):
        noise = torch.randn(tensor.size()) * std + mean
        return torch.clamp(tensor + noise, 0, 1)

    def ncm_forward(self, add_noise=False):
        if add_noise:
            for k in self.u_i:
                self.u_i[k] = self.add_gaussian_noise(self.u_i[k])
            for k in self.u_ij:
                self.u_ij[k] = self.add_gaussian_noise(self.u_ij[k])
            self.u = torch.cat(list(self.states.values()) + list(self.u_i.values()) + list(self.u_ij.values()), dim=0)
        f = self.model.forward(self.u)  # Pass self.u here
        return torch.sigmoid(f)
def calculate_prob(graph, f, target_node):
    nodes_n1_n2 = graph.fn[target_node] | graph.sn[target_node]
    if not nodes_n1_n2:
        return 0.0  # Return 0 if no neighbors are present
    sum_prob = 0.0
    for v_j in nodes_n1_n2:
        # product = 1.0
        if (target_node, v_j) in graph.p or (v_j, target_node) in graph.p:
            # product *= f
            sum_prob += f.item()
    probability = sum_prob / len(nodes_n1_n2) if nodes_n1_n2 else 0.0
    # print(f"Debug: Calculated Probability for Node {target_node}: {probability}")
    return probability
def calculate_expected_prob(cg, P_do,label_probs):
    expected_value = 0.0
    for y, y_prob in label_probs.items():
        inner_sum = 0.0
        for v_i in cg.one_hop_neighbors:
            inner_sum += P_do
        expected_value += y_prob * inner_sum
        return expected_value / len(cg.one_hop_neighbors) if cg.one_hop_neighbors else 0.0
def compute_probability_of_node_label(cg, target_node, role_id):
    unique_labels = np.unique(role_id)
    num_nodes = len(role_id)
    all_combinations = product(unique_labels, repeat=num_nodes)
    label_probabilities = {label: 0 for label in unique_labels}
    for combination in all_combinations:
        temp_role_id = list(combination)
        current_label = temp_role_id[target_node]
        label_probabilities[current_label] += 1
    total_combinations = len(unique_labels) ** num_nodes
    for label, count in label_probabilities.items():
        label_probabilities[label] = count / total_combinations
    return label_probabilities
def train(cg, learning_rate, h_size, h_layers, num_epochs, data, role_id,target_node):
    cg.target_node, cg.one_hop_neighbors, cg.two_hop_neighbors, cg.out_of_neighborhood = cg.categorize_neighbors(target_node=target_node)
    ncm = NCM(cg, target_node, learning_rate=learning_rate, h_size=h_size, h_layers=h_layers, data=data)
    optimizer = optim.Adam(ncm.model.parameters(), lr=ncm.learning_rate)
    # new_v = {cg.target_node}.union(cg.one_hop_neighbors).union(cg.two_hop_neighbors) # For syn and real
    new_v = {cg.target_node}.union(cg.one_hop_neighbors) # For appendix example
    loss_history = []  # List to store loss at each epoch
    for i in range(num_epochs):
        f = ncm.ncm_forward(add_noise=True)
        P_do = calculate_prob(cg, f, cg.target_node)
        label_probs = compute_probability_of_node_label(cg, cg.target_node, role_id)
        expected_p = calculate_expected_prob(cg, P_do,label_probs)
        expected_p_tensor = torch.tensor([expected_p], dtype=torch.float32) if isinstance(expected_p,float) else expected_p
        output = (expected_p_tensor.clone().detach() >= 0.05).float()
        loss = torch.nn.functional.binary_cross_entropy(f.view(1),torch.tensor([role_id[target_node]], dtype=torch.float).view(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())  # Record the loss
    return loss_history, loss, ncm.model.state_dict(), expected_p, output, new_v