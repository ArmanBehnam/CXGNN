from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Bernoulli

class CausalGraph:
    def __init__(self, V, path=[], unobserved_edges=[]):
        self.v = list(V)
        self.set_v = set(V)
        self.labels = {node: Bernoulli(0.5).sample((1,)) for node in self.v}
        self.fn = {v: set() for v in V}  # First neighborhood
        self.sn = {v: set() for v in V}  # Second neighborhood
        self.on = {v: set() for v in V}  # Out of neighborhood
        self.p = set(map(tuple, map(sorted, path)))  # Path to First neighborhood
        self.ue = set(map(tuple, map(sorted, unobserved_edges)))  # Unobserved edges

        for v1, v2 in path:
            self.fn[v1].add(v2)
            self.fn[v2].add(v1)
            self.p.add(tuple(sorted((v1, v2))))

    def __iter__(self):
        return iter(self.v)

    def categorize_neighbors(self,target_node):
        # centrality = {v: len(self.fn[v]) for v in self.v}
        # target_node = max(centrality, key=centrality.get)
        if target_node not in self.set_v:
            return

        one_hop_neighbors = self.fn[target_node]
        two_hop_neighbors = set()

        for neighbor in one_hop_neighbors:
            two_hop_neighbors |= self.fn[neighbor]

        two_hop_neighbors -= one_hop_neighbors
        two_hop_neighbors.discard(target_node)
        out_of_neighborhood = self.set_v - (one_hop_neighbors | two_hop_neighbors | {target_node})

        self.sn[target_node] = two_hop_neighbors
        self.on[target_node] = out_of_neighborhood
        return target_node, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood

    def degrees(self):
        # Calculate degrees of nodes in the graph
        return {node: len(self.fn[node]) for node in self.v}
    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.v)
        G.add_edges_from(self.p)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color="lightblue", edge_color="grey")
        plt.savefig('causal.png')
        plt.show()

    def graph_search(self,cg, v1, v2=None, edge_type="path",target_node = None):
        assert edge_type in ["path", "unobserved"]
        assert v1 in cg.set_v
        assert v2 in cg.set_v or v2 is None

        target, one_hop_neighbors, two_hop_neighbors, out_of_neighborhood = cg.categorize_neighbors(target_node)

        q = deque([v1])
        seen = {v1}
        while len(q) > 0:
            cur = q.popleft()
            cur_fn = cg.fn[cur]
            cur_sn = cg.sn[target_node]
            cur_on = cg.on[target_node]

            cur_neighbors = cur_fn if edge_type == "path" else (cur_sn | cur_on)

            for neighbor in cur_neighbors:
                if neighbor not in seen:
                    if v2 is not None:
                        if (neighbor == v2 and edge_type == "path" and neighbor in one_hop_neighbors) or \
                                (neighbor == v2 and edge_type == "unobserved" and neighbor in (
                                        two_hop_neighbors | out_of_neighborhood)):
                            return True
                    seen.add(neighbor)
                    q.append(neighbor)

        if v2 is None:
            return seen

        return False

    def calculate_probabilities(self, dataset):
        node_counts = {node: 0 for node in self.v}
        total_samples = len(dataset)

        for i in dataset:
            for node, value in i.items():
                if value == 1:
                    node_counts[node] += 1

        node_probabilities = {node: count / total_samples for node, count in node_counts.items()}
        return node_probabilities
    def calculate_joint_probabilities(self, dataset):
        joint_counts = {(node_i, node_j): 0 for node_i in self.v for node_j in self.v if node_i != node_j}
        total_samples = len(dataset)

        for sample in dataset:
            for node_i, node_j in joint_counts.keys():
                if sample[node_i] == 1 and sample[node_j] == 1:
                    joint_counts[(node_i, node_j)] += 1

        joint_probabilities = {}
        min_prob = 1  # initialize the min_prob to 1

        # First, calculate the probabilities for the existing links
        for (node_i, node_j), count in joint_counts.items():
            if (node_i, node_j) in self.p or (node_j, node_i) in self.p:
                prob = count / total_samples
                joint_probabilities[(node_i, node_j)] = prob
                joint_probabilities[(node_j, node_i)] = prob  # update for bidirectional link
                if prob < min_prob:
                    min_prob = prob  # update the minimum probability

        # Now, calculate the probabilities for the non-existing links using the Gumbel distribution
        for (node_i, node_j), count in joint_counts.items():
            if (node_i, node_j) not in self.p and (node_j, node_i) not in self.p:
                # generate a random value from a Gumbel distribution
                gumbel_noise = np.random.gumbel()
                # rescale the gumbel noise to be in [0, min_prob)
                # scaled_gumbel_noise = min_prob * (gumbel_noise - np.min(gumbel_noise)) / (np.max(gumbel_noise) - np.min(gumbel_noise))
                joint_probabilities[(node_i, node_j)] = gumbel_noise
                joint_probabilities[(node_j, node_i)] = gumbel_noise  # update for bidirectional link

        return joint_probabilities
