import causal
import alg2
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
nodes = ['A', 'B', 'C', 'D']
edges = [('A', 'B'), ('A', 'C'), ('B', 'D')]
# Define the nodes and the connections (paths) between them
cg = causal.CausalGraph([0,1,2,3], [(0, 1), (0, 2), (1, 3)])
# Manually set predicted classes for A, B, C as 1 and D as 0
role_id = [1, 1, 1, 0]
# Example of initializing other variables
role_id_tensor = torch.tensor(role_id)  # Example, replace with actual logic
num_epochs = 10  # Example, set to your desired number of epochs
# Prepare your data (assuming 'predicted_classes' and 'node_imp' are already defined)
data1 = pd.DataFrame({'node_label': role_id}, index=cg.v)
# Assuming relative_positives and role_id are already defined
relative_positives = (role_id_tensor == 1).nonzero(as_tuple=True)[0].tolist()
# Train the model using the provided algorithm
models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node = alg2.alg_2(Graph=cg, num_epochs=num_epochs, data=data1, role_id=role_id)
# Set node and edge colors based on the model results
print(best_new_v)
node_colors_new_v = ["#FFE134" if node in best_new_v else "#400040" for node in cg.v]
edge_colors = ["red" if u in best_new_v and v in best_new_v else "grey" for u, v in cg.p]
# edge_colors = ["red" if (u, v) in [(edges[i][0], edges[i][1]) for i in best_new_v] else "grey" for u, v in edges]

# Draw the graph
# Create a NetworkX graph from edges for visualization
G = nx.Graph()
G.add_nodes_from(cg.v)
G.add_edges_from(cg.p)
pos = nx.spring_layout(G, seed=1234)
nx.draw(G, pos, with_labels=True, node_size=400, font_size=10, font_weight='bold',node_color=node_colors_new_v, edge_color=edge_colors)
plt.show()
print('The ground truth is: ', relative_positives)
print('Our method finding is: ', best_new_v)
# Calculate recall, accuracy, and validity
my_predictions = torch.zeros_like(role_id_tensor)
my_predictions[list(best_new_v) if isinstance(best_new_v, set) else best_new_v] = 1

relative_positives_tensor = torch.zeros_like(role_id_tensor)
relative_positives_tensor[relative_positives] = 1

predicted_tensor = torch.zeros_like(role_id_tensor)
predicted_tensor[list(best_new_v)] = 1

my_recall = torch.sum(predicted_tensor[relative_positives]).item() / len(relative_positives)
my_acc = (torch.sum((role_id_tensor == 1) & (my_predictions == 1)).item()) / (torch.sum(role_id_tensor == 1).item()) * 100
my_gt_find = int(set(best_new_v) == set(relative_positives))
my_validity = int(all(item in relative_positives for item in best_new_v))

print(f"My Explanation recall is: {my_recall * 100:.2f}%")
print("My Explanation found the ground truth? ", my_gt_find)
print("Is our final subgraph valid? ", my_validity)
print(f"How well the ground truth is found? {my_acc:.2f}%\n")