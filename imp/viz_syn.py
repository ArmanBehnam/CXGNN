import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
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
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
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
def build_graph(width_basis,basis_type,Ground_truth,start=0,rdm_basis_plugins=False,add_random_edges=0,m=5):
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
        if basis_type == "ba":
            shape_connecting_node = np.random.choice(list(graph_s.nodes()))
            basis_connecting_node = plugins[shape_id]
            basis.add_edges_from([(shape_connecting_node, basis_connecting_node)])
        else:
            basis.add_edges_from([(start, plugins[shape_id])])

        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
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
def generate_random_graph(basis_type="ba", Ground_truth="grid"):
    num_shapes = 1
    width_basis = np.random.randint(5, 15)  # Random width for the basis
    m = np.random.randint(2, 4)  # Random 'm' for BA graph
    G, role_id, _ = build_graph(width_basis, basis_type, Ground_truth, start=0, m=m)
    G = perturb([G], 0.01)[0]
    name = basis_type + "_" + str(width_basis) + "_" + str(num_shapes)
    # nx.draw(G, node_size=20, with_labels=False,
    #         node_color=["red" if role_id[node] == 1 else "blue" for node in G.nodes()])
    # plt.show()
    return G, role_id, name

num_iterations = 10
graphs_per_figure = 3
for i in range(0, num_iterations, graphs_per_figure):
    plt.figure(figsize=(10, 12), dpi=300)
    for j in range(graphs_per_figure):
        G, role_id, name = generate_random_graph(basis_type="ba", Ground_truth="cycle")
        plt.subplot(graphs_per_figure,1, j+1)
        nx.draw(G, node_size=30, with_labels=False, node_color=["red" if role_id[node] == 1 else "blue" for node in G.nodes()])
    plt.tight_layout()
    plt.show()