from torch.nn import Sequential, Linear, ReLU, ELU
import logging
class Logger:
    logger = None

    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger

    @staticmethod
    def init_logger(level=logging.INFO,
                    fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)

        fmt = logging.Formatter(fmt)

        # stream handler
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if filename:
            # file handler
            fh = logging.FileHandler(filename)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        logger.setLevel(level)
        Logger.logger = logger
        return logger
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Reinforced Screener")

    # # ===== dataset ===== #
    # parser.add_argument("--dataset", nargs="?", default="mutag", help="Choose a dataset:[last-fm,amazon-book,yelp2018]")
    #
    # # ===== train ===== #
    # parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--dim', type=int, default=64, help='embedding size')
    # parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    # parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    # parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    # parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    # parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    # parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    # parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    # parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    # parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    # parser.add_argument("--gpu_id", type=int, default=6, help="gpu id")
    # parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    # parser.add_argument('--test_flag', nargs='?', default='part',
    #                     help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    # parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    #
    # # ===== relation context ===== #
    # parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--dataset_name", type=str, default="vg", help="sigmoid, softmax")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--reward_mode", type=str, default="binary", help="cross_entropy, binary")

    # parser.add_argument("--edge_scoring_type", type=str, default="sigmoid", help="sigmoid, softmax")
    # parser.add_argument("--reward_type", type=str, default="binary", help="cross_entropy, binary")
    # parser.add_argument("--reward_discount_type", type=str, default="ascending", help="ascending, null, descending")
    # parser.add_argument("--optimize_scope", type=str, default="all", help="all, part")
    # parser.add_argument("--multiple_explainers", type=bool, default=True, help="use multiple or single explainer(s)")

    return parser.parse_args()
import copy
from tqdm import tqdm
from torch_geometric.data import DataLoader
def filter_correct_data(model, dataset, loader, flag='Training', batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
    idx = 0
    for g in tqdm(iter(loader), total=len(loader)):
        g.to(device)
        model(g.x, g.edge_index, g.edge_attr, g.batch)
        if g.y == model.readout.argmax(dim=1):
            graph_mask[idx] = True
        idx += 1

    loader = DataLoader(dataset[graph_mask], batch_size=1, shuffle=False)
    print("number of graphs in the %s:%4d" % (flag, graph_mask.nonzero().size(0)))
    return dataset, loader
def filter_correct_data_batch(model, dataset, loader, flag='training', batch_size=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = []
    for g in tqdm(iter(loader), total=len(loader)):
        g.to(device)
        model(g.x, g.edge_index, g.edge_attr, g.batch)

        tmp = (g.y == model.readout.argmax(dim=1))
        graph_mask += tmp.tolist()

    # must convert the graph_mask to the boolean tensor.
    graph_mask = torch.BoolTensor(graph_mask)

    shuffle_flag = False
    if flag == 'training':
        shuffle_flag = True

    loader = DataLoader(dataset[graph_mask], batch_size=batch_size, shuffle=shuffle_flag)
    print("number of graphs in the %s:%4d" % (flag, sum(graph_mask)))
    return dataset, loader
def relabel_graph(graph, selection):
    subgraph = copy.deepcopy(graph)

    # Retrieval properties of the explanatory subgraph
    # .... the edge_index.
    subgraph.edge_index = graph.edge_index.T[selection].T

    # .... the edge_attr.
    if graph.edge_attr is not None:
        subgraph.edge_attr = graph.edge_attr[selection]
    else:
        subgraph.edge_attr = None

    # .... the nodes.
    sub_nodes = torch.unique(subgraph.edge_index)
    # .... the node features.
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    row, col = graph.edge_index
    # Remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]

    return subgraph
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
import os.path as osp
from torch_geometric.nn import global_mean_pool
device = "cpu"
from imp_syn import *
from imp_real import *

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

    def get_node_reps(self, x, edge_index, edge_attr=None, batch=None):
        # Implement the logic to get node representations
        # This can be the output of one of the layers, or some other computation
        # Example implementation:
        x = F.relu(self.conv1(x, edge_index))
        return x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        # Implement your logic here
        # For example, apply a GNN layer and then global pooling:
        x = self.conv1(x, edge_index)  # Example: Apply one GNN layer
        x = global_mean_pool(x, batch)  # Example: Global mean pooling
        return x
class RC_Explainer(torch.nn.Module):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False, edge_attr_dim=0):
        super(RC_Explainer, self).__init__()

        self.model = _model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

        self.num_labels = _num_labels
        self.hidden_size = _hidden_size
        self.use_edge_attr = _use_edge_attr

        # Define the input dimension for the edge_action_rep_generator
        input_dim = self.hidden_size * 2
        if _use_edge_attr:
            input_dim += edge_attr_dim  # Add edge attribute dimension if used

        # Edge action representation generator
        self.edge_action_rep_generator = Sequential(
            Linear(input_dim, self.hidden_size * 2),
            ReLU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        )

        # Edge action probability generator
        # The input to this layer must match the output of the edge_action_rep_generator
        self.edge_action_prob_generator = Sequential(
            Linear(self.hidden_size, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.num_labels)
        )
    def build_edge_action_prob_generator(self):
        input_size = self.hidden_size * 2  # The output size of edge_action_rep_generator
        if self.use_edge_attr:
            edge_attr_dim = 32  # edge attribute dimension
            input_size += edge_attr_dim

        edge_action_prob_generator = Sequential(
            Linear(input_size, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.num_labels)
        ).to(device)
        return edge_action_prob_generator

    def forward(self, graph, state, target_y):
        num_edges = graph.edge_index.size(1)
        target_y = torch.randint(0, 2, (num_edges,))

        edge_index = graph.edge_index
        edge_attr = graph.edge_attr if graph.edge_attr is not None else None
        node_reps = self.model.get_node_reps(graph.x, edge_index, edge_attr, graph.batch)
        action_reps = torch.cat([node_reps[edge_index[0]], node_reps[edge_index[1]]], dim=1)
        action_reps = self.edge_action_rep_generator(action_reps.to(device))

        action_probs = self.predict(action_reps, target_y)

        # print("Action probabilities:", action_probs)
        # print("Size of action probabilities:", action_probs.size())

        # No thresholding, considering all edges as important for debugging
        important_edges = torch.arange(action_probs.size(0))
        # print("Important edges:", important_edges)
        # print("Number of important edges:", important_edges.size(0))

        return action_probs, important_edges

    def predict(self, ava_action_reps, target_y):
        action_probs = self.edge_action_prob_generator(ava_action_reps)

        # print("Action representation shape:", ava_action_reps.shape)
        # print("Action probabilities shape:", action_probs.shape)

        # Reshape target_y for compatibility with gather
        target_y_reshaped = target_y.view(-1, 1)
        # print("Target_y reshaped shape:", target_y_reshaped.shape)

        # Ensure the number of actions (rows) match between action_probs and target_y
        if target_y_reshaped.size(0) != action_probs.size(0):
            raise ValueError(
                f"Dimension mismatch: target_y length {target_y_reshaped.size(0)} does not match the number of actions {action_probs.size(0)}")

        # Use gather to select relevant probabilities
        selected_action_probs = action_probs.gather(1, target_y_reshaped).squeeze(1)

        # Apply softmax to the selected probabilities
        action_probs = F.softmax(selected_action_probs, dim=-1)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5):
        params = list(self.edge_action_rep_generator.parameters()) + \
                 list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer
class RC_Explainer_pro(RC_Explainer):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_pro, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = torch.nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)).to(device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def predict(self, ava_action_reps, target_y):
        action_probs = []
        for i in range(target_y.size(0)):
            i_explainer = self.edge_action_prob_generator[target_y[i].item()]
            i_action_prob = i_explainer(ava_action_reps[i].unsqueeze(0))
            action_probs.append(i_action_prob)

        action_probs = torch.cat(action_probs, dim=0)
        action_probs = action_probs.reshape(-1)
        action_probs = F.softmax(action_probs, dim=0)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5):
        params = list(self.edge_action_rep_generator.parameters())

        for i_explainer in self.edge_action_prob_generator:
            params += list(i_explainer.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer
class RC_Explainer_Batch(torch.nn.Module):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_Batch, self).__init__()

        self.model = _model
        self.model = self.model.to(device)
        self.model.eval()

        self.num_labels = _num_labels
        self.hidden_size = _hidden_size
        self.use_edge_attr = _use_edge_attr

        self.temperature = 0.1

        input_dim = 64  # This should be adjusted to match your actual input size

        self.edge_action_rep_generator = Sequential(
            Linear(input_dim, self.hidden_size * 4),
            ELU(),
            Linear(self.hidden_size * 4, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        ).to(device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = Sequential(
            Linear(self.hidden_size, self.hidden_size),
            ELU(),
            Linear(self.hidden_size, self.num_labels)
        ).to(device)
        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        if isinstance(train_flag, torch.Tensor):
            train_flag = torch.any(train_flag).item()

        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state] if graph.edge_attr is not None else None

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state] if graph.edge_attr is not None else None

        ava_node_reps_0 = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        ava_node_reps_1 = self.model.get_node_reps(graph.x, ocp_edge_index, ocp_edge_attr, graph.batch)
        ava_node_reps = ava_node_reps_0 - ava_node_reps_1

        ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                     ava_node_reps[ava_edge_index[1]]], dim=1).to(device)
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)
        if graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(graph.x.device)
        ava_action_batch = graph.batch[ava_edge_index[0]]
        if graph.y is None:
            graph.y = torch.zeros(graph.num_nodes, dtype=torch.long).to(graph.x.device)

        ava_y_batch = graph.y[ava_action_batch]


        if self.use_edge_attr and ava_edge_attr is not None:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, ava_edge_reps], dim=1)

        ava_action_probs = self.predict(ava_action_reps, ava_y_batch, ava_action_batch)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions

        return ava_action_probs, added_action_probs, added_actions

    def predict(self, ava_action_reps, target_y, ava_action_batch):
        action_probs = self.edge_action_prob_generator(ava_action_reps)
        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        action_probs = softmax(action_probs, ava_action_batch)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

    def load_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        self.load_state_dict(torch.load(path))

    def save_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        torch.save(self.state_dict(), path)
class RC_Explainer_Batch_star(RC_Explainer_Batch):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_Batch_star, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = torch.nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)
            ).to(device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        if isinstance(train_flag, torch.Tensor):
            if train_flag.numel() == 1:
                # Convert single-element tensor to Python boolean
                train_flag = train_flag.item() > 0
            else:
                # Handle tensors with more than one element (adjust as needed)
                train_flag = train_flag.any().item() > 0
        graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        if len(torch.where(state==True)[0]) == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state] if graph.edge_attr is not None else None

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state] if graph.edge_attr is not None else None
        ava_node_reps = self.model.get_node_reps(graph.x, ava_edge_index, ava_edge_attr, graph.batch)

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]],
                                         ava_edge_reps], dim=1).to(device)
        else:

            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]]], dim=1).to(device)

        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch]

        # get the unique elements in batch, in cases where some batches are out of actions.
        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)

        ava_action_probs = self.predict_star(graph_rep, subgraph_rep, ava_action_reps, ava_y_batch, ava_action_batch)

        # assert len(ava_action_probs) == sum(~state)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions, unique_batch

        return ava_action_probs, added_action_probs, added_actions, unique_batch

    def predict_star(self, graph_rep, subgraph_rep, ava_action_reps, target_y, ava_action_batch):
        action_graph_reps = graph_rep - subgraph_rep
        action_graph_reps = action_graph_reps[ava_action_batch]
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        # action_probs = softmax(action_probs, ava_action_batch)
        # action_probs = F.sigmoid(action_probs)
        return action_probs
def run_rc_explainer_on_syn_graph(G, role_id):
    graph_data = from_networkx_to_torch(G, role_id)
    model = GIN(1, 32, 2)
    model.eval()
    with torch.no_grad():
        predicted_classes = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
    state = torch.rand(len(graph_data.edge_index[0])) > 0.5
    explainer = RC_Explainer(model, 2, 32, True)
    _, edge_importances = explainer(graph_data, state, predicted_classes)
    if edge_importances.dtype != torch.float32 and edge_importances.dtype != torch.float64:
        edge_importances = edge_importances.float()
        # print("method output", edge_importances)
    importance_threshold = edge_importances.mean()
    prediction = torch.zeros(G.number_of_nodes(), dtype=torch.float)
    for idx, edge in enumerate(graph_data.edge_index.T):
        if idx < edge_importances.size(0) and edge_importances[idx] > importance_threshold:
            node_u, node_v = edge[0].item(), edge[1].item()
            prediction[node_u] = 1
            prediction[node_v] = 1
    relative_positives = (torch.tensor(role_id, dtype=torch.float) == 1).nonzero(as_tuple=True)[0]
    print("ground_truth", role_id)
    print("prediction", prediction)
    # print("relative_positives", relative_positives)
    # print(torch.sum(prediction[relative_positives]).item(), torch.sum(torch.tensor(role_id) == 1), len(relative_positives))
    rc_recall = torch.sum(prediction[relative_positives]).item() / torch.sum(prediction == 1)
    print(rc_recall)
    rc_acc = torch.sum(prediction[relative_positives]).item() / torch.sum(torch.tensor(role_id) == 1)
    print(rc_acc)
    prediction_new = [int(item) for item in prediction]
    # print("prediction_new", prediction_new)
    rc_gt_find = int(prediction_new == role_id)
    # rc_validity = int(all(item in relative_positives for item in prediction.nonzero().flatten()))
    return rc_recall, rc_acc, rc_gt_find
def run_experiments_syn(num_iterations=None, width_basis=None, m=None):
    recalls, accuracies, gt_finds = [], [], []
    for _ in range(num_iterations):
        G, role_id, _ = build_graph(width_basis, "ba", "grid", 0, m=2)
        G = perturb([G], 0.01)[0]
        recall, accuracy, gt_find = run_rc_explainer_on_syn_graph(G, role_id)
        recalls.append(recall)
        accuracies.append(accuracy)
        # accuracies1.append(accuracy1)
        gt_finds.append(gt_find)
        # validities.append(validity)

    # Calculate averages
    avg_recall = sum(recalls) / num_iterations
    avg_accuracy = sum(accuracies) / num_iterations
    # avg_accuracy1 = sum(accuracies) / num_iterations
    avg_gt_find = sum(gt_finds) / num_iterations
    # avg_validity = sum(validities) / num_iterations

    # Print results
    print(f"Average Recall: {avg_recall * 100:.2f}%")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    # print(f"Average Accuracy1: {avg_accuracy1 * 100:.2f}%")
    print(f"Average Ground Truth Found: {avg_gt_find * 100:.2f}%")
    # print(f"Average Validity: {avg_validity * 100:.2f}%")
def run_rc_explainer_on_real_graph(num_iterations=None, dataset=None):
    total_nodes = 0
    total_edges = 0
    recalls, accuracies, gt_finds, validities = [], [], [], []
    for i in range(num_iterations):
        data, graph_data, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
        total_nodes += graph_data.number_of_nodes()
        total_edges += graph_data.number_of_edges()

        if has_one:  # Proceed only if there is a 1 in role_id
            model = GIN(1, 32, 2)
            model.eval()
            with torch.no_grad():
                predicted_classes = model(data.x, data.edge_index).argmax(dim=1)
            state = torch.rand(len(data.edge_index[0])) > 0.5
            explainer = RC_Explainer(model, 2, 32, True)

            _, edge_importances = explainer(data, state, predicted_classes)

            # Convert edge_importances to a floating-point tensor if it's not already
            if edge_importances.dtype != torch.float32 and edge_importances.dtype != torch.float64:
                edge_importances = edge_importances.float()

            # Define a threshold for edge importance
            importance_threshold = edge_importances.mean()

            # Initialize node_importance_labels tensor
            prediction = torch.zeros(graph_data.number_of_nodes(), dtype=torch.float)

            # Label nodes as important if they are part of an important edge
            for idx, edge in enumerate(data.edge_index.T):
                if idx < edge_importances.size(0) and edge_importances[idx] > importance_threshold:
                    node_u, node_v = edge[0].item(), edge[1].item()
                    prediction[node_u] = 1
                    prediction[node_v] = 1
            # print(role_id)
            print(prediction)
            relative_positives = (torch.tensor(role_id, dtype=torch.float) == 1).nonzero(as_tuple=True)[0]
            # rc_recall = torch.sum(node_importance_labels[relative_positives]).item() / len(relative_positives)
            # rc_acc = (torch.sum((node_importance_labels == 1) & (torch.tensor(role_id, dtype=torch.float) == 1)).item()) / (torch.sum(torch.tensor(role_id, dtype=torch.float) == 1).item()) * 100
            # rc_gt_find = int(set(node_importance_labels.nonzero().flatten().tolist()) == set(relative_positives.tolist()))
            rc_recall = torch.sum(prediction[relative_positives]).item() / torch.sum(torch.tensor(prediction) == 1)
            print(rc_recall)
            rc_acc = torch.sum(prediction[relative_positives]).item() / torch.sum(torch.tensor(role_id) == 1)
            print(rc_acc)
            prediction_new = [int(item) for item in prediction]
            # print("prediction_new", prediction_new)
            rc_gt_find = int(prediction_new == role_id)
            rc_validity = int(all(item in relative_positives.tolist() for item in prediction.nonzero().flatten().tolist()))
            print(rc_validity)
            recalls.append(rc_recall)
            accuracies.append(rc_acc)
            gt_finds.append(rc_gt_find)
            validities.append(rc_validity)

    # Calculate and print averages
    avg_recall = sum(recalls) / num_iterations
    avg_accuracy = sum(accuracies) / num_iterations
    avg_gt_find = sum(gt_finds) / num_iterations
    avg_validity = sum(validities) / num_iterations

    print(f"Average Recall: {avg_recall * 100:.2f}%")
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Average Ground Truth Found: {avg_gt_find * 100:.2f}%")
    print(f"Average Validity: {avg_validity * 100:.2f}%")

# run_experiments_syn(num_iterations=500, width_basis=5, m=2)

data_path_benzene = 'D:/University/Research1/GraphXAI/graphxai/datasets/real_world/benzene/benzene.npz'
dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# data_path_FluorideCarbonyl = 'D:/University/Research1/GraphXAI/graphxai/datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
# dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)
run_rc_explainer_on_real_graph(num_iterations=1000,dataset=dataset_Benzene)