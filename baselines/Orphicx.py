from imp_syn import *
from imp_real import *
from torch_geometric.nn import global_mean_pool
from torch import nn, optim
import torch.nn as nn
from torch.nn.modules.module import Module
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = self.linear(input)

        # Adjust the shapes for batched or non-batched data
        if input.dim() == 3 and adj.dim() == 3:
            # Both input and adj are batched
            output = torch.bmm(adj, support)
        elif input.dim() == 2 and adj.dim() == 2:
            # Neither input nor adj is batched
            output = torch.matmul(adj, support)
        else:
            raise ValueError("Inconsistent shape between input and adjacency matrix")

        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        # print("mu shape:", mu.shape, "logvar shape:", logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print("z shape:", z.shape)
        return self.dc(z), mu, logvar
class VGAE3(VGAE):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1_1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden2 = self.gc1_1(hidden1, adj)
        return self.gc2(hidden2, adj), self.gc3(hidden2, adj)
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        if z.dim() == 2:
            z = z.unsqueeze(0)
            # Now z should be 3D, and we can use torch.bmm
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        # If z was originally 2D, squeeze the output to maintain original dimensions
        if adj.size(0) == 1:
            adj = adj.squeeze(0)
        return adj
class VGAE3MLP(VGAE3):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, decoder_hidden_dim1, decoder_hidden_dim2, K, dropout):
        super(VGAE3MLP, self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout)
        self.dc = InnerProductDecoderMLP(output_dim, decoder_hidden_dim1, decoder_hidden_dim2, dropout, act=lambda x: x)
        self.dropout = nn.Dropout(p=dropout)
class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, act=torch.sigmoid):
        super(InnerProductDecoderMLP, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim1)  # Adjust input_dim here
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        # print("Shape of z in InnerProductDecoderMLP:", z.shape)
        z = F.relu(self.fc(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)
        if z.dim() == 2:
            z = z.unsqueeze(0)

        # Perform batch matrix multiplication
        adj = torch.bmm(z, z.transpose(1, 2))

        # Apply activation function
        adj = self.act(adj)

        # Squeeze out the batch dimension if it was originally 2D
        if adj.size(0) == 1 and z.dim() == 2:
            adj = adj.squeeze(0)

        return adj
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    bce = F.binary_cross_entropy_with_logits(preds_flat, labels_flat, pos_weight=pos_weight, reduction='mean')

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), -1))
    return norm * bce + KLD
"""
joint_uncond:
    Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
Inputs:
    - params['Nalpha'] monte-carlo samples per causal factor
    - params['Nbeta']  monte-carlo samples per noncausal factor
    - params['K']      number of causal factors
    - params['L']      number of noncausal factors
    - params['M']      number of classes (dimensionality of classifier output)
    - decoder
    - classifier
    - device
Outputs:
    - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
    - info['xhat']
    - info['yhat']
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def joint_uncond(params, decoder, classifier, adj, feat, act=torch.sigmoid, device=None):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)

    # Initialize lists to store results
    logits_list = []
    yhat_list = []
    # print(f"adj shape: {adj.shape}, feat shape: {feat.shape}")

    for i in range(params['Nalpha']):
        for j in range(params['Nbeta']):
            alpha = torch.randn((1, adj.shape[0], params['K']), device=device)
            beta = torch.randn((1, adj.shape[0], params['L']), device=device)
            zs = torch.cat([alpha, beta], dim=-1)
            xhat = act(decoder(zs)) * adj
    # Process each sample independently to save memory
    # for i in range(params['Nalpha']):
    #     for j in range(params['Nbeta']):
    #         # Generate random samples for alpha and beta
    #         alpha = torch.randn((1, adj.shape[0], params['K']), device=device)
    #         beta = torch.randn((1, adj.shape[0], params['L']), device=device)
    #
    #         # Concatenate alpha and beta
    #         zs = torch.cat([alpha, beta], dim=-1)
    #
    #         # Decode zs and apply to adjacency matrix
    #         xhat = act(decoder(zs)) * adj
            # xhat_pooled = global_mean_pool(xhat, torch.arange(adj.shape[0], device=device))
            # print(f"Iteration {i}, {j}: alpha shape: {alpha.shape}, beta shape: {beta.shape}")
            # print(f"zs shape: {zs.shape}, xhat shape: {xhat.shape}, xhat values: {xhat}")

            # Pass the expanded features and pooled xhat to the classifier
            logits = classifier(feat.unsqueeze(0), xhat)[0]
            # logits = classifier(feat.unsqueeze(0), xhat_pooled)
            logits_list.append(logits)

            # Calculate softmax probabilities
            yhat = F.softmax(logits, dim=-1)
            yhat_list.append(yhat)

    # Concatenate results from all samples
    logits_concat = torch.cat(logits_list, dim=0)
    yhat_concat = torch.cat(yhat_list, dim=0).view(params['Nalpha'], params['Nbeta'], -1)

    # Compute causal effect using concatenated results
    p = yhat_concat.mean(1)
    I = torch.sum(p * torch.log(p + eps), dim=1).mean()
    q = p.mean(0)
    I -= torch.sum(q * torch.log(q + eps))

    return -I, {'xhat': xhat, 'yhat': yhat_concat}
def beta_info_flow(params, decoder, classifier, adj, feat, node_idx=None, act=torch.sigmoid, mu=0, std=1, device=None):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)

    # Initialize alpha_mu, alpha_std, beta_mu, and beta_std
    if torch.is_tensor(mu):
        alpha_mu = mu[:, :params['K']]
        beta_mu = mu[:, params['K']:]
        alpha_std = std[:, :params['K']]
        beta_std = std[:, params['K']:]
    else:
        alpha_mu = 0
        beta_mu = 0
        alpha_std = 1
        beta_std = 1
    # Process each sample independently
    for i in range(params['Nalpha']):
        for j in range(params['Nbeta']):
            # Generate random samples for alpha and beta
            alpha = torch.randn((1, adj.shape[-1], params['K']), device=device).mul(alpha_std).add_(alpha_mu)
            beta = torch.randn((1, adj.shape[-1], params['L']), device=device).mul(beta_std).add_(beta_mu)
            zs = torch.cat([alpha, beta], dim=-1)

            # Decode zs and apply to adjacency matrix
            xhat = act(decoder(zs)) * adj
            xhat_pooled = global_mean_pool(xhat, torch.arange(adj.shape[0], device=device))
            # print(f"Iteration {i}, {j}: alpha shape: {alpha.shape}, beta shape: {beta.shape}")
            # print(f"zs shape: {zs.shape}, xhat shape: {xhat.shape}, xhat values: {xhat}")

            # Classify the pooled features
            if node_idx is None:
                logits = classifier(feat.unsqueeze(0), xhat_pooled)[0]
            else:
                logits = classifier(feat.unsqueeze(0), xhat_pooled)[0][:, node_idx, :]

            # Calculate softmax probabilities
            softmax_dim = -1 if logits.dim() == 1 else 1
            yhat = F.softmax(logits, dim=softmax_dim)
            p = yhat.mean(0)
            I += torch.sum(torch.mul(p, torch.log(p + eps)))
            q += p

    # Finalize causal effect calculation
    q /= (params['Nalpha'] * params['Nbeta'])
    I /= (params['Nalpha'] * params['Nbeta'])
    I -= torch.sum(torch.mul(q, torch.log(q + eps)))

    return -I, None
    # for i in range(0, params['Nalpha']):
    #     alpha = torch.randn((100, params['K']), device=device).mul(alpha_std).add_(alpha_mu).unsqueeze(0).repeat(params['Nbeta'], 1, 1)
    #     beta = torch.randn((params['Nbeta'], 100, params['L']), device=device).mul(beta_std).add_(beta_mu)
    #     zs = torch.cat([alpha, beta], dim=-1)
    #     # decode and classify batch of Nbeta samples with same alpha
    #     xhat = torch.sigmoid(decoder(zs)) * adj
    #     yhat = F.softmax(classifier(feat, xhat)[0], dim=1)
    #     p = 1. / float(params['Nbeta']) * torch.sum(yhat, 0)  # estimate of p(y|alpha)
    #     I = I + 1. / float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p + eps)))
    #     q = q + 1. / float(params['Nalpha']) * p  # accumulate estimate of p(y)
    # I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    # negCausalEffect = -I
    # info = {"xhat": xhat, "yhat": yhat}
    # return negCausalEffect, info
"""
joint_uncond_singledim:
    Sample-based estimate of "joint, unconditional" causal effect
    for single latent factor, -I(z_i; Yhat). Note the interpretation
    of params['Nalpha'] and params['Nbeta'] here: Nalpha is the number
    of samples of z_i, and Nbeta is the number of samples of the other
    latent factors.
Inputs:
    - params['Nalpha']
    - params['Nbeta']
    - params['K']
    - params['L']
    - params['M']
    - decoder
    - classifier
    - device
    - dim (i : compute -I(z_i; Yhat) **note: i is zero-indexed!**)
Outputs:
    - negCausalEffect (sample-based estimate of -I(z_i; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond_singledim(params, decoder, classifier, adj, feat, dim, node_idx=None, act=torch.sigmoid, mu=0, std=1,device=None):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = feat.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    adj = adj.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    if torch.is_tensor(mu):
        alpha_mu = mu
        beta_mu = mu[:, dim]

        alpha_std = std
        beta_std = std[:, dim]
    else:
        alpha_mu = 0
        beta_mu = 0
        alpha_std = 1
        beta_std = 1

    alpha = torch.randn((params['Nalpha'], adj.shape[-1]), device=device).mul(alpha_std).add_(alpha_mu).repeat(1,
                                                                                                               params[
                                                                                                                   'Nbeta']).view(
        params['Nalpha'] * params['Nbeta'], adj.shape[-1])
    zs = torch.randn((params['Nalpha'] * params['Nbeta'], adj.shape[-1], params['z_dim']), device=device).mul(
        beta_std).add_(beta_mu)
    zs[:, :, dim] = alpha
    xhat = act(decoder(zs)) * adj
    if node_idx is None:
        logits = classifier(feat, xhat)[0]
    else:
        logits = classifier(feat, xhat)[0][:, node_idx, :]
    yhat = F.softmax(logits, dim=1).view(params['Nalpha'], params['Nbeta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return -I, None
def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Fill in the adjacency matrix based on edge_index
    for edge in edge_index.t():
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1  # If the graph is undirected

    return adj_matrix
class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCNClassifier, self).__init__()
        # self.gc1 = GraphConvolution(input_dim, hidden_dim, dropout=0.5, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim, num_classes, dropout=0.5, act=lambda x: x)
        super(GCNClassifier, self).__init__()
        # More layers or hidden units can be added here
        self.gc1 = GraphConvolution(input_dim, hidden_dim, dropout=0.5, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, dropout=0.5, act=F.relu)  # Additional layer
        self.gc3 = GraphConvolution(hidden_dim, num_classes, dropout=0.5, act=lambda x: x)  # Output layer

    def forward(self, x, adj):
        # Same as before, but include the new layer in the forward pass
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.gc2(x, adj)  # Pass through the additional layer
        x = F.relu(x)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=-1)
    # def forward(self, x, adj):
    #     if x.dim() == 3:
    #         # Handling for batched input
    #         batch_size, num_nodes, num_features = x.size()
    #
    #         # Reshape x for batch processing
    #         x = x.reshape(batch_size * num_nodes, num_features)
    #
    #         # Adjust adj for batch processing to match x's shape
    #         adj = adj.repeat(batch_size, 1, 1)
    #         adj = adj.reshape(batch_size * num_nodes, batch_size * num_nodes)
    #     elif x.dim() == 2:
    #         # Assuming x is already in the shape [num_nodes, num_features]
    #         batch_size, num_features = 1, x.size(1)
    #         num_nodes = x.size(0)
    #
    #     # Ensure that the number of nodes in x matches the size of adj
    #     if x.size(0) != adj.size(0):
    #         raise ValueError("Inconsistent shape between input and adjacency matrix after reshaping")
    #
    #     x = self.gc1(x, adj)
    #     x = F.relu(x)
    #     x = self.gc2(x, adj)
    #
    #     # Reshape x back to 3D tensor if it was originally 3D
    #     if batch_size > 1:
    #         x = x.reshape(batch_size, num_nodes, -1)
    #
    #     return F.log_softmax(x, dim=-1)

def orphicx(G=None, role_id=None):
    num_shapes = 1
    data = from_networkx_to_torch(G, role_id)
    data.y = torch.tensor(role_id, dtype=torch.long)  # Add this line
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True)
    data.x = (data.x - mean) / (std + 1e-5)
    # print(data.x)
    # data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    # print("Label data (data.y):", data.y)

    # Initialize VGAE3MLP Model
    dropout = 0.5  # Feel free to adjust this value as needed
    input_feat_dim = 1  # Assuming each node has 1 feature
    hidden_dim1 = 32
    hidden_dim2 = 16
    decoder_hidden_dim1 = 16
    decoder_hidden_dim2 = 16
    K = 5 # Number of causal factors
    dropout = 0.5  # Dropout rate
    model = VGAE3MLP(input_feat_dim, hidden_dim1, hidden_dim2, 108, decoder_hidden_dim1, decoder_hidden_dim2, K, dropout)

    # Training configuration
    data = data.to(device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  # Adjust gamma to control the rate decrease
    early_stopping_patience = 20
    min_loss = float('inf')
    patience_counter = 0
    epochs = 100
    num_nodes = data.x.size(0)
    adj_matrix = edge_index_to_adjacency_matrix(data.edge_index, num_nodes)
    adj_matrix = adj_matrix.to(device)
    num_edges = data.edge_index.size(1) / 2  # Assuming undirected graph
    pos_weight = (num_nodes**2 - 2 * num_edges) / (2 * num_edges)
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)  # Convert pos_weight to a tensor
    norm = num_nodes**2 / (2 * num_edges)
    data.adj_label = adj_matrix
    classifier = GCNClassifier(input_dim=108, hidden_dim=64, num_classes=2).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classes = np.unique(data.y.cpu().numpy())
    class_weights = compute_class_weight('balanced', classes=classes, y=data.y.cpu().numpy())
    # class_weights = torch.tensor([0.1, 0.9], dtype=torch.float).to(device)  # Adjust based on class distribution
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # class_weights_tensor = torch.tensor([0.5, 2.0], dtype=torch.float).to(device)
    # print("Class weights:", class_weights_tensor)
    causal_params = {
        'Nalpha': 100,  # Number of alpha samples
        'Nbeta': 100,  # Number of beta samples
        'K': K,  # Number of causal factors
        'L': 100,  # Number of non-causal factors
        'M': 2,  # Assuming num_classes is defined and represents number of output classes
        'decoder': model,  # Your trained model
        'classifier': classifier,  # Assuming you have a classifier model if needed
        'device': device
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        recovered, mu, logvar = model(data.x, adj_matrix)
        # print("Mean of mu:", mu.mean().item())
        # print("Std of logvar:", logvar.exp().mean().sqrt().item())
        embeddings = mu  # or use the reparameterized z
        graph_pred = classifier(embeddings, adj_matrix)

        graph_labels = data.y  # Replace this with actual label loading if data.y is not correct
        vgae_loss = loss_function(recovered, data.adj_label, mu, logvar, num_nodes, norm, pos_weight_tensor)
        # classifier_loss = F.nll_loss(graph_pred, graph_labels, weight=weight.to(device))
        classifier_loss = F.nll_loss(graph_pred, graph_labels, weight=class_weights_tensor)
        total_loss = vgae_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        classifier_optimizer.step()
        scheduler.step()
        if total_loss < min_loss:
            min_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            # print(f"Stopping early at epoch {epoch}")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item()}")
            # print(f"Epoch {epoch}, VGAE Loss: {vgae_loss.item()}, Classifier Loss: {classifier_loss.item()}")

    model.eval()
    classifier.eval()
    with torch.no_grad():
        recovered, _, _ = model(data.x, adj_matrix)
        embeddings, _ = model.encode(data.x, adj_matrix)
        graph_pred = classifier(embeddings, adj_matrix)
        predicted_classes = torch.argmax(graph_pred, dim=1)
        # print(f"Classifier Predictions (logits): {graph_pred}")
        # print(f"Predicted Classes Distribution: {predicted_classes.bincount()}")
        # print(f"recovered shape: {recovered.shape}")
        # print(f"embeddings shape: {embeddings.shape}")
        # print(f"graph_pred shape: {graph_pred.shape}, predicted_classes values: {predicted_classes}")

    # # Update causal_params to use the trained classifier
    # print("Training completed and causal analysis done.")
    # joint_uncond_effect, _ = joint_uncond(causal_params, model.dc, classifier, adj_matrix, embeddings, device=device)
    # beta_info_flow_effect, _ = beta_info_flow(causal_params, model.dc, classifier, adj_matrix, embeddings, device=device)
    # print("Joint Unconditional Causal Effect:", joint_uncond_effect.item())
    # print("Beta Information Flow Effect:", beta_info_flow_effect.item())
    # # Assuming these are the parameters for joint_uncond and beta_info_flow methods
    # joint_uncond_effect, _ = joint_uncond(causal_params, model.dc, classifier, adj_matrix, embeddings, device=device)
    # beta_info_flow_effect, _ = beta_info_flow(causal_params, model.dc, classifier, adj_matrix, embeddings, device=device)
    #
    # print("Joint Unconditional Causal Effect:", joint_uncond_effect.item())
    # print("Beta Information Flow Effect:", beta_info_flow_effect.item())
    prediction = torch.argmax(classifier(embeddings, adj_matrix), dim=1)
    true_labels = torch.tensor(role_id, dtype=torch.long)
    print("orphicx finding", prediction)

    # Calculate node importance scores based on their contribution to the correct classification
    # For simplicity, let's use the absolute difference between predicted and true labels as a rudimentary importance score
    node_importance_scores = torch.abs(prediction - true_labels)
    node_imp = torch.tensor(role_id, dtype=torch.float)
    relative_positives = (node_imp == 1).nonzero(as_tuple=True)[0]
    # print("relative_positives", relative_positives)
    # Create tensors for metrics calculations
    # relative_positives = (torch.tensor(role_id, dtype=torch.float) == 1).nonzero(as_tuple=True)[0]
    relative_positives_tensor = torch.zeros_like(true_labels)
    relative_positives_tensor[relative_positives] = 1
    print("data is: ", relative_positives_tensor)
    predicted_tensor = torch.zeros_like(true_labels)
    predicted_tensor[prediction] = 1
    # print("orphicx finding", predicted_tensor)
    # my_recall = torch.sum(predicted_tensor[relative_positives]).item() / len(relative_positives)
    # my_acc = torch.sum((true_labels == 1) & (predicted_tensor == 1)).item() / torch.sum(true_labels == 1).item()
    # my_gt_find = int(set(predicted_classes.tolist()) == set(relative_positives))
    # my_validity = int(all(item in relative_positives for item in predicted_classes.tolist()))

    # rc_recall = torch.sum(node_importance_labels[relative_positives]).item() / len(relative_positives)
    # rc_acc = (torch.sum((node_importance_labels == 1) & (torch.tensor(role_id, dtype=torch.float) == 1)).item()) / (torch.sum(torch.tensor(role_id, dtype=torch.float) == 1).item()) * 100
    # rc_gt_find = int(set(node_importance_labels.nonzero().flatten().tolist()) == set(relative_positives.tolist()))
    # my_recall = torch.sum(predicted_classes[relative_positives]).item() / torch.sum(torch.tensor(predicted_classes) == 1)
    if torch.sum(prediction == 1).item() > 0:
        my_recall = torch.sum(prediction[relative_positives]).item() / torch.sum(prediction == 1)
        print("recall is: ", my_recall)
    else:
        my_recall = 0
    my_acc = torch.sum(prediction[relative_positives]).item() / torch.sum(torch.tensor(role_id) == 1)
    print("acc is: ", my_acc)
    prediction_new = [int(item) for item in prediction]
    my_gt_find = int(prediction_new == role_id)
    my_validity = int(all(item in relative_positives.tolist() for item in prediction.nonzero().flatten().tolist()))
    # print(my_validity)
    # Print Metrics
    # print(f"My Explanation accuracy is: {my_acc * 100:.2f}%")
    # print(f"My Explanation recall is: {my_recall * 100:.2f}%")
    # print("My Explanation found the ground truth? ", my_gt_find)
    return my_recall, my_acc, my_gt_find, my_validity
def run_experiments_syn(num_iterations=None, width_basis=None, m=None):
    recalls, accuracies, gt_finds, validities = [], [], [], []
    for _ in range(num_iterations):
        G, role_id, _ = build_graph(width_basis, "ba", "grid", 0, m=3)
        G = perturb([G], 0.01)[0]
        recall, accuracy, gt_find, validity = orphicx(G, role_id)
        recalls.append(recall)
        accuracies.append(accuracy)
        gt_finds.append(gt_find)
        validities.append(validity)

    # Calculate averages
    avg_recall = sum(recalls) / num_iterations
    avg_accuracy = sum(accuracies) / num_iterations
    avg_gt_find = sum(gt_finds) / num_iterations
    avg_validity = sum(validities) / num_iterations

    # Print results
    print(f"Average Recall: {avg_recall*100:.2f}")
    print(f"Average Accuracy: {avg_accuracy*100:.2f}")
    print(f"Average Ground Truth Found: {avg_gt_find*100:.2f}")
    print(f"Average Validity: {avg_validity*100:.2f}")
run_experiments_syn(num_iterations=500, width_basis=5, m=3)
def run_orphicx_explainer_on_real_graph(num_iterations=None,dataset=None):
    total_nodes = 0
    total_edges: int = 0
    recalls, accuracies, gt_finds, validities = [], [], [], []
    for i in range(num_iterations):
        data, graph_data, role_id, node_imp, gt_exp, gt_agg, has_one = preprocess(dataset, i)
        total_nodes += graph_data.number_of_nodes()
        total_edges += graph_data.number_of_edges()
        if has_one:  # Proceed only if there is a 1 in role_id
            recall, accuracy, gt_find, validity = orphicx(graph_data, role_id)
            recalls.append(recall)
            accuracies.append(accuracy)
            gt_finds.append(gt_find)
            validities.append(validity)
    # Calculate averages
    avg_recall = sum(recalls) / num_iterations
    avg_accuracy = sum(accuracies) / num_iterations
    avg_gt_find = sum(gt_finds) / num_iterations
    avg_validity = sum(validities) / num_iterations
    # Print results
    print(f"Average Recall: {avg_recall * 100:.2f}")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}")
    print(f"Average Ground Truth Found: {avg_gt_find * 100:.2f}")
    print(f"Average Validity: {avg_validity * 100:.2f}")

# data_path_benzene = 'D:/University/Research1/GraphXAI/graphxai/datasets/real_world/benzene/benzene.npz'
# dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# data_path_FluorideCarbonyl = 'D:/University/Research1/GraphXAI/graphxai/datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
# dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)
# run_orphicx_explainer_on_real_graph(num_iterations=1000,dataset=dataset_FluorideCarbonyl)
