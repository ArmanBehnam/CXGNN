import alg1
def print_expected_p_for_each_node(models):
    for node, model_info in models.items():
        expected_p = model_info['expected_p']
        print(f"Node: {node}, Expected Probability: {expected_p}")
def alg_2(Graph,num_epochs,data,role_id):
    if num_epochs is None:
        num_epochs = 100  # Default value, change as necessary
    models = {}
    for node in Graph.v:
        Graph.target_node = node
        loss_history, total_loss, model, expected_p, output, new_v = alg1.train(Graph, 0.005, 32, 2, num_epochs, data, role_id, node)
        models[node] = {'model': model,
            'expected_p': expected_p,
            'total_loss': total_loss,
            'output': output,
            'new_v': new_v,
            'loss_history': loss_history}
    best_node = max(models.keys(), key=lambda k: models[k]['expected_p'])
    best_model = models[best_node]['model']
    best_total_loss = models[best_node]['total_loss']
    best_expected_p = models[best_node]['expected_p']
    best_output = models[best_node]['output']
    best_new_v = models[best_node]['new_v']
    print_expected_p_for_each_node(models)
    print(best_new_v, best_node)
    return models, best_total_loss, best_model, best_expected_p, best_output, best_new_v, best_node
