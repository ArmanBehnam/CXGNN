from graphxai.datasets import AlkaneCarbonyl, Benzene, FluorideCarbonyl
from imp_syn import ba_house, ba_grid, ba_cycle, tree_house, tree_grid, tree_cycle
from imp_real import benzene, fluoride_carbonyl
# from viz_real import vis_Mutagenicity, vis_NCI1, vis_ENZYMES, vis_AIDS, vis_benzene, vis_PROTEINS, vis_FluorideCarbonyl

# ba_house(basis_type="ba", Ground_truth="house", width_basis = 5, m = 3, num_epochs = 2, num_iterations = 500)
# ba_grid(basis_type="ba", Ground_truth="grid", width_basis = 5, m = 3, num_epochs = 50, num_iterations = 100)
# ba_cycle(basis_type="ba", Ground_truth="cycle", width_basis = 5, m = 3, num_epochs = 50, num_iterations = 500)
# tree_house(basis_type="tree", Ground_truth="house", width_basis = 3, m = 2, num_epochs = 10, num_iterations = 500)
# tree_grid(basis_type="tree", Ground_truth="grid", width_basis = 2, m = 2, num_epochs = 10, num_iterations = 2)
# tree_cycle(basis_type="tree", Ground_truth="cycle", width_basis = 3, m = 2, num_epochs = 50, num_iterations = 100)

# data_path_benzene = '/datasets/real_world/benzene/benzene.npz'
# data_path_FluorideCarbonyl = '/datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
# dataset_Benzene = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_benzene)
# dataset_FluorideCarbonyl = FluorideCarbonyl(split_sizes = (0.75, 0.05, 0.2), data_path = data_path_FluorideCarbonyl)

# benzene(dataset = dataset_Benzene, num_epochs = 1, num_iterations = 10)
# fluoride_carbonyl(dataset = dataset_FluorideCarbonyl, num_epochs = 10, num_iterations = 200)

# vis_FluorideCarbonyl(dataset_FluorideCarbonyl)
