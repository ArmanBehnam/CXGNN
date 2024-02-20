# CXGNN
A GNN causal explainer based on a graph's causal structure and it's corresponding neural causal model. It outperforms the existing GNN explainers in exactly finding the ground-truth explanations.

## Motivations
> Graph classification via causal inference in the true way (based on interventional data)

> Understanding and quantifying cause-and-effect relations between variables

> Identify the causal explanatory subgraph

## Contributions
> The first GNN causal explainer

> We leverage the neural-causal connection, design the GNN neural causal models, and train them to identify the causal explanatory subgraph.

> Our results show the effectiveness of CXGNN and its superiority over the state-of-the-art association-based and causality-inspired GNN explainers

## Method 
Details of training GNN-NCMs are shown in Algorithm 1. This algorithm takes the causal structure G with respect to a reference node v as input and returns a well-trained GNN-NCM ![equation](https://latex.codecogs.com/gif.latex?%5Cwidehat%7B%5Cmathcal%7BM%7D%7D%28%5Cmathcal%7BG%7D%2C%20%5Ctheta%5E%2A%29). The underlying subgraph of the causal structure centered by v∗ is then treated as the causal explanatory subgraph Γ. Algorithm 2 describes the learning process to find Γ.

<p align="center">
  <img src="./images/Method.png" width="1000"> 
</p>


## Results
### Synthetic Graphs
We note that there are different ways for the ground truth subgraph to attach to the base synthetic graph. We can see CXGNN’s output exactly matches the ground truth in these cases, while the existing GNN explainers cannot. One reason could be that existing GNN explainers are sensitive to the spurious relation. 

<p align="center">
  <img src="./images/syn.png" width="1000"> 
<caption>Visualizing explanation results (subgraph containing the red nodes) by our CXGNN on the synthetic graphs.</caption>
</p>

### Real World Dataset
The [HELOC Dataset](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2) (Home Equity Line of Credit) is an anonymized dataset provided by FICO.
The fundamental task is to predict credit risk. A simple ANN is trained for this, reaching 70% validation accuracy. Causal input variables and their ranges are found using the pipeline above.
 
<p align="center">
  <img src="./images/real.png" width="500"> 
  <caption>Visualizing explanation results (subgraph containing the red nodes) by our CXGNN on the real-world graphs</caption>
  </p>
<br/><br/>
We observe the explanatory subgraphs found by CXGNN approximately/exactly match the ground truth. However, no existing GNN explainers can do so.

## Related Literature

1. Agarwal, Chirag, et al. "Evaluating explainability for graph neural networks." Scientific Data 10.1 (2023): 144.
2. Luo, Dongsheng, et al. "Parameterized explainer for graph neural network." Advances in neural information processing systems 33 (2020): 19620-19631.
3. Pearl, Judea, and Dana Mackenzie. The book of why: the new science of cause and effect. Basic books, 2018.
4. Geiger, Atticus, et al. "Inducing causal structure for interpretable neural networks." International Conference on Machine Learning. PMLR, 2022.
5. Shan, Caihua, et al. "Reinforcement learning enhanced explainer for graph neural networks." Advances in Neural Information Processing Systems 34 (2021): 22523-22533.
6. Lin, Wanyu, et al. "Orphicx: A causality-inspired latent variable model for interpreting graph neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
