# BLTDMDGL
Bayesian Logistic Tensor Decomposition incorporating Hybrid Graph Deep Learning for Predicting Higher-Order Drug-Microbe-Disease Associations

Exploring potential associations among drugs, microbes, and diseases provides valuable insights into the mechanisms underlying human health conditions. This understanding is vital for advancing disease prevention and drug development. Currently, several computational models focus on predicting higher-order relationships among diverse biological entities. However, existing deep learning methods often require negative sampling and struggle to fully encompass the entire sample space. Furthermore, many tensor decomposition techniques involve numerous hyperparameters, making them difficult to adapt to complex data structures and effectively extract nonlinear relationships. To address these limitations, we propose a Bayesian Logistic Tensor Decomposition model incorporating Hybrid Graph Deep Learning, referred to as BLTDMDGL, which aims to identify potential drug-microbe-disease triplet associations. Firstly, leveraging the intrinsic characteristics of multi-source data, this model uses weighted graphs to represent single-type biological entities and hypergraphs to capture higher-order relationships among drugs, microbes, and diseases. Secondly, to enhance the applicability and nonlinear learning capabilities of tensor decomposition, it employs Hybrid Graph Deep Learning to generate prior expectations of latent variables while establishing an attention mechanism to facilitate the integration of multi-source features. Finally, this method combines Logistic Tensor Decomposition and Hybrid Graph Deep Learning within a Bayesian framework and employs a Variational Expectation-Maximization algorithm to enable adaptive inference of model parameters and latent variables. Predictive performance evaluated across four experimental scenarios on two benchmark datasets shows that BLTDMDGL outperforms other state-of-the-art methods, achieving superior AUPR, AUC, and F1 scores for triplet predictions in both balanced and severely imbalanced datasets. Additionally, it exhibits higher hit rates in predictive tasks involving novel drugs, microbes, and diseases. Case studies further substantiate that BLTDMDGL effectively predicts potential associations between drugs, microbes, and diseases.

#The workflow of our proposed BLTDMDGL model

![image](https://github.com/Mayingjun20179/BLTDMDGL/blob/main/workflow.png)

#Environment Requirement

tensorly==0.8.1

torch==2.4.1+cu121

pandas==2.0.3

deepchem==2.8.0

rdkit==2022.9.4

networkx==2.8.8

torch-geometric==2.6.1

torch_scatter==2.1.2+pt24cu121

#Documentation

DATA1: Experimental data for baseline data Data1

DATA2: Experimental data for baseline data Data2

result1: After running the program, the location where the experimental result of the benchmark data Data1 is stored.

result2: After running the program, the location where the experimental result of the benchmark data Data2 is stored.

#Usage

First, install all the packages required by “requirements.txt”.

Second, run the program “Main_BLTDMDGL_CV.py” to get all the prediction results of BLTDMDGL for the two benchmark datasets in the scenarios of CV_triplet, CV_drug, CV_micro and CV_dis.
