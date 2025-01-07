# VBMGDL
A Variational Bayesian Inference Approach with Hybrid Graph Deep Learning for Predicting Higher-Order Drug-Microbe-Disease Associations

Exploring the higher-order relationships of drugs, microbes, and diseases (DMD) allows us to understand the underlying mechanisms of human disease from multiple perspectives, which is of great importance in advancing disease prevention and drug development. Existing deep learning methods often require negative sampling and are difficult to cover the entire sample space, while most tensor decomposition methods contain numerous hyperparameters and are difficult to adapt to complex data structures and explore nonlinear relationships.  To this end, we propose a variational Bayesian inference model with hybrid graph deep learning, VBMGDL, to identify potential DMD triple associations. Several experimental results show that VBMGDL exhibits better prediction performance for the ternary prediction task in both balanced and extremely unbalanced datasets, and achieves higher hit rates for the prediction of new drugs, new microbes, and new diseases compared to other state-of-the-art methods. In addition, case studies further demonstrate that VBMGDL can be a powerful tool for DMD higher-order association prediction.

#The workflow of our proposed VBMGDL model

![image](https://github.com/Mayingjun20179/VBMGDL/blob/main/workflow.png)

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

Second, run the program “Main_VBMGDL_CV.py” to get all the prediction results of VBMGDL for the two benchmark datasets in the scenarios of CV_triplet, CV_drug, CV_micro and CV_dis.
