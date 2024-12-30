# VBMGDL
Variational Bayesian Inference with Hybrid Graph Deep Learning for Drug-Microbe-Disease Association Prediction

The exploration of potential drug-microbe-disease associations allows us to understand the underlying mechanisms of human diseases from multiple perspectives, which is of great importance in promoting disease prevention and drug development. Currently, there are several computational models that focus on predicting higher-order associations of multivariate biological entities. However, existing deep learning methods often require negative sampling and are difficult to cover the entire sample space, while most tensor decomposition methods contain numerous hyperparameters and are difficult to adapt to complex data structures and explore nonlinear relationships. To this end, we propose a variational Bayesian inference model with hybrid graph deep learning, VBMGDL, to identify potential drug-microbe-disease triple associations. To enhance the applicability and nonlinear learning capability of the Bayesian model, we introduce a hybrid graph deep learning model to generate prior expectations of latent variables and establish an attention mechanism to achieve adaptive fusion of multi-source features. Meanwhile, to ensure solution efficiency, we develop a variational expectation maximization algorithm to achieve model inference. Experimental results under four experimental scenarios on two benchmark datasets show that, compared with other state-of-the-art methods, the triple prediction by VBMGDL exhibits excellent AUPR, AUC, and F1 values for both balanced and extremely unbalanced datasets, and higher hit rates on the prediction tasks of new drugs, new microbes, and new diseases. In addition, case studies further demonstrated that VBMGDL can effectively predict potential drug-microbe-disease associations.

#The workflow of our proposed VBMGDL model

![image](https://github.com/user-attachments/assets/22354f52-5652-4b98-80c0-e88529465d33)

#Environment Requirement

tensorly==0.8.1

torch==2.4.1 ((GPU version))

pandas==2.0.3

deepchem==2.8.0

rdkit==2022.9.4

networkx==2.8.8

torch-geometric==2.6.1

torch_scatter==2.1.2+pt24cu121

#Usage

First, install all the packages required by “requirements.txt”.

Second, run the program “Main_VBMGDL_CV.py” to get all the prediction results of VBMGDL for the two benchmark datasets in the scenarios of CV_triplet, CV_drug, CV_micro and CV_dis.
