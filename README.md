# Multitask Learning on Graph Neural Networks: Learning Multiple Graph Centrality Measures with a Unified Network

Code for the Paper "Multitask Learning on Graph Neural Networks: Learning Multiple Graph Centrality Measures with a Unified Network" published at ICANN2019 (The arxiv version is outdated)

The repository is organised in different folders for different models/configurations.

The first level contains the code to replicate the experiments, with the other subfolders containing experiment logs, saved models and other artifacts such as plots.

To replicate a model:
Extract the zipped dataset files in a folder, copy and paste the first-level models onto that folder and run the train.sh file, which will train a model for each centrality and a multitask one with all 4 centralities.

The models are:
* The RN model is contained in the folder "centrality-compare".
* The AM model is contained in the folder "centrality-normalised-on-model-rank".
* The AN model is contained in the folder "centrality-normalised-rank".

If you have any issues running the code or need any help replicating the results feel free to contact me.

If you use this or the GNN builder in your research please cite our ICANN paper.
