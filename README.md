# Explainable deep learning for identifying cancer driver genes and generating driver-like artificial representative variants based on the Cancer Dependency Map

In this project, we utilized the Cancer Dependency Map (DepMap) to identify potential cancer driver genes and generate artificial representative driver variants. Using DepMap dependency score data, we developed a biologically informed, supervised deep learning model for each frequently mutated gene, leveraging functional SNPs to predict its mutation status. The fitness of the model represents how likely the gene is a cancer driver gene. To generate artificial representative variants (ARVs), we built an explainable autoencoder, guided by the distribution of real driver SNPs. This allowed us to identify important pathways for the dependence profile of a cell line. 

This project corresponds to the following paper: Yin, Q., Chen, L.. Explainable deep learning for identifying cancer driver genes and generating driver-like artificial representative variants based on the Cancer Dependency Map, in preparation.

## Dependencies
The models are built with Python 3 (>= 3.9.2) with the following packages:

* numpy >= 1.21.6
* pandas >= 1.5.3
* scipy >= 1.11.4
* keras >= 2.9.0
* tensorflow >= 2.9.0
* scikit-learn >= 1.0.2
* networkx >= 2.6.3

## Installation
Clone the github repository and enter DepMap-analysis directory with

    $ git clone https://github.com/qyyin0516/DepMap-analysis.git
    $ cd DepMap-analysis
  
However, the folder `DepMap-analysis/dataset` is stored in Google Drive because of the file size limitations of GitHub. Please download the folder via https://drive.google.com/drive/folders/1CWI-P40QcIpNmYxleX5y-6KvhfznYadg?usp=sharing. Thank you! 

## Usage
Executing `code/gene/main.py` evaluates the supervised models, providing fitness metrics such as AUC and accuracy, along with identifying important pathways. Users need to specify the input gene list and the output file name for the fitness results of all genes (AUC and accuracy). Similarly, executing `code/mutation/main.py` generates ARVs and calculates the relevance scores of pathways, averaged across all cell lines. Users should specify the output file name for the ARVs and pathways.

The list below is the options for `code/gene/main.py`.


    --input_gene                    path of input gene list (required)
    --output_performance            path of output fitness results (required)
    --if_functional                 if functional SNPs are used (optional, default: True)
    --n_hidden                      number of hidden layers of the neural network (optional, default: 3)
    --learning_rate                 learning rate (optional, default: 0.01)
    --batch_size                    batch size (optional, default: 128)
    --num_epochs                    epochs for training the neural network (optional, default: 100)
    --alpha                         L2 regularization parameter (optional, default: 0.0001)
    --AUC_cutoff                    threshold to measure the gene is a driver gene or not (optional, default: 0.6)
    --Dp_cutoff                     threshold to measure the pathway is important or not (optional, default: 0.1)

Here is an example.

    $ python code/gene/main.py  --input_gene "../../dataset/InputGene/3008Gene.csv"\
                                --output_performance "output_performance.csv"\

The list below is the options for `code/mutation/main.py`.

    --fake_SNP_file_name            path of output ARVs (required)
    --pathway_file_name             path of output relevance scores of pathways (required)
    --encoded_dim                   the number of ARVs or the dimensions of the latent layer (optional, default: 1024)
    --n_hidden                      number of hidden layers of the neural network (optional, default: 3)
    --learning_rate                 learning rate (optional, default: 0.01)
    --num_epochs                    epochs for training the neural network (optional, default: 1000)
    --alpha_binomial                penalty parameter for binomial distribution (optional, default: 0.001)
    --alpha_regularization          penalty parameter for regularization (optional, default: 0.1)
    --regularization_type           type of regularization (optional, default: 'L2')
    --epsilon                       parameter for LRP-epsilon (optional, default: 0.01)

Here is an example.

    $ python code/mutation/main.py  --fake_SNP_file_name "output_fake_SNP.csv"\
                                    --pathway_file_name "output_pathway.csv"\
