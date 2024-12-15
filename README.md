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
  
However, the folder `DepMap-analysis/dataset` is stored in Google Drive because of the file size limitations of GitHub. Please download the folder via XXX. Thank you! 

## Usage
The input of 
