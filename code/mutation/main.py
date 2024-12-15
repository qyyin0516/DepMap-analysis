import numpy as np
import pandas as pd
import math
import random
import argparse
from pathway_hierarchy import *
from neural_network import *
from utils import *
from scipy.stats import wasserstein_distance

random.seed(2)
np.random.seed(2)


parser = argparse.ArgumentParser()
parser.add_argument('--fake_SNP_file_name', type=str)
parser.add_argument('--pathway_file_name', type=str)
parser.add_argument('--encoded_dim', type=int, default=1024)
parser.add_argument('--n_hidden', type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--alpha_binomial", type=float, default=0.001)
parser.add_argument("--alpha_regularization", type=float, default=0.1)
parser.add_argument("--regularization_type", type=str, default='L2')
parser.add_argument("--epsilon", type=float, default=0.01)


def main():
    args = parser.parse_args()
    
    data = pd.read_csv("../../dataset/CRISPRGeneEffect.csv", index_col=0)
    expr = pd.read_csv("../../dataset/OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    
    gene_col = list(data.columns)
    for i in range(len(gene_col)):
        gene_col[i] = gene_col[i].split(' ')[0]
    data.columns = gene_col
    gene_info = pd.read_csv("../../dataset/InputGene/ScreenedGene.csv")
    gene_info = gene_info.drop_duplicates(subset=['From'], keep='first')
    data = data[gene_info.iloc[:, 0]]
    data.columns = gene_info.iloc[:, 1]
    ifnull = data.isnull().sum()
    data = data[ifnull[ifnull == 0].index]
    
    gene_col = list(expr.columns)
    for i in range(len(gene_col)):
        gene_col[i] = gene_col[i].split(' ')[0]
    expr.columns = gene_col
    gene_info = pd.read_csv("../../dataset/InputGene/ExpressionGene.csv")
    gene_info = gene_info.drop_duplicates(subset=['From'], keep='first')
    expr = expr[gene_info.iloc[:, 0]]
    expr.columns = gene_info.iloc[:, 1]
    ifnull = expr.isnull().sum()
    expr = expr[ifnull[ifnull == 0].index]
    
    data = data.loc[:, ~data.columns.duplicated()]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    idx = list(set(data.index) & set(expr.index))
    idx.sort()
    data = data.loc[idx, :]
    expr = expr.loc[idx, :]
    
    data = (data - data.mean()) / (data.max() - data.min())
    expr = (expr - expr.mean()) / (expr.max() - expr.min())
    
    dt = pd.concat([data, expr], axis=1)
    dt = dt.dropna(axis=1)
    n_dep = data.shape[1]
    n_expr = expr.shape[1]
    
    mutation = pd.read_csv("../../dataset/OmicsSomaticMutations.csv")
    mutation = mutation[mutation['VariantType'] == 'SNP']
    mutation = mutation[mutation['LikelyDriver']]
    mutation = mutation[['Chrom', 'Pos', 'HugoSymbol', 'ModelID']]
    
    clinvar = pd.read_csv("../../dataset/ClinVar/ClinVar_variant_summary.txt", delimiter='\t')
    clinvar = clinvar[clinvar['Assembly'] == "GRCh38"]
    clinvar = clinvar[clinvar['Type'] == "single nucleotide variant"]
    pathogenicity = pd.read_csv("../../dataset/ClinVar/pathogenicity.csv", index_col=0)
    pathogenetic_type = list(pathogenicity[pathogenicity['Pathogenicity'] == 'Y']['Category'])
    mutation_patho = clinvar[clinvar['ClinicalSignificance'].isin(pathogenetic_type)]
    mutation_patho = mutation_patho[["Chromosome", "Start", "GeneSymbol"]]
    
    mutation_patho['Chromosome'] = mutation_patho['Chromosome'].apply(lambda x: 'chr' + str(x))
    mutation['ID'] = mutation['Chrom'] + '-' + mutation['Pos'].astype(str)
    mutation_patho['ID'] = mutation_patho['Chromosome'] + '-' + mutation_patho['Start'].astype(str)
    mutation = mutation.sort_values(by=['ID'])
    mutation_patho = mutation_patho.sort_values(by=['ID'])
    
    mutation = pd.merge(mutation, mutation_patho, on='ID', how='inner')
    
    mutation = mutation.sort_values(["Chrom", "Pos"])
    mutation = mutation[["Chrom", "Pos", 'ModelID']]
    mutation = mutation.drop_duplicates()
    mutation.index = range(mutation.shape[0])
    mutation['SNP'] = mutation['Chrom'] + '-' + mutation['Pos'].map(str)
    model_SNP = pd.DataFrame(data=0, index=mutation['SNP'].unique(), columns=data.index)
    for i in range(mutation.shape[0]):
        if mutation["ModelID"][i] in model_SNP.columns:
            model_SNP.loc[mutation["SNP"][i]][mutation["ModelID"][i]] = 1
    
    pathway_genes = get_gene_pathways("../../dataset/reactome/Ensembl2Reactome_All_Levels.txt", species='human')
    pathway_names = '../../dataset/reactome/ReactomePathways.txt'
    relations_file_name = '../../dataset/reactome/ReactomePathwaysRelation.txt'
    masking, layers_node, dt = get_masking(pathway_names, pathway_genes, relations_file_name, dt.T, args.encoded_dim, n_dep, n_expr, n_hidden=args.n_hidden, species='human')
    dt_np = np.array(dt.T)
    layers_node[len(layers_node) - 1] = list(dt.index)
    
    batch_size = dt_np.shape[0]
    encoder_layer_sizes = []
    decoder_layer_sizes = [] 
    for i in range(len(masking)-1, -1, -1):
        encoder_layer_sizes.append(masking[i].shape[1])
    encoder_layer_sizes.append(masking[0].shape[0])
    encoder_layer_sizes.append(args.encoded_dim)
    decoder_layer_sizes.append(args.encoded_dim)
    decoder_layer_sizes.append(masking[0].shape[0])
    for i in range(len(masking)):
        decoder_layer_sizes.append(masking[i].shape[1])
    
    model_SNP = model_SNP[model_SNP.sum(axis=1) > 0]
    estimated_p = estimate_p(model_SNP.sum(axis=1), model_SNP.shape[1], initial_guess=0.8)
    
    autoencoder = train_autoencoder(dt_np, masking, encoder_layer_sizes, decoder_layer_sizes, model_SNP.shape[1], estimated_p, batch_size=batch_size, reg_type=args.regularization_type, alpha_reg=args.alpha_regularization, alpha_binomial=args.alpha_binomial, num_epochs=args.num_epochs, learning_rate=args.learning_rate, print_loss=False)
    fake_snp = np.array(autoencoder.encoder(dt_np))
    fake_snp = np.where(fake_snp < 0.5, 0, 1)  # This step is the reason why we obtain some "all-0".
    fake_snp = pd.DataFrame(fake_snp)
    fake_snp.to_csv(args.fake_SNP_file_name)
    
    R = calculate_relevance(dt_np, autoencoder, epsilon=args.epsilon)
    for i in range(len(layers_node)):
        R[i] = pd.DataFrame(R[i], index=dt.columns, columns=layers_node[len(layers_node) - 1 - i])
    for i in range(1, len(layers_node)):
        R[i + len(layers_node) - 1] = pd.DataFrame(R[i + len(layers_node) - 1], index=dt.columns, columns=layers_node[i])
    R_layer_mean = []
    for i in range(args.n_hidden * 2 + 3):
        R_layer_mean.append(R[i].mean(axis=0))
    R_all_layer = pd.DataFrame()
    num_pathway = []
    for i in range(args.n_hidden + 2, args.n_hidden * 2 + 2):
        R_all_layer = pd.concat([R_all_layer, R_layer_mean[i]])
        num_pathway += [len(R_layer_mean[i])] * len(R_layer_mean[i]) 
    result = pd.DataFrame({'layer': num_pathway, 'R': R_all_layer[0]})
    result = result.sort_values(by='R', ascending=False)
    result.to_csv(args.pathway_file_name)
    
if __name__ == '__main__':
    main()
