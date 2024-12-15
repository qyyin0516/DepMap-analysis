import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathway_hierarchy import *
from neural_network import *
from utils import *

random.seed(1999)
np.random.seed(1999)


parser = argparse.ArgumentParser()
parser.add_argument('--input_gene', type=str)
parser.add_argument('--output_performance', type=str)
parser.add_argument('--if_functional', type=bool, default=True)
parser.add_argument('--n_hidden', type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.0001)
parser.add_argument("--AUC_cutoff", type=float, default=0.6)
parser.add_argument("--Dp_cutoff", type=float, default=0.1)


def main():
    args = parser.parse_args()
    
    data = pd.read_csv("../../dataset/CRISPRGeneEffect.csv", index_col=0)
    mutation = pd.read_csv("../../dataset/OmicsSomaticMutations.csv")
    mutation = mutation[mutation['VariantType'] == 'SNP']
    mutation = mutation[['Chrom', 'Pos', 'HugoSymbol', 'ModelID']]
    
    if args.if_functional:
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
    mutation = mutation[['ModelID', 'HugoSymbol']]
    mutation = mutation.sort_values(["HugoSymbol", "ModelID"])
    mutation = mutation.drop_duplicates()
    mutation.index = range(mutation.shape[0])
    
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
    
    mut_gene = pd.read_csv(args.input_gene)
    label = pd.DataFrame(data=0, index=mut_gene.iloc[:, 0], columns=data.index)
    for i in range(label.shape[0]):
        mut_sub = mutation[mutation['HugoSymbol'] == mut_gene.iloc[i, 0]]
        model_sub = list(set(mut_sub['ModelID']) & set(label.columns))
        label.iloc[i][model_sub] = 1
        
    result = pd.DataFrame(columns=['auc', 'acc'])
    for i in range(label.shape[0]):
        x_train, x_test, y_train, y_test = train_test_split(data, label.iloc[i, :], test_size=0.25)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        pathway_genes = get_gene_pathways("../../dataset/reactome/Ensembl2Reactome_All_Levels.txt", species='human')
        pathway_names = '../../dataset/reactome/ReactomePathways.txt'
        relations_file_name = '../../dataset/reactome/ReactomePathwaysRelation.txt'
        root_name = [0, 1]
        masking, layers_node, gene_out = get_masking(pathway_names, pathway_genes, relations_file_name, x_train.T.index.tolist(), 
                                                    root_name, n_hidden=args.n_hidden)
        x_train = x_train.T.loc[gene_out, :]
        x_test = x_test.T.loc[gene_out, :]
    
        if y_train.iloc[:, 0].sum() > 0:
            dt = x_train
            dt.loc['label'] = y_train.iloc[:, 0]
            dt = dt.T
            dt0 = dt[dt['label'] == 0]
            dt1 = dt[dt['label'] == 1]
            index = np.random.randint(len(dt1), size=int(len(dt) - len(dt1)))
            up_dt1 = dt1.iloc[list(index)]
            up_dt = pd.concat([up_dt1, dt0])
            y_train = pd.DataFrame(up_dt['label'])
            x_train = up_dt
            del x_train['label']
            x_train = x_train.T
        
        y_train_pred_df = pd.DataFrame(data=0, index=x_train.columns, columns=list(range(2, len(masking) + 2)))
        y_test_pred_df = pd.DataFrame(data=0, index=x_test.columns, columns=list(range(2, len(masking) + 2)))
        activation_output = {}
        for output_layer in range(2, len(masking) + 2):
            print("Current neural network has " + str(output_layer - 1) + " hidden layers.")
            output_train, output_test = model(np.array(x_train),
                                              one_hot_coding(y_train),
                                              np.array(x_test),
                                              layers_node,
                                              masking,
                                              output_layer,
                                              learning_rate=args.learning_rate,
                                              minibatch_size=args.batch_size,
                                              num_epochs=args.num_epochs,
                                              gamma=args.alpha,
                                              print_cost=False)
            for j in range(len(output_train)):
                if (j != output_layer - 1):
                    output_train[j + 1] = pd.DataFrame(data=output_train[j + 1],
                                                       index=layers_node[len(layers_node) - 2 - j],
                                                       columns=x_train.columns)
                else:
                    output_train[j + 1] = pd.DataFrame(data=output_train[j + 1], index=[0, 1],
                                                       columns=x_train.columns)
            activation_output[output_layer] = output_train
            y_train_pred, y_test_pred = get_predictions(output_train, output_test, output_layer)
            y_train_pred_df.loc[:, output_layer] = pd.DataFrame(y_train_pred,
                                                                index=x_train.columns,
                                                                columns=[output_layer])
            y_test_pred_df.loc[:, output_layer] = pd.DataFrame(y_test_pred,
                                                               index=x_test.columns,
                                                               columns=[output_layer])
        y_train_pred_final = y_train_pred_df.T.mode().T.loc[x_train.columns, :][0]
        y_test_pred_final = y_test_pred_df.T.mode().T.loc[x_test.columns, :][0]
        result.loc[label.index[i]] = [manual_auc(y_test, y_test_pred_final),
                                      accuracy_score(y_test, y_test_pred_final)]
        result.to_csv(args.output_performance)
        if result.iloc[i, 0] >= args.AUC_cutoff:
           pathways = get_pathway_importance(y_train, activation_output, thr=args.Dp_cutoff)
           pathways.to_csv("pathways_" + label.index[i] + ".csv")


if __name__ == '__main__':
    main()
