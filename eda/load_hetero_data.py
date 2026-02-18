import pandas as pd
import torch
from torch_geometric.data import HeteroData

def load_hetero_biosnap(chg_path, dch_path):
    df_chg = pd.read_csv(chg_path, sep='\t', compression='gzip')
    df_dch = pd.read_csv(dch_path, sep='\t', compression='gzip')
    all_drugs = pd.unique(pd.concat([df_chg.iloc[:, 0], df_dch.iloc[:, 1]]))
    all_genes = df_chg.iloc[:, 1].unique()
    all_diseases = df_dch.iloc[:, 0].unique()

    drug_map = {node: i for i, node in enumerate(all_drugs)}
    gene_map = {node: i for i, node in enumerate(all_genes)}
    disease_map = {node: i for i, node in enumerate(all_diseases)}

    data = HeteroData()

    data['drug'].x = torch.randn(len(all_drugs), 128)
    data['gene'].x = torch.randn(len(all_genes), 128)
    data['disease'].x = torch.randn(len(all_diseases), 128)

    edge_index_chg = torch.tensor([
        df_chg.iloc[:, 0].map(drug_map).values,
        df_chg.iloc[:, 1].map(gene_map).values
    ], dtype=torch.long)
    data['drug', 'targets', 'gene'].edge_index = edge_index_chg

    edge_index_dch = torch.tensor([
        df_dch.iloc[:, 1].map(drug_map).values, # Drug is in the second column u DCh dataset
        df_dch.iloc[:, 0].map(disease_map).values
    ], dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = edge_index_dch

    return data