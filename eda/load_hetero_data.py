import pandas as pd
import torch
from torch_geometric.data import HeteroData

def load_hetero_biosnap(chg_path, dch_path):
    # 1. Učitavanje sirovih podataka
    df_chg = pd.read_csv(chg_path, sep='\t', compression='gzip')
    df_dch = pd.read_csv(dch_path, sep='\t', compression='gzip')

    # 2. Identifikacija jedinstvenih entiteta
    # Uzimamo sve lekove iz oba fajla da bismo imali jedinstven set 'Drug' čvorova
    all_drugs = pd.unique(pd.concat([df_chg.iloc[:, 0], df_dch.iloc[:, 1]]))
    all_genes = df_chg.iloc[:, 1].unique()
    all_diseases = df_dch.iloc[:, 0].unique()

    # 3. Mapiranja (svaki tip čvora ima svoj mapping od 0)
    drug_map = {node: i for i, node in enumerate(all_drugs)}
    gene_map = {node: i for i, node in enumerate(all_genes)}
    disease_map = {node: i for i, node in enumerate(all_diseases)}

    data = HeteroData()

    # 4. Dodavanje čvorova (koristimo random features kao i pre)
    data['drug'].x = torch.randn(len(all_drugs), 128)
    data['gene'].x = torch.randn(len(all_genes), 128)
    data['disease'].x = torch.randn(len(all_diseases), 128)

    # 5. Dodavanje ivica (Drug - Target - Gene)
    edge_index_chg = torch.tensor([
        df_chg.iloc[:, 0].map(drug_map).values,
        df_chg.iloc[:, 1].map(gene_map).values
    ], dtype=torch.long)
    data['drug', 'targets', 'gene'].edge_index = edge_index_chg

    # 6. Dodavanje ivica (Drug - Treats - Disease)
    edge_index_dch = torch.tensor([
        df_dch.iloc[:, 1].map(drug_map).values, # Lek je u drugoj koloni u DCh
        df_dch.iloc[:, 0].map(disease_map).values
    ], dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = edge_index_dch

    return data