from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import numpy as np

def load_data_flexible(path):
    print(f"Učitavam: {path}")
    # Automatski detektuje separator na osnovu ekstenzije
    sep = '\t' if path.endswith('.tsv.gz') else ','
    df = pd.read_csv(path, sep=sep, compression='gzip')
    
    col1, col2 = df.columns[0], df.columns[1]
    unique_nodes = pd.unique(df[[col1, col2]].values.ravel('K'))
    mapping = {node: i for i, node in enumerate(unique_nodes)}
    
    edge_index = torch.tensor([
        df[col1].map(mapping).values,
        df[col2].map(mapping).values
    ], dtype=torch.long)
    
    # Za neusmerene grafove dodajemo oba smera
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    data = Data(edge_index=edge_index, num_nodes=len(unique_nodes))
    # Dodajemo random features pošto BioSNAP nema atribute
    data.x = torch.randn((data.num_nodes, 128)) 
    return data

@torch.no_grad()
def evaluate_detailed(device, model, data, x, original_degrees, threshold=3):
    model.eval()
    z = model.encode(x.to(device), data.edge_index.to(device))
    out = model.decode(z, data.edge_label_index.to(device)).view(-1).sigmoid().cpu().numpy()
    labels = data.edge_label.cpu().numpy()
    
    # 1. AUC metrike
    overall_auc = roc_auc_score(labels, out)
    overall_ap = average_precision_score(labels, out) # DODATO
    
    # 2. Cold Start analitika
    edge_src = data.edge_label_index[0].cpu().numpy()
    edge_dst = data.edge_label_index[1].cpu().numpy()
    cold_mask = (original_degrees[edge_src] <= threshold) | (original_degrees[edge_dst] <= threshold)
    
    if cold_mask.sum() > 0:
        cold_auc = roc_auc_score(labels[cold_mask], out[cold_mask])
        cold_ap = average_precision_score(labels[cold_mask], out[cold_mask]) # DODATO
    else:
        cold_auc, cold_ap = 0.0, 0.0
        
    return overall_auc, cold_auc, overall_ap, cold_ap


def get_hetero_data(chg_path, dch_path):
    # 1. Učitavanje sirovih podataka
    df_chg = pd.read_csv(chg_path, sep='\t', compression='gzip')
    df_dch = pd.read_csv(dch_path, sep='\t', compression='gzip')

    # 2. Identifikacija jedinstvenih entiteta
    all_drugs = pd.unique(pd.concat([df_chg.iloc[:, 0], df_dch.iloc[:, 1]]))
    all_genes = df_chg.iloc[:, 1].unique()
    all_diseases = df_dch.iloc[:, 0].unique()

    # 3. Mapiranja
    drug_map = {node: i for i, node in enumerate(all_drugs)}
    gene_map = {node: i for i, node in enumerate(all_genes)}
    disease_map = {node: i for i, node in enumerate(all_diseases)}

    data = HeteroData()

    # 4. Dodavanje čvorova (Inicijalizacija)
    data['drug'].x = torch.randn(len(all_drugs), 128)
    data['gene'].x = torch.randn(len(all_genes), 128)
    data['disease'].x = torch.randn(len(all_diseases), 128)

    # 5. Dodavanje ivica (Drug - Target - Gene)
    # Koristimo np.array da izbegnemo UserWarning i ubrzamo proces
    edge_index_chg = torch.tensor(np.array([
        df_chg.iloc[:, 0].map(drug_map).values,
        df_chg.iloc[:, 1].map(gene_map).values
    ]), dtype=torch.long)
    data['drug', 'targets', 'gene'].edge_index = edge_index_chg

    # 6. Dodavanje ivica (Drug - Treats - Disease)
    edge_index_dch = torch.tensor(np.array([
        df_dch.iloc[:, 1].map(drug_map).values,
        df_dch.iloc[:, 0].map(disease_map).values
    ]), dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = edge_index_dch

    # Ključno za GNN: Dodajemo obrnute ivice da bi informacija tekla u oba smera
    # npr. gene -> drug -> disease
    data = T.ToUndirected()(data)

    return data


def get_full_hetero_data(chg_path, dch_path):
    # 1. Učitavanje sirovih podataka
    df_chg = pd.read_csv(chg_path, sep='\t', compression='gzip')
    df_dch = pd.read_csv(dch_path, sep='\t', compression='gzip')

    # 2. Identifikacija jedinstvenih entiteta
    all_drugs = pd.unique(pd.concat([df_chg.iloc[:, 0], df_dch.iloc[:, 1]]))
    all_genes = df_chg.iloc[:, 1].unique()
    all_diseases = df_dch.iloc[:, 0].unique()

    # 3. Mapiranja
    drug_map = {node: i for i, node in enumerate(all_drugs)}
    gene_map = {node: i for i, node in enumerate(all_genes)}
    disease_map = {node: i for i, node in enumerate(all_diseases)}

    data = HeteroData()

    # 4. Dodavanje čvorova (Inicijalizacija)
    data['drug'].x = torch.randn(len(all_drugs), 128)
    data['gene'].x = torch.randn(len(all_genes), 128)
    data['disease'].x = torch.randn(len(all_diseases), 128)

    # 5. Dodavanje ivica (Drug - Target - Gene)
    # Koristimo np.array da izbegnemo UserWarning i ubrzamo proces
    edge_index_chg = torch.tensor(np.array([
        df_chg.iloc[:, 0].map(drug_map).values,
        df_chg.iloc[:, 1].map(gene_map).values
    ]), dtype=torch.long)
    data['drug', 'targets', 'gene'].edge_index = edge_index_chg

    # 6. Dodavanje ivica (Drug - Treats - Disease)
    edge_index_dch = torch.tensor(np.array([
        df_dch.iloc[:, 1].map(drug_map).values,
        df_dch.iloc[:, 0].map(disease_map).values
    ]), dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = edge_index_dch

    # Ključno za GNN: Dodajemo obrnute ivice da bi informacija tekla u oba smera
    # npr. gene -> drug -> disease
    data = T.ToUndirected()(data)
    
    return data, drug_map, disease_map
