import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import degree
import numpy as np

sys.path.append('/home/ivanam/CBMS_bioWork/')

from load_data import get_hetero_data
from models.hetero_gnn import HeteroEncoder, EdgePredictor, to_hetero

CHG_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/ChG-Miner_miner-chem-gene.tsv.gz'
DCH_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('results_hetero', exist_ok=True)
log_file = open("results_hetero/hetero_training_log.txt", "w")
sys.stdout = log_file

def train_and_eval_hetero(data, model_type='GAT'):
    transform = RandomLinkSplit(
        num_val=0.1, 
        num_test=0.2,
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'rev_treats', 'drug')],
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)
    
    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    encoder = HeteroEncoder(hidden_channels=64, out_channels=32)
    model = to_hetero(encoder, data.metadata(), aggr='sum').to(DEVICE)
    predictor = EdgePredictor().to(DEVICE)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    best_val_auc = 0
    final_test_results = (0, 0) # AUC, AP

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        edge_label_index = train_data['drug', 'treats', 'disease'].edge_label_index
        edge_label = train_data['drug', 'treats', 'disease'].edge_label
        
        preds = predictor(z_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(preds, edge_label)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_z = model(val_data.x_dict, val_data.edge_index_dict)
                val_edge_index = val_data['drug', 'treats', 'disease'].edge_label_index
                val_labels = val_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
                
                val_preds = predictor(val_z, val_edge_index).sigmoid().cpu().numpy()
                val_auc = roc_auc_score(val_labels, val_preds)
                val_ap = average_precision_score(val_labels, val_preds)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    
                    test_z = model(test_data.x_dict, test_data.edge_index_dict)
                    test_edge_index = test_data['drug', 'treats', 'disease'].edge_label_index
                    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
                    
                    test_preds = predictor(test_z, test_edge_index).sigmoid().cpu().numpy()
                    test_auc = roc_auc_score(test_labels, test_preds)
                    test_ap = average_precision_score(test_labels, test_preds)
                    
                    final_test_results = (test_auc, test_ap)

                print(f" Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    evaluate_cold_start(test_data, test_preds, data)
    return final_test_results

def get_top_cold_predictions(test_data, test_preds, data_original, drug_map, disease_map, top_n=5):
    
    edge_index_disease = data_original['drug', 'treats', 'disease'].edge_index[0].cpu()
    drug_degree = degree(edge_index_disease, num_nodes=data_original['drug'].num_nodes)
    
    rev_drug_map = {v: k for k, v in drug_map.items()}
    rev_disease_map = {v: k for k, v in disease_map.items()}
    
    test_drug_indices = test_data['drug', 'treats', 'disease'].edge_label_index[0].cpu()
    test_disease_indices = test_data['drug', 'treats', 'disease'].edge_label_index[1].cpu()
    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu()
    
    cold_mask = (drug_degree[test_drug_indices] <= 2) & (test_labels == 1)
    
    cold_preds = test_preds[cold_mask]
    cold_drugs = test_drug_indices[cold_mask]
    cold_diseases = test_disease_indices[cold_mask]
    
    sorted_indices = np.argsort(cold_preds)[::-1]
    
    print("\n=== TOP 5 COLD-START SUCCESS STORIES ===")
    print(f"{'Drug ID':<12} | {'Disease ID':<12} | {'Confidence':<10}")
    print("-" * 40)
    
    for i in sorted_indices[:top_n]:
        d_id = rev_drug_map[int(cold_drugs[i])]
        dis_id = rev_disease_map[int(cold_diseases[i])]
        prob = cold_preds[i]
        print(f"{d_id:<12} | {dis_id:<12} | {prob:.4f}")


def evaluate_cold_start(test_data, test_preds, data_original):
    edge_index_disease = data_original['drug', 'treats', 'disease'].edge_index[0].cpu()
    num_drugs = data_original['drug'].num_nodes
    drug_degree = degree(edge_index_disease, num_nodes=num_drugs)
    
    test_drug_indices = test_data['drug', 'treats', 'disease'].edge_label_index[0].cpu()
    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
    
    cold_mask = (drug_degree[test_drug_indices] <= 2).numpy()
    
    print("\n" + "="*40)
    print("DETALJNA COLD START ANALIZA")
    print("="*40)
    
    if cold_mask.sum() > 1:
        cold_labels = test_labels[cold_mask]
        cold_preds = test_preds[cold_mask]
        
        if len(np.unique(cold_labels)) > 1:
            cold_auc = roc_auc_score(cold_labels, cold_preds)
            cold_ap = average_precision_score(cold_labels, cold_preds)
            
            print(f"Status: USPEŠNO IZRAČUNATO")
            print(f"Prag (Degree): <= 2 veze sa bolestima")
            print(f"Broj testiranih parova: {cold_mask.sum()}")
            print(f"Cold AUC: {cold_auc:.4f}")
            print(f"Cold AP:  {cold_ap:.4f}")
        else:
            print("Status: Nedovoljno varijabilnosti u klasama (sve su 0 ili sve 1).")
    else:
        print("Status: Nema dovoljno hladnih čvorova u test setu.")
    
    print("="*40 + "\n")

def run_experiment():
    print("=== STARTING HETEROGENEOUS EXPERIMENT ===")
    
    data = get_hetero_data(CHG_PATH, DCH_PATH)
    model_types = ['GAT'] 
    
    results = []
    for m_type in model_types:
        auc, ap = train_and_eval_hetero(data, m_type)
        results.append({
            'Model': f'Hetero-{m_type}',
            'Test_AUC': round(auc, 4),
            'Test_AP': round(ap, 4)
        })
    df = pd.DataFrame(results)
    df.to_csv('results_hetero/hetero_final_results.csv', index=False)
    
    print("\n--- FINAL HETEROGENEOUS RESULTS ---")
    print(df.to_string())
    print("\n=== EXPERIMENT FINISHED ===")

if __name__ == "__main__":
    try:
        run_experiment()
    finally:
        log_file.close()