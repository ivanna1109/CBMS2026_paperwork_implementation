import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, average_precision_score

# Putanja do tvojih modula
sys.path.append('/home/ivanam/CBMS_bioWork/')

from load_data import get_full_hetero_data
from models.hetero_gnn import HeteroEncoder, HeteroSAGEEncoder, EdgePredictor, to_hetero
from utils.visualizations import plot_degree_performance, save_case_study_table

# --- KONFIGURACIJA ---
CHG_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/ChG-Miner_miner-chem-gene.tsv.gz'
DCH_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('results_final', exist_ok=True)
log_file = open("results_final/final_experiment_log.txt", "w")
sys.stdout = log_file

def analyze_performance_by_degree(test_data, test_preds, data_original, model_type):
    """Računa AUC za različite pragove stepena i vraća listu rezultata za grafik."""
    edge_index_disease = data_original['drug', 'treats', 'disease'].edge_index[0].cpu()
    drug_degree = degree(edge_index_disease, num_nodes=data_original['drug'].num_nodes)
    
    test_drug_indices = test_data['drug', 'treats', 'disease'].edge_label_index[0].cpu()
    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
    
    thresholds = [1, 2, 3, 5, 10]
    degree_results = []
    
    print(f"\nAnalysis by Degree for {model_type}:")
    for t in thresholds:
        mask = (drug_degree[test_drug_indices] <= t).numpy()
        if mask.sum() > 0 and len(np.unique(test_labels[mask])) > 1:
            auc = roc_auc_score(test_labels[mask], test_preds[mask])
            degree_results.append({'Model': model_type, 'Threshold': t, 'AUC': auc})
            print(f"  Degree <= {t}: AUC = {auc:.4f}")
    
    return degree_results

def get_top_cold_predictions(test_data, test_preds, data_original, drug_map, disease_map, top_n=10):
    """Identifikuje najbolje hladne predikcije i vraća ih kao listu rečnika."""
    edge_index_disease = data_original['drug', 'treats', 'disease'].edge_index[0].cpu()
    drug_degree = degree(edge_index_disease, num_nodes=data_original['drug'].num_nodes)
    
    rev_drug_map = {v: k for k, v in drug_map.items()}
    rev_disease_map = {v: k for k, v in disease_map.items()}
    
    test_drug_indices = test_data['drug', 'treats', 'disease'].edge_label_index[0].cpu()
    test_disease_indices = test_data['drug', 'treats', 'disease'].edge_label_index[1].cpu()
    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu()
    
    cold_mask = (drug_degree[test_drug_indices] <= 2) & (test_labels == 1)
    
    if cold_mask.sum() == 0:
        return []

    cold_preds = test_preds[cold_mask.numpy()]
    cold_drugs = test_drug_indices[cold_mask]
    cold_diseases = test_disease_indices[cold_mask]
    
    sorted_indices = np.argsort(cold_preds)[::-1]
    case_study_list = []
    
    for i in sorted_indices[:top_n]:
        d_id = rev_drug_map[int(cold_drugs[i])]
        dis_id = rev_disease_map[int(cold_diseases[i])]
        case_study_list.append({
            'Drug_ID': d_id, 
            'Disease_ID': dis_id, 
            'Confidence': float(cold_preds[i])
        })
    
    return case_study_list

def train_and_eval_hetero(data, drug_map, disease_map, model_type='GAT'):
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.2,
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'rev_treats', 'drug')],
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)
    train_data, val_data, test_data = train_data.to(DEVICE), val_data.to(DEVICE), test_data.to(DEVICE)

    if model_type == 'GAT':
        encoder = HeteroEncoder(hidden_channels=64, out_channels=32)
    else:
        encoder = HeteroSAGEEncoder(hidden_channels=64, out_channels=32)
        
    model = to_hetero(encoder, data.metadata(), aggr='sum').to(DEVICE)
    predictor = EdgePredictor().to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    best_val_auc = 0
    test_preds_final = None

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        preds = predictor(z_dict, train_data['drug', 'treats', 'disease'].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(preds, train_data['drug', 'treats', 'disease'].edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_z = model(val_data.x_dict, val_data.edge_index_dict)
                val_preds = predictor(val_z, val_data['drug', 'treats', 'disease'].edge_label_index).sigmoid().cpu().numpy()
                val_labels = val_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
                val_auc = roc_auc_score(val_labels, val_preds)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    test_z = model(test_data.x_dict, test_data.edge_index_dict)
                    test_preds_final = predictor(test_z, test_data['drug', 'treats', 'disease'].edge_label_index).sigmoid().cpu().numpy()
                
                print(f" {model_type} Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Prikupljanje podataka za eksterne analize
    deg_res = analyze_performance_by_degree(test_data, test_preds_final, data, model_type)
    case_study = get_top_cold_predictions(test_data, test_preds_final, data, drug_map, disease_map)
    
    test_labels = test_data['drug', 'treats', 'disease'].edge_label.cpu().numpy()
    final_auc = roc_auc_score(test_labels, test_preds_final)
    final_ap = average_precision_score(test_labels, test_preds_final)
    
    return final_auc, final_ap, deg_res, case_study

def run_experiment():
    print("=== STARTING FINAL RESEARCH EXPERIMENT WITH VISUALIZATION ===")
    data, drug_map, disease_map = get_full_hetero_data(CHG_PATH, DCH_PATH)
    
    all_degree_results = []
    summary_results = []
    
    for m_type in ['GAT', 'SAGE']:
        print(f"\n\n{'#'*40}\n# RUNNING MODEL: {m_type}\n{'#'*40}")
        auc, ap, deg_res, case_study = train_and_eval_hetero(data, drug_map, disease_map, m_type)
        
        summary_results.append({'Model': m_type, 'Test_AUC': round(auc, 4), 'Test_AP': round(ap, 4)})
        all_degree_results.extend(deg_res)
        
        # Čuvanje case study tabele za svaki model posebno
        save_case_study_table(case_study, m_type, 'results_final/case_study')

    # Kreiranje finalnog grafika
    plot_degree_performance(all_degree_results, save_path='results_final/degree_performance_plot.png')

    # Finalni ispis
    df = pd.DataFrame(summary_results)
    print("\n" + "="*30)
    print("FINAL SUMMARY")
    print("="*30)
    print(df.to_string(index=False))
    print("="*30)

if __name__ == "__main__":
    try:
        run_experiment()
    finally:
        log_file.close()