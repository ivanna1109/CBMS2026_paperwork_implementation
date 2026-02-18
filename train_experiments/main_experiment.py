import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree

# Putanja do tvojih modula
sys.path.append('/home/ivanam/CBMS_bioWork/')

from load_data import load_data_flexible, evaluate_detailed
from models.gnn import GNN_Model 

# --- KONFIGURACIJA ---
FILES = {
    'ChG-Miner': '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/ChG-Miner_miner-chem-gene.tsv.gz',
    'DCh-Miner': '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preusmeravanje svega u fajl
os.makedirs('results', exist_ok=True)
log_file = open("results/training_log.txt", "w")
sys.stdout = log_file # Od ovog trenutka svaki print ide u ovaj fajl

def train_and_eval(data, model_type):
    node_deg = degree(data.edge_index[0], data.num_nodes).numpy()
    
    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)
    
    model = GNN_Model(in_channels=128, hidden_channels=64, out_channels=32, model_type=model_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_data = train_data.to(DEVICE)
    best_val_auc = 0
    final_results = (0, 0, 0, 0) # AUC, Cold_AUC, AP, Cold_AP
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x.to(DEVICE), train_data.edge_index.to(DEVICE))
        out = model.decode(z, train_data.edge_label_index.to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label.to(DEVICE))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # Pratimo sve 4 metrike na validacionom skupu
            val_auc, val_cold_auc, val_ap, val_cold_ap = evaluate_detailed(DEVICE, model, val_data, val_data.x, node_deg)
            
            # Čuvamo najbolje rezultate na osnovu Overall AUC-a
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # Final_results su testni rezultati postignuti sa najboljim modelom
                final_results = evaluate_detailed(DEVICE, model, test_data, test_data.x, node_deg)
            
            # Ispis u log fajl (sys.stdout preusmeren na training_log.txt)
            print(f" Epoch {epoch:03d} | Loss: {loss:.4f}")
            print(f"   > Validacija: AUC={val_auc:.4f}, AP={val_ap:.4f}")
            print(f"   > Cold Start Validacija: AUC={val_cold_auc:.4f}, AP={val_cold_ap:.4f}")
            print("-" * 50)
            
    return final_results

def run_full_experiment():
    print("=== STARTING FULL EXPERIMENT ===")
    results = []
    
    for name, path in FILES.items():
        print(f"\n{'='*10} DATASET: {name} {'='*10}")
        data = load_data_flexible(path)
        
        for m_type in ['SAGE', 'GAT']:
            print(f"\nTraining model: {m_type}")
            auc, cold_auc, ap, cold_ap = train_and_eval(data, m_type)
            results.append({
                'Dataset': name,
                'Model': m_type,
                'AUC_Overall': round(auc, 4),
                'AUC_ColdStart': round(cold_auc, 4),
                'AP_Overall': round(ap, 4),
                'AP_ColdStart': round(cold_ap, 4)
            })
            
    df = pd.DataFrame(results)
    df.to_csv('results/enhanced_train_metrics.csv', index=False)
    
    print("\n--- FINAL RESULTS TABLE ---")
    print(df.to_string())
    print("\n=== EXPERIMENT FINISHED ===")

if __name__ == "__main__":
    try:
        run_full_experiment()
    finally:
        log_file.close() # Zatvaranje fajla na kraju