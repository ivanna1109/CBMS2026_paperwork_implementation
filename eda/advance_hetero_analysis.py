import torch
import os
import sys
# Putanja do modula
sys.path.append('/home/ivanam/CBMS_bioWork/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from eda.load_hetero_data import load_hetero_biosnap

def run_analysis(data, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'hetero_analysis_report.txt')
    
    print(f"Analiziram heterogeni graf i pišem izveštaj u: {report_path}")
    
    with open(report_path, 'w') as f:
        f.write("=== HETEROGENEOUS KNOWLEDGE GRAPH REPORT ===\n\n")
        f.write("1. NODE STATISTICS\n")
        for node_type in data.node_types:
            f.write(f"  - {node_type.capitalize()}: {data[node_type].num_nodes} nodes\n")
        f.write("\n2. EDGE STATISTICS\n")
        for edge_type in data.edge_types:
            f.write(f"  - {edge_type}: {data[edge_type].num_edges} edges\n")

        f.write("\n3. CROSS-DOMAIN CONNECTIVITY (THE BRIDGE)\n")
        
        drug_deg_disease = degree(data['drug', 'treats', 'disease'].edge_index[0], data['drug'].num_nodes).numpy()
        drug_deg_gene = degree(data['drug', 'targets', 'gene'].edge_index[0], data['drug'].num_nodes).numpy()
        
        df_drugs = pd.DataFrame({'deg_dis': drug_deg_disease, 'deg_gene': drug_deg_gene})

        for t in [1, 2, 3]:
            cold_mask = df_drugs['deg_dis'] <= t
            cold_count = cold_mask.sum()
            recovered = df_drugs[cold_mask & (df_drugs['deg_gene'] >= 5)].shape[0]
            
            f.write(f"Drugs with <= {t} disease connections: {cold_count}\n")
            f.write(f"  -> Potential knowledge recovery (>= 5 gene links): {recovered} drugs\n")

        f.write("\n4. LATEX SUMMARY TABLE\n" + "-"*15 + "\n")
        f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Entity & Count & Relations \\\\ \n\\midrule\n")
        f.write(f"Drugs & {data['drug'].num_nodes} & {data['drug','treats','disease'].num_edges} (Treats) \\\\ \n")
        f.write(f"Genes & {data['gene'].num_nodes} & {data['drug','targets','gene'].num_edges} (Targets) \\\\ \n")
        f.write(f"Diseases & {data['disease'].num_nodes} & - \\\\ \n\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table}\n")

    plt.figure(figsize=(10, 8))
    sns.set_context("paper", font_scale=1.2)
    joint_plot = sns.jointplot(data=df_drugs, x='deg_dis', y='deg_gene', kind="reg", color="teal")
    joint_plot.set_axis_labels('Degree in Disease Domain', 'Degree in Gene Domain')
    plt.savefig(os.path.join(output_dir, 'hetero_degree_plot.png'), dpi=300)
    print("Grafikon sačuvan: hetero_degree_plot.png")

if __name__ == "__main__":
    CHG_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/ChG-Miner_miner-chem-gene.tsv.gz'
    DCH_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'
    
    hetero_data = load_hetero_biosnap(CHG_PATH, DCH_PATH)
    run_analysis(hetero_data)
    print("\nGotovo! Pogledaj 'results' folder.")