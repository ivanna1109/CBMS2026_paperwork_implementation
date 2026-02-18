import torch
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import degree
from torch_geometric.data import Data

# Apsolutna putanja do fajla koji si skinula
LOCAL_PATH = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'

def load_biosnap_manually(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fajl nije pronađen na lokaciji: {path}")
    
    print(f"Učitavanje lokalnog fajla: {path}")
    # sep='\t' jer je .tsv, compression='gzip' jer je .gz
    df = pd.read_csv(path, sep='\t', compression='gzip')
    
    # Ispisujemo kolone u konzolu čisto da proverimo nazive
    print(f"Kolone u fajlu: {df.columns.tolist()}")
    
    # ChG-Miner obično ima kolone '# Drug' i 'Gene'
    # Koristimo iloc da budemo sigurni da uzimamo prve dve kolone bez obzira na tačan naziv
    col_drug = df.columns[0]
    col_gene = df.columns[1]
    
    # Mapiranje ID-jeva u rang [0, num_nodes-1]
    unique_nodes = pd.unique(df[[col_drug, col_gene]].values.ravel('K'))
    mapping = {node: i for i, node in enumerate(unique_nodes)}
    
    # Kreiranje edge_index-a
    edge_index = torch.tensor([
        df[col_drug].map(mapping).values,
        df[col_gene].map(mapping).values
    ], dtype=torch.long)
    
    # BioSNAP je neusmeren - dupliramo ivice (drug->gene i gene->drug)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    data = Data(edge_index=edge_index, num_nodes=len(unique_nodes))
    return data

# --- Ostatak skripte za statistiku ---

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
stats_file = os.path.join(output_dir, 'dataset_analysis_report-DCh.txt')

with open(stats_file, 'w') as f:
    f.write("=== BIOSNAP DATASET ANALYSIS REPORT (DCh-Miner) ===\n\n")
    
    data = load_biosnap_manually(LOCAL_PATH)

    num_nodes = data.num_nodes
    # Delimo sa 2 jer smo malopre duplirali ivice za GNN, a za statistiku nas zanimaju unikatne veze
    num_edges = data.num_edges // 2
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))

    f.write(f"1. NETWORK TOPOLOGY\n")
    f.write(f"Total Nodes: {num_nodes}\n")
    f.write(f"Total Unique Edges: {num_edges}\n")
    f.write(f"Graph Density: {density:.8e}\n\n")

    # Analiza stepena (koristimo originalni edge_index pre dupliranja ili delimo rezultat)
    node_degrees = degree(data.edge_index[0], num_nodes).numpy()
    df_deg = pd.DataFrame(node_degrees, columns=['degree'])
    
    avg_deg = df_deg['degree'].mean()
    f.write(f"2. DEGREE STATISTICS\n")
    f.write(f"Average Degree: {avg_deg:.2f}\n")
    f.write(f"Max Degree: {df_deg['degree'].max()}\n\n")

    f.write(f"3. COLD START ANALYSIS\n")
    for threshold in [1, 2, 3, 5]:
        count = df_deg[df_deg['degree'] <= threshold].shape[0]
        perc = (count / num_nodes) * 100
        f.write(f"Nodes with degree <= {threshold}: {count} ({perc:.2f}%)\n")
    
    # LaTeX kod...
    f.write(f"\n4. LATEX TABLE\n" + "-"*10 + "\n")
    f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|r|}\n\\hline\n")
    f.write(f"Nodes & {num_nodes} \\\\ \nEdges & {num_edges} \\\\ \n")
    f.write(f"Avg. Degree & {avg_deg:.2f} \\\\ \nDensity & {density:.4e} \\\\ \\hline\n")
    f.write("\\end{tabular}\n\\end{table}\n")

# Vizuelizacija
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.histplot(df_deg['degree'], bins=50, color='red')
plt.yscale('log')
plt.title('Degree Distribution')
plt.savefig(os.path.join(output_dir, 'degree_distribution_DCh.png'))

print(f"Done! Results in: {stats_file}")