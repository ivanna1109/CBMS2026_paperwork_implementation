import pandas as pd

from torch_geometric.data import HeteroData
import torch
import pandas as pd
import os

def check_overlap(chg_path, dch_path):
    # 1. Učitavanje (oba su TSV u BioSNAP-u, bez obzira na ekstenziju)
    print("Učitavam fajlove...")
    df_chg = pd.read_csv(chg_path, sep='\t', compression='gzip')
    df_dch = pd.read_csv(dch_path, sep='\t', compression='gzip')

    # 2. Identifikacija kolona prema tvojoj load_hetero_biosnap logici
    # ChG: Drug je u prvoj koloni (index 0)
    # DCh: Drug je u drugoj koloni (index 1)
    chg_drugs_all = df_chg.iloc[:, 0].astype(str).unique()
    dch_drugs_all = df_dch.iloc[:, 1].astype(str).unique()

    chg_drugs_set = set(chg_drugs_all)
    dch_drugs_set = set(dch_drugs_all)

    # 3. Analiza preklapanja
    common = chg_drugs_set.intersection(dch_drugs_set)
    
    print("\n" + "="*30)
    print("ANALIZA POKLAPANJA LEKOVA")
    print("="*30)
    print(f"Lekova u ChG (Drug-Gene):     {len(chg_drugs_set)}")
    print(f"Lekova u DCh (Drug-Disease):  {len(dch_drugs_set)}")
    print(f"Zajedničkih lekova:           {len(common)}")
    
    if len(common) > 0:
        percentage = (len(common) / len(dch_drugs_set)) * 100
        print(f"Procenat pokrivenosti DCh:    {percentage:.2f}%")
        print("\nPrimeri zajedničkih ID-jeva:")
        print(list(common)[:5])
    else:
        print("\nUPOZORENJE: Nema zajedničkih lekova! Proveri separatore ili kolone.")
    print("="*30)

# Putanje
chg_path = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/ChG-Miner_miner-chem-gene.tsv.gz'
dch_path = '/home/ivanam/CBMS_bioWork/eda/data/BioSNAP/DCh-Miner_miner-disease-chemical.tsv.gz'

if __name__ == "__main__":
    check_overlap(chg_path, dch_path)