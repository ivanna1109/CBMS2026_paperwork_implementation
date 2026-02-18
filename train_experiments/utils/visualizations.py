import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_degree_performance(all_results, save_path='results_final/degree_performance.png'):
    """
    Crta linijski grafik koji poredi GAT i SAGE kroz različite stepene (degrees).
    all_results: lista rečnika [{'Model': 'GAT', 'Threshold': 1, 'AUC': 0.83}, ...]
    """
    df = pd.DataFrame(all_results)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Crtanje linija
    plot = sns.lineplot(data=df, x='Threshold', y='AUC', hue='Model', marker='o', linewidth=2.5)
    
    plt.title('Model Performance by Node Degree (Cold-Start Analysis)', fontsize=14)
    plt.xlabel('Degree Threshold (Drug-Disease Connections)', fontsize=12)
    plt.ylabel('Test ROC-AUC', fontsize=12)
    plt.xticks(df['Threshold'].unique())
    plt.legend(title='Architecture')
    
    # Čuvanje slike
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--- Grafik uspešno sačuvan na: {save_path}")

def save_case_study_table(case_study_data, model_name, save_path):
    """Čuva top predikcije u CSV fajl radi lakšeg kopiranja u rad."""
    df = pd.DataFrame(case_study_data)
    full_path = f"{save_path}_{model_name}_top_preds.csv"
    df.to_csv(full_path, index=False)
    print(f"--- Case study tabela ({model_name}) sačuvana na: {full_path}")