# Overcoming Small-Degree Cold-Start in Heterogeneous Biological Networks through Latent Bridges

##  Overview

This work addresses the critical cold-start challenge in link prediction within biological knowledge graphs. We propose a cross-domain message-passing architecture that utilizes a **latent gene-bridge** to provide topological context for poorly connected or "unseen" drug-disease associations.

### Key Highlights:
- **Architectural Trade-off:** Comparison between GraphSAGE (stable mean-pooling) and GAT (self-attention).
- **Attention Collapse:** Analysis of why self-attention fails in extreme sparsity.
- **Latent Bridge:** Demonstrating how auxiliary genomic data compensates for primary disease domain sparsity.
- **Performance:** GraphSAGE achieves a Cold AUC of **0.8945**, significantly outperforming GAT in sparse scenarios.

## Repository Structure

```text
├── eda/                        # Exploratory Data Analysis notebooks and dataset processing scripts
├── train_experiments/          # Core training scripts and GNN implementations
└── README.md                   # This file
