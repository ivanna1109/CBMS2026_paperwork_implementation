import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv, to_hetero

class HeteroEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # GATConv sa (-1, -1) automatski detektuje ulazne dimenzije
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_dict, edge_label_index):
        # Predviđamo vezu između leka i bolesti
        # z_dict sadrži embeddinge za sve tipove čvorova
        z_drug = z_dict['drug'][edge_label_index[0]]
        z_disease = z_dict['disease'][edge_label_index[1]]
        return (z_drug * z_disease).sum(dim=-1)
    

class HeteroSAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv je često stabilniji i brži od GAT-a
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x