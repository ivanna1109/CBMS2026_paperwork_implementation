import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class GNN_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GAT'):
        super(GNN_Model, self).__init__()
        self.model_type = model_type
        
        if model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            # Koristimo 4 glave (heads) za stabilnost, ali izlaz mora biti out_channels
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
            self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def encode(self, x, edge_index):
        # Prvi sloj + Aktivacija + Dropout (bitno za retke grafove)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Drugi sloj (Latentni prostor / Embeddings)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Link Prediction: Dot product između čvorova koji čine ivicu
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def decode_all(self, z):
        # Korisno za krajnju vizuelizaciju cele matrice
        prob_adj = z @ z.t()
        return torch.sigmoid(prob_adj)