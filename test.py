print("Test!!!")

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

# --------- Config ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes = 20000      # súbelo para más carga GPU
num_edges_prob = 0.0005
in_channels = 64
hidden_channels = 128
out_channels = 10
batch_size = 4

print("Device:", device)

# --------- Modelo ----------
class GATWithPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.pool1 = TopKPooling(hidden_channels * 4, ratio=0.5)

        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.pool2 = TopKPooling(hidden_channels * 4, ratio=0.5)

        self.lin = torch.nn.Linear(hidden_channels * 4, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.elu(self.gat2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = global_mean_pool(x, batch)
        return self.lin(x)

# --------- Datos sintéticos ----------
edge_index = erdos_renyi_graph(num_nodes, num_edges_prob)
x = torch.randn((num_nodes, in_channels))
batch = torch.zeros(num_nodes, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, batch=batch).to(device)

# --------- Run ----------
model = GATWithPooling().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

torch.cuda.synchronize()
for step in range(20):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = out.mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"Step {step} | Loss {loss.item():.4f}")

print("Done.")
