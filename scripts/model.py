"""
model.py — GATClassifier with configurable pooling.

Pooling options
---------------
mean_max  concat(global_mean_pool, global_max_pool)  — output dim = hidden*2
mean      global_mean_pool                           — output dim = hidden
max       global_max_pool                            — output dim = hidden
sum       global_add_pool                            — output dim = hidden
diff      Differential Hierarchical Pooling (DHP)    — output dim = hidden*2

DiffPool (diff)
---------------
A learnable soft-assignment pooling layer that clusters nodes into
`diff_clusters` super-nodes, then applies a global mean+max over them.
It adds two auxiliary losses (link_loss + entropy_loss) accessible via
`model.aux_loss` after each forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool, global_add_pool

try:
    from torch_geometric.nn import dense_diff_pool
    from torch_geometric.utils import to_dense_batch, to_dense_adj
    _DIFF_POOL_OK = True
except ImportError:
    _DIFF_POOL_OK = False

POOLING_OPTIONS = ("mean_max", "mean", "max", "sum", "diff")


class DiffPoolReadout(nn.Module):
    """Differential Hierarchical Pooling readout.

    Learns soft cluster assignments S ∈ R^{N×K} over the node embeddings
    produced by the GAT layers, then pools to K super-nodes.

    The auxiliary loss (link_loss + entropy_loss) is stored in `self.aux_loss`
    after each forward call so the caller can add it to the main criterion.
    """

    def __init__(self, in_channels: int, hidden: int, n_clusters: int = 10):
        super().__init__()
        if not _DIFF_POOL_OK:
            raise ImportError(
                "DiffPool requires torch_geometric.nn.dense_diff_pool and "
                "torch_geometric.utils.{to_dense_batch, to_dense_adj}. "
                "Please upgrade torch-geometric."
            )
        self.n_clusters = n_clusters
        self.aux_loss   = torch.tensor(0.0)

        # Assignment GNN: maps node embeddings → soft cluster assignments
        self.assign_net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_clusters),
        )
        # Embedding GNN: maps node embeddings → cluster feature space
        self.embed_net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
        )

    def forward(
        self,
        x:          torch.Tensor,   # (total_nodes, in_channels)
        edge_index: torch.Tensor,   # (2, E)
        batch:      torch.Tensor,   # (total_nodes,)
    ) -> torch.Tensor:              # (B, hidden*2)
        x_dense, mask = to_dense_batch(x, batch)       # (B, N_max, C)
        adj           = to_dense_adj(edge_index, batch) # (B, N_max, N_max)

        s = self.assign_net(x_dense)  # (B, N_max, K) — cluster assignments
        z = self.embed_net(x_dense)   # (B, N_max, hidden) — cluster features

        x_pool, _, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        # x_pool: (B, K, hidden)

        self.aux_loss = link_loss + ent_loss

        # Global mean + max over the K super-nodes → (B, hidden*2)
        mean_h = x_pool.mean(dim=1)
        max_h  = x_pool.max(dim=1).values
        return torch.cat([mean_h, max_h], dim=1)


class GATClassifier(nn.Module):
    """
    Three-layer Graph Attention Network with configurable readout pooling.

    Architecture (for all pooling types):
        GATConv(in → hidden, heads)         → BN → ELU → Dropout
        GATConv(hidden*heads → hidden, heads) → BN → ELU → Dropout
        GATConv(hidden*heads → hidden, 1)   → BN → ELU
        <pooling readout>                   → pool_out_dim
        Linear(pool_out_dim → hidden) → ReLU → Dropout → Linear(hidden → 2)
    """

    def __init__(
        self,
        in_channels:   int,
        hidden:        int,
        heads:         int,
        dropout:       float,
        pooling:       str = "mean_max",
        diff_clusters: int = 10,
    ):
        super().__init__()
        assert pooling in POOLING_OPTIONS, (
            f"pooling must be one of {POOLING_OPTIONS}, got '{pooling}'"
        )
        self.dropout      = dropout
        self.pooling_type = pooling

        self.conv1 = GATConv(in_channels,    hidden, heads=heads, concat=True,  dropout=dropout)
        self.bn1   = nn.BatchNorm1d(hidden * heads)

        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True,  dropout=dropout)
        self.bn2   = nn.BatchNorm1d(hidden * heads)

        self.conv3 = GATConv(hidden * heads, hidden, heads=1,     concat=False, dropout=dropout)
        self.bn3   = nn.BatchNorm1d(hidden)

        if pooling == "mean_max":
            pool_out = hidden * 2
        elif pooling in ("mean", "max", "sum"):
            pool_out = hidden
        elif pooling == "diff":
            self.diff_pool = DiffPoolReadout(hidden, hidden, n_clusters=diff_clusters)
            pool_out = hidden * 2

        self.mlp = nn.Sequential(
            nn.Linear(pool_out, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    # ── readout ────────────────────────────────────────────────────────────────

    def pool_readout(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured pooling and return the graph-level embedding h."""
        if self.pooling_type == "mean_max":
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=1)
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        if self.pooling_type == "max":
            return global_max_pool(x, batch)
        if self.pooling_type == "sum":
            return global_add_pool(x, batch)
        if self.pooling_type == "diff":
            return self.diff_pool(x, edge_index, batch)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
    ) -> torch.Tensor:
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        h = self.pool_readout(x, edge_index, batch)
        return self.mlp(h)

    # ── aux loss ───────────────────────────────────────────────────────────────

    @property
    def aux_loss(self) -> torch.Tensor:
        """DiffPool auxiliary loss; zero for all other pooling types."""
        if self.pooling_type == "diff":
            return self.diff_pool.aux_loss
        return torch.tensor(0.0)
