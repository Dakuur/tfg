"""
model.py — GATClassifier with configurable pooling + patient-level MIL aggregator.

Pooling options (intra-slide, within one WSI section graph)
------------------------------------------------------------
mean_max  concat(global_mean_pool, global_max_pool)  — output dim = hidden*2
mean      global_mean_pool                           — output dim = hidden
max       global_max_pool                            — output dim = hidden
sum       global_add_pool                            — output dim = hidden
diff      Differential Hierarchical Pooling (DHP)    — output dim = hidden*2

Patient-level aggregation (inter-slide, across all slides of a patient)
------------------------------------------------------------------------
noisy_or   P(pat=N1) = 1 - Π(1-P(slide_i=N1))   [default — probabilistic OR]
max        max(P(slide_i=N1))
lse        log-sum-exp smooth maximum
mean       mean(P(slide_i=N1))
attention  Gated Attention MIL (Ilse et al., 2018) — uses PatientAggregator

Usage
-----
For slide-level prediction (e.g. frontend visualisation):
    logits = model(x, edge_index, batch)

For patient-level training (all slides at once):
    h      = model.encode(x, edge_index, batch)   # (n_slides, D)
    pat_h  = patient_aggregator(h)                # (1, D)
    logits = model.mlp(pat_h)                     # (1, 2)
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

POOLING_OPTIONS      = ("mean_max", "mean", "max", "sum", "diff")
AGGREGATION_OPTIONS  = ("noisy_or", "max", "lse", "mean", "attention")


# ── Intra-slide: DiffPool readout ──────────────────────────────────────────────

class DiffPoolReadout(nn.Module):
    """Differential Hierarchical Pooling readout (intra-slide).

    Learns soft cluster assignments S ∈ R^{N×K} over node embeddings,
    pools to K super-nodes, then applies global mean+max over them.
    Auxiliary loss (link + entropy) is stored in `self.aux_loss`.
    """

    def __init__(self, in_channels: int, hidden: int, n_clusters: int = 10):
        super().__init__()
        if not _DIFF_POOL_OK:
            raise ImportError(
                "DiffPool requires torch_geometric >= 2.x "
                "(dense_diff_pool, to_dense_batch, to_dense_adj)."
            )
        self.n_clusters = n_clusters
        self.aux_loss   = torch.tensor(0.0)

        self.assign_net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, n_clusters),
        )
        self.embed_net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
        )

    def forward(self, x, edge_index, batch):
        x_dense, mask = to_dense_batch(x, batch)
        adj           = to_dense_adj(edge_index, batch)

        s = self.assign_net(x_dense)
        z = self.embed_net(x_dense)

        x_pool, _, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        self.aux_loss = link_loss + ent_loss

        mean_h = x_pool.mean(dim=1)
        max_h  = x_pool.max(dim=1).values
        return torch.cat([mean_h, max_h], dim=1)   # (B, hidden*2)


# ── Inter-slide: Gated Attention MIL ──────────────────────────────────────────

class PatientAggregator(nn.Module):
    """Gated Attention MIL (Ilse et al., 2018) — aggregates slide embeddings
    to a single patient-level representation.

    Input:  H ∈ R^{n_slides × embed_dim}
    Output: h_pat ∈ R^{1 × embed_dim}

    Attention:
        A_i = softmax( w^T · (tanh(V·h_i) ⊙ σ(U·h_i)) )
        h_pat = Σ A_i · h_i
    """

    def __init__(self, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden)
        self.U = nn.Linear(embed_dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (n_slides, embed_dim)  →  (1, embed_dim)"""
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))  # (n_slides, 1)
        A = F.softmax(A, dim=0)
        return (A * H).sum(dim=0, keepdim=True)                        # (1, embed_dim)

    @property
    def embed_dim(self) -> int:
        return self.V.in_features


# ── Main model ─────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):
    """
    Three-layer Graph Attention Network with configurable readout pooling.

    The model has two usage modes:

    Slide-level  (standard):
        logits = model(x, edge_index, batch)

    Patient-level (training with MIL):
        h = model.encode(x, edge_index, batch)  # returns embedding before MLP
        # aggregate h across slides (externally), then:
        logits = model.mlp(patient_h)
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

    def pool_readout(self, x, edge_index, batch):
        """Apply the configured pooling → graph-level embedding h."""
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

    # ── encode (without MLP head) ──────────────────────────────────────────────

    def encode(self, x, edge_index, batch):
        """Run GAT encoder + pooling.  Returns embedding before the MLP head.

        With a batched PyG graph (n graphs), returns shape (n, pool_out).
        Used for patient-level MIL training: call this on all slides of a
        patient at once, then aggregate with PatientAggregator, then call
        self.mlp() on the aggregated vector.
        """
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        return self.pool_readout(x, edge_index, batch)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x, edge_index, batch):
        return self.mlp(self.encode(x, edge_index, batch))

    # ── aux loss ───────────────────────────────────────────────────────────────

    @property
    def aux_loss(self):
        """DiffPool auxiliary loss; zero for all other pooling types."""
        if self.pooling_type == "diff":
            return self.diff_pool.aux_loss
        return torch.tensor(0.0)
