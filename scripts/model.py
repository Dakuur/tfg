"""
model.py — GATClassifier with configurable pooling + patient-level MIL aggregator.

Pooling options (intra-slide, within one WSI section graph)
------------------------------------------------------------
mean_max  concat(global_mean_pool, global_max_pool)  — output dim = hidden*2
mean      global_mean_pool                           — output dim = hidden
max       global_max_pool                            — output dim = hidden
sum       global_add_pool                            — output dim = hidden
diff      Hierarchical DiffPool between GAT layers   — output dim = hidden*2
          Architecture: GAT1 → DiffPool1(N→K1) → GAT2 → DiffPool2(K1→K2) → GAT3 → mean+max
          K1 = diff_clusters, K2 = max(diff_clusters // 3, 5)

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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention, global_max_pool, global_mean_pool, global_add_pool

try:
    from torch_geometric.nn import dense_diff_pool
    from torch_geometric.utils import to_dense_batch, to_dense_adj
    _DIFF_POOL_OK = True
except ImportError:
    _DIFF_POOL_OK = False

POOLING_OPTIONS     = ("mean_max", "mean", "max", "sum", "diff", "attention")
AGGREGATION_OPTIONS = ("noisy_or", "max", "lse", "mean", "attention")


# ── Hierarchical DiffPool step ─────────────────────────────────────────────────

class HierarchicalDiffPool(nn.Module):
    """One step of hierarchical DiffPool: reduces N sparse nodes → K super-nodes.

    Returns the pooled nodes in sparse PyG format (x_flat, edge_index, batch)
    so subsequent GATConv layers can operate on the reduced graph.

    Super-node connectivity is fully connected (adj_pool ≈ dense for soft S),
    which is cheap since K is small (O(K²) edges per graph in the batch).

    Auxiliary loss (link + entropy) is accumulated in `self.aux_loss`.
    """

    def __init__(self, in_channels: int, hidden: int, n_clusters: int):
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

        # Pre-build the local fully-connected edge_index for K nodes (no self-loops)
        rows = torch.arange(n_clusters)
        src  = rows.repeat(n_clusters)
        dst  = rows.repeat_interleave(n_clusters)
        no_self = src != dst
        self.register_buffer("_local_ei", torch.stack([src[no_self], dst[no_self]]))

    def forward(
        self,
        x:          torch.Tensor,
        edge_index:  torch.Tensor,
        batch:       torch.Tensor,
    ):
        """
        Args:
            x          : (N, in_channels)
            edge_index : (2, E) sparse edges
            batch      : (N,)  graph assignment per node
        Returns:
            x_flat     : (B*K, hidden)
            new_ei     : (2, B*K*(K-1))
            new_batch  : (B*K,)
        """
        x_dense, mask = to_dense_batch(x, batch)
        adj           = to_dense_adj(edge_index, batch, max_num_nodes=x_dense.size(1))

        s = self.assign_net(x_dense)   # (B, max_N, K)
        z = self.embed_net(x_dense)    # (B, max_N, hidden)

        x_pool, _, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        # x_pool: (B, K, hidden)
        self.aux_loss = link_loss + ent_loss

        B = x_pool.size(0)
        K = self.n_clusters

        x_flat    = x_pool.reshape(B * K, -1)
        new_batch = torch.arange(B, device=x.device).repeat_interleave(K)

        # Offset the pre-built local edge_index per batch item
        offsets = (torch.arange(B, device=x.device) * K).repeat_interleave(
            self._local_ei.size(1)
        )
        new_ei = self._local_ei.repeat(1, B) + offsets.unsqueeze(0)

        return x_flat, new_ei, new_batch


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
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))
        A = F.softmax(A, dim=0)
        return (A * H).sum(dim=0, keepdim=True)


# ── Main model ─────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):
    """
    Three-layer Graph Attention Network with configurable readout pooling.

    For pooling='diff', DiffPool is applied hierarchically between GAT layers:
        GAT1 → DiffPool1(N→K1) → GAT2 → DiffPool2(K1→K2) → GAT3 → mean+max

    For all other pooling types, the three GAT layers process the original
    graph topology and a single global pool is applied at the end.

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
        in_channels:         int,
        hidden:              int,
        heads:               int,
        dropout:             float,
        pooling:             str = "mean_max",
        diff_clusters:       int = 10,
        patient_aggregation: str = "noisy_or",
    ):
        super().__init__()
        assert pooling in POOLING_OPTIONS, (
            f"pooling must be one of {POOLING_OPTIONS}, got '{pooling}'"
        )
        assert patient_aggregation in AGGREGATION_OPTIONS, (
            f"patient_aggregation must be one of {AGGREGATION_OPTIONS}"
        )
        self.dropout             = dropout
        self.pooling_type        = pooling
        self.patient_aggregation = patient_aggregation

        # Layer 1 — same for all pooling types
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.bn1   = nn.BatchNorm1d(hidden * heads)

        if pooling == "diff":
            # Hierarchical DiffPool: pooling between layers
            K1 = diff_clusters
            K2 = max(diff_clusters // 3, 5)
            self.diff_pool1 = HierarchicalDiffPool(hidden * heads, hidden, K1)
            self.diff_pool2 = HierarchicalDiffPool(hidden * heads, hidden, K2)
            # Layer 2 & 3 take `hidden` input (embed_net output of DiffPool)
            self.conv2 = GATConv(hidden,          hidden, heads=heads, concat=True,  dropout=dropout)
            self.bn2   = nn.BatchNorm1d(hidden * heads)
            self.conv3 = GATConv(hidden,          hidden, heads=1,     concat=False, dropout=dropout)
            self.bn3   = nn.BatchNorm1d(hidden)
            pool_out   = hidden * 2   # mean + max of final K2 super-nodes
        else:
            self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True,  dropout=dropout)
            self.bn2   = nn.BatchNorm1d(hidden * heads)
            self.conv3 = GATConv(hidden * heads, hidden, heads=1,     concat=False, dropout=dropout)
            self.bn3   = nn.BatchNorm1d(hidden)
            if pooling == "mean_max":
                pool_out = hidden * 2
            elif pooling == "attention":
                self.global_attn = GlobalAttention(nn.Linear(hidden, 1))
                pool_out = hidden
            else:
                pool_out = hidden

        if patient_aggregation == "attention":
            self.patient_aggregator = PatientAggregator(pool_out)

        self.mlp = nn.Sequential(
            nn.Linear(pool_out, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    # ── readout (non-diff pooling only) ────────────────────────────────────────

    def pool_readout(self, x, edge_index, batch):
        """Global graph-level pool for non-diff pooling types."""
        if self.pooling_type == "mean_max":
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=1)
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        if self.pooling_type == "max":
            return global_max_pool(x, batch)
        if self.pooling_type == "sum":
            return global_add_pool(x, batch)
        if self.pooling_type == "attention":
            return self.global_attn(x, batch)

    # ── encode (without MLP head) ──────────────────────────────────────────────

    def encode(self, x, edge_index, batch):
        """Run GAT encoder + pooling.  Returns embedding before the MLP head.

        With a batched PyG graph (n graphs), returns shape (n, pool_out).
        Used for patient-level MIL training: call this on all slides of a
        patient at once, then aggregate with PatientAggregator, then call
        self.mlp() on the aggregated vector.
        """
        if self.pooling_type == "diff":
            return self._encode_hierarchical(x, edge_index, batch)
        return self._encode_standard(x, edge_index, batch)

    def _encode_standard(self, x, edge_index, batch):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        return self.pool_readout(x, edge_index, batch)

    def _encode_hierarchical(self, x, edge_index, batch):
        """Hierarchical path: DiffPool progressively reduces nodes between layers."""
        # GAT layer 1: N nodes → hidden*heads features
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # DiffPool 1: N → K1 super-nodes
        x, edge_index, batch = self.diff_pool1(x, edge_index, batch)

        # GAT layer 2: K1 nodes → hidden*heads features
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # DiffPool 2: K1 → K2 super-nodes
        x, edge_index, batch = self.diff_pool2(x, edge_index, batch)

        # GAT layer 3: K2 nodes → hidden features
        x = F.elu(self.bn3(self.conv3(x, edge_index)))

        # Final readout: mean + max over K2 super-nodes
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

    # ── patient-level aggregation ──────────────────────────────────────────────

    def _aggregate_patients(
        self, h: torch.Tensor, patient_batch: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate slide embeddings → patient logits (n_patients, 2).

        Args:
            h             : (n_slides, pool_out) — output of encode()
            patient_batch : (n_slides,) — patient index per slide (0-based)
        Returns:
            logits (n_patients, 2)
        """
        n_patients = int(patient_batch.max().item()) + 1

        if self.patient_aggregation == "attention":
            pat_h = torch.cat(
                [self.patient_aggregator(h[patient_batch == i])
                 for i in range(n_patients)],
                dim=0,
            )
            return self.mlp(pat_h)

        slide_logits = self.mlp(h)
        results: list[torch.Tensor] = []
        for i in range(n_patients):
            sl = slide_logits[patient_batch == i]
            if self.patient_aggregation == "lse":
                logit_pat = torch.logsumexp(sl[:, 1], dim=0)
            else:
                p_n1 = F.softmax(sl, dim=1)[:, 1]
                if self.patient_aggregation == "noisy_or":
                    p_pat = 1.0 - torch.prod(1.0 - p_n1)
                elif self.patient_aggregation == "mean":
                    p_pat = p_n1.mean()
                else:  # max
                    p_pat = p_n1.max()
                p_pat     = p_pat.clamp(1e-7, 1.0 - 1e-7)
                logit_pat = torch.log(p_pat / (1.0 - p_pat))
            results.append(torch.stack([torch.zeros_like(logit_pat), logit_pat]))

        return torch.stack(results, dim=0)  # (n_patients, 2)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:             torch.Tensor,
        edge_index:    torch.Tensor,
        batch:         torch.Tensor,
        patient_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, batch)
        if patient_batch is None:
            return self.mlp(h)
        return self._aggregate_patients(h, patient_batch)

    # ── aux loss ───────────────────────────────────────────────────────────────

    @property
    def aux_loss(self):
        """Sum of DiffPool auxiliary losses; zero for all other pooling types."""
        if self.pooling_type == "diff":
            return self.diff_pool1.aux_loss + self.diff_pool2.aux_loss
        return torch.tensor(0.0)
