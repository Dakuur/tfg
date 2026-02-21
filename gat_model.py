"""
gat_model.py
------------
Carga un grafo desde JSON (generado por generate_graph.py) e implementa:
  - Graph Attention Network (GAT) desde cero con NumPy/PyTorch
  - Múltiples estrategias de pooling: max, mean, sum, top-k, hierarchical

Requisitos: pip install torch numpy matplotlib
"""

import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===========================================================================
# 1. Carga de datos
# ===========================================================================

def load_graph(path: str):
    with open(path) as f:
        g = json.load(f)

    meta  = g["meta"]
    nodes = g["nodes"]
    edges = g["edges"]

    N = meta["num_nodes"]
    F = meta["num_features"]

    # Matriz de features [N, F]
    X = torch.tensor(
        [n["features"] for n in nodes], dtype=torch.float32
    )

    # Posiciones [N, 2]
    pos = torch.tensor(
        [[n["x"], n["y"]] for n in nodes], dtype=torch.float32
    )

    # Labels [N]
    labels = torch.tensor([n["label"] for n in nodes], dtype=torch.long)

    # Lista de aristas [E, 2]  (src, dst)
    edge_index = torch.tensor(
        [[e["src"], e["dst"]] for e in edges], dtype=torch.long
    ).t().contiguous()   # shape [2, E]

    # Pesos de aristas [E]
    edge_weight = torch.tensor(
        [e["weight"] for e in edges], dtype=torch.float32
    )

    return X, pos, labels, edge_index, edge_weight, meta


# ===========================================================================
# 2. GAT Layer (implementación manual, sin PyG)
# ===========================================================================

class GATLayer(nn.Module):
    """
    Una capa de Graph Attention Network.

    h_i' = σ( Σ_{j∈N(i)} α_ij · W · h_j )

    donde α_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
    """

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int = 1, dropout: float = 0.6,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_heads    = num_heads
        self.concat       = concat
        self.dropout      = dropout

        # Proyección lineal por cabeza
        self.W = nn.Parameter(
            torch.empty(num_heads, in_features, out_features)
        )
        # Vector de atención por cabeza: opera sobre [Wh_i || Wh_j]
        self.a = nn.Parameter(
            torch.empty(num_heads, 2 * out_features)
        )

        self.leaky_relu = nn.LeakyReLU(alpha)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.view(self.num_heads * self.in_features,
                                             self.out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        """
        h          : [N, in_features]
        edge_index : [2, E]   (src, dst)
        returns    : [N, num_heads * out_features]  if concat
                     [N, out_features]               if not concat (mean)
        """
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # h_proj: [N, H, F_out]
        h_proj = torch.einsum("ni,hio->nho", h, self.W)

        # Para cada arista: concat features de src y dst
        h_src = h_proj[src]   # [E, H, F_out]
        h_dst = h_proj[dst]   # [E, H, F_out]
        e_ij  = torch.cat([h_src, h_dst], dim=-1)  # [E, H, 2*F_out]

        # Puntuación de atención
        score = self.leaky_relu(
            (e_ij * self.a.unsqueeze(0)).sum(dim=-1)  # [E, H]
        )

        # Softmax sobre vecinos de cada nodo destino
        # Usamos scatter softmax manual
        alpha = self._sparse_softmax(score, dst, N)  # [E, H]

        # Dropout sobre atención
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Agregación: h_out[i] = Σ_j α_ij * h_proj[j]
        h_out = torch.zeros(N, self.num_heads, self.out_features,
                            device=h.device)
        alpha_exp = alpha.unsqueeze(-1)                # [E, H, 1]
        h_out.index_add_(0, dst,
                         alpha_exp * h_src)            # mensaje de src a dst

        if self.concat:
            h_out = h_out.view(N, self.num_heads * self.out_features)
        else:
            h_out = h_out.mean(dim=1)                  # [N, F_out]

        return F.elu(h_out)

    @staticmethod
    def _sparse_softmax(score: torch.Tensor,
                        dst: torch.Tensor, N: int) -> torch.Tensor:
        """softmax por nodo destino para cada cabeza."""
        # score: [E, H], dst: [E]
        score_max = torch.full((N, score.size(1)), -1e9, device=score.device)
        score_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(score),
                                  score, reduce="amax", include_self=True)
        score_exp = torch.exp(score - score_max[dst])

        denom = torch.zeros(N, score.size(1), device=score.device)
        denom.index_add_(0, dst, score_exp)
        denom = denom.clamp(min=1e-16)

        return score_exp / denom[dst]


# ===========================================================================
# 3. GAT completo
# ===========================================================================

class GAT(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_classes: int,
                 heads: int = 4, dropout: float = 0.6):
        super().__init__()
        self.layer1 = GATLayer(in_features, hidden,
                               num_heads=heads, dropout=dropout, concat=True)
        self.layer2 = GATLayer(hidden * heads, out_classes,
                               num_heads=1, dropout=dropout, concat=False)
        self._drop_rate = dropout  # renombrado para evitar conflicto con nn.Module

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self._drop_rate, training=self.training)
        x = self.layer1(x, edge_index)
        x = F.dropout(x, p=self._drop_rate, training=self.training)
        x = self.layer2(x, edge_index)
        return x   # logits por nodo [N, out_classes]


# ===========================================================================
# 4. Pooling global
# ===========================================================================

class GraphPooling:
    """
    Diferentes estrategias de max pooling y alternativas para
    reducir todo el grafo a un único vector (graph-level).
    """

    @staticmethod
    def max_pool(node_embeddings: torch.Tensor) -> torch.Tensor:
        """Max pooling clásico: valor máximo por feature."""
        return node_embeddings.max(dim=0).values

    @staticmethod
    def mean_pool(node_embeddings: torch.Tensor) -> torch.Tensor:
        """Mean pooling: promedio por feature."""
        return node_embeddings.mean(dim=0)

    @staticmethod
    def sum_pool(node_embeddings: torch.Tensor) -> torch.Tensor:
        """Sum pooling: suma por feature."""
        return node_embeddings.sum(dim=0)

    @staticmethod
    def topk_pool(node_embeddings: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Top-K pooling: selecciona los k nodos con mayor norma L2
        y aplica max sobre ellos.
        """
        norms = node_embeddings.norm(dim=1)
        k = min(k, node_embeddings.size(0))
        topk_idx = norms.topk(k).indices
        return node_embeddings[topk_idx].max(dim=0).values

    @staticmethod
    def hierarchical_pool(node_embeddings: torch.Tensor,
                          pos: torch.Tensor,
                          clusters: int = 3) -> torch.Tensor:
        """
        Hierarchical pooling simplificado: agrupa nodos por k-means de posición,
        aplica max dentro de cada clúster, luego max global entre clústeres.
        """
        N = node_embeddings.size(0)
        clusters = min(clusters, N)

        # K-Means simple con NumPy
        emb_np  = node_embeddings.detach().numpy()
        pos_np  = pos.detach().numpy()

        # Inicializar centroides con k-means++
        rng = np.random.default_rng(0)
        centroids = pos_np[rng.choice(N, clusters, replace=False)]

        for _ in range(20):
            dists    = np.linalg.norm(pos_np[:, None] - centroids[None], axis=2)
            assign   = dists.argmin(axis=1)
            new_c    = np.array([
                pos_np[assign == c].mean(axis=0) if (assign == c).any()
                else centroids[c]
                for c in range(clusters)
            ])
            if np.allclose(centroids, new_c, atol=1e-4):
                break
            centroids = new_c

        cluster_reprs = []
        for c in range(clusters):
            mask = assign == c
            if mask.any():
                cluster_reprs.append(
                    torch.tensor(emb_np[mask]).max(dim=0).values
                )
        cluster_stack = torch.stack(cluster_reprs)
        return cluster_stack.max(dim=0).values

    @classmethod
    def all_poolings(cls, node_embeddings: torch.Tensor,
                     pos: torch.Tensor) -> dict:
        return {
            "max":          cls.max_pool(node_embeddings),
            "mean":         cls.mean_pool(node_embeddings),
            "sum":          cls.sum_pool(node_embeddings),
            "topk_3":       cls.topk_pool(node_embeddings, k=3),
            "hierarchical": cls.hierarchical_pool(node_embeddings, pos),
        }


# ===========================================================================
# 5. Entrenamiento y visualización
# ===========================================================================

def train(model, x, edge_index, labels, epochs=200, lr=0.005, wd=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=wd)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(dim=1)
                acc  = (pred == labels).float().mean().item()
            print(f"Epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.2%}")
    return losses


def visualize(pos, labels, pred, losses, pooling_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -- Grafo con labels reales vs predichos --
    ax = axes[0]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (xi, yi) in enumerate(pos.tolist()):
        c = colors[labels[i].item()]
        ax.scatter(xi, yi, c=c, s=200, zorder=3, edgecolors="black")
        match = "✓" if pred[i] == labels[i] else "✗"
        ax.annotate(f"{i}{match}", (xi, yi),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    handles = [mpatches.Patch(color=colors[c], label=f"Clase {c}") for c in range(3)]
    ax.legend(handles=handles, fontsize=8)
    ax.set_title("Grafo - Clases reales (✓=correcto)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # -- Curva de pérdida --
    axes[1].plot(losses, color="#8e44ad")
    axes[1].set_title("Curva de pérdida (entrenamiento)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)

    # -- Comparación de pooling --
    ax = axes[2]
    names = list(pooling_results.keys())
    # Norma L2 de cada vector pooled como métrica de comparación
    norms = [pooling_results[k].norm().item() for k in names]
    bars = ax.bar(names, norms, color=["#1abc9c","#e67e22","#e74c3c","#3498db","#9b59b6"])
    ax.set_title("Norma L2 del embedding de grafo\n(por estrategia de pooling)")
    ax.set_ylabel("‖embedding‖₂")
    for bar, v in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = "gat_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualización guardada en '{out_path}'")
    plt.show()


# ===========================================================================
# 6. Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAT + Pooling sobre grafo generado por generate_graph.py"
    )
    parser.add_argument("--input",   type=str, default="graph_data.json")
    parser.add_argument("--hidden",  type=int, default=8,
                        help="Dimensión oculta por cabeza de atención")
    parser.add_argument("--heads",   type=int, default=4,
                        help="Número de cabezas de atención (capa 1)")
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--lr",      type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--pooling", type=str, default="all",
                        choices=["max","mean","sum","topk","hierarchical","all"])
    args = parser.parse_args()

    print("=" * 55)
    print(" Graph Attention Network + Pooling")
    print("=" * 55)

    # Cargar grafo
    X, pos, labels, edge_index, edge_weight, meta = load_graph(args.input)
    N, F = X.shape
    num_classes = labels.unique().numel()
    print(f"\nGrafo cargado: {N} nodos, {F} features, {num_classes} clases")
    print(f"Aristas: {edge_index.shape[1]}")

    # Modelo
    model = GAT(
        in_features=F,
        hidden=args.hidden,
        out_classes=num_classes,
        heads=args.heads,
        dropout=args.dropout,
    )
    print(f"\nModelo GAT: {sum(p.numel() for p in model.parameters())} parámetros")
    print(f"  Capa 1: GATLayer({F} → {args.hidden}, heads={args.heads})")
    print(f"  Capa 2: GATLayer({args.hidden * args.heads} → {num_classes}, heads=1)")

    # Entrenamiento
    print(f"\nEntrenando {args.epochs} épocas...")
    losses = train(model, X, edge_index, labels,
                   epochs=args.epochs, lr=args.lr)

    # Embeddings finales
    model.eval()
    with torch.no_grad():
        node_emb = model.layer1(X, edge_index)   # [N, hidden*heads]
        logits   = model(X, edge_index)
        pred     = logits.argmax(dim=1)

    acc = (pred == labels).float().mean().item()
    print(f"\nAccuracy final: {acc:.2%}")

    # Pooling
    print("\n--- Embeddings de grafo (pooling) ---")
    if args.pooling == "all":
        pooling_results = GraphPooling.all_poolings(node_emb, pos)
    else:
        fn_map = {
            "max":          lambda: {"max": GraphPooling.max_pool(node_emb)},
            "mean":         lambda: {"mean": GraphPooling.mean_pool(node_emb)},
            "sum":          lambda: {"sum": GraphPooling.sum_pool(node_emb)},
            "topk":         lambda: {"topk_3": GraphPooling.topk_pool(node_emb)},
            "hierarchical": lambda: {"hierarchical": GraphPooling.hierarchical_pool(node_emb, pos)},
        }
        pooling_results = fn_map[args.pooling]()

    for name, emb in pooling_results.items():
        print(f"  {name:14s}: shape={list(emb.shape)}  "
              f"norm={emb.norm().item():.4f}  "
              f"min={emb.min().item():.4f}  "
              f"max={emb.max().item():.4f}")

    # Visualización
    visualize(pos, labels, pred, losses, pooling_results)