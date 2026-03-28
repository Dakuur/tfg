"""
generate_graph.py
-----------------
Generates a fully-connected graph with features on each node,
x/y positions, and exports everything to a JSON file.
"""

import json
import random
import math
import argparse


def generate_fully_connected_graph(
    num_nodes: int = 10,
    num_features: int = 4,
    seed: int = 42,
    output_file: str = "graph_data.json",
):
    random.seed(seed)

    # ---- Nodes ----
    nodes = []
    for i in range(num_nodes):
        # Position on a circle with added noise
        angle  = 2 * math.pi * i / num_nodes
        radius = 1.0 + random.uniform(-0.2, 0.2)
        x      = radius * math.cos(angle) + random.uniform(-0.1, 0.1)
        y      = radius * math.sin(angle) + random.uniform(-0.1, 0.1)

        features = [round(random.gauss(0, 1), 6) for _ in range(num_features)]
        label    = random.randint(0, 2)  # random class 0-2

        nodes.append({
            "id":       i,
            "x":        round(x, 6),
            "y":        round(y, 6),
            "features": features,
            "label":    label,
        })

    # ---- Edges (fully connected, bidirectional) ----
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Weight based on inverse Euclidean distance
                dx     = nodes[i]["x"] - nodes[j]["x"]
                dy     = nodes[i]["y"] - nodes[j]["y"]
                dist   = math.sqrt(dx**2 + dy**2)
                weight = round(1.0 / (dist + 1e-6), 6)
                edges.append({
                    "src":    i,
                    "dst":    j,
                    "weight": weight,
                })

    graph = {
        "meta": {
            "num_nodes":    num_nodes,
            "num_edges":    len(edges),
            "num_features": num_features,
            "seed":         seed,
        },
        "nodes": nodes,
        "edges": edges,
    }

    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Graph saved to '{output_file}'")
    print(f"  Nodes   : {num_nodes}")
    print(f"  Edges   : {len(edges)}  (fully connected)")
    print(f"  Features: {num_features} per node")
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a fully-connected graph")
    parser.add_argument("--nodes",    type=int, default=1000, help="Number of nodes")
    parser.add_argument("--features", type=int, default=256,  help="Features per node")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default="graph_data.json")
    args = parser.parse_args()

    generate_fully_connected_graph(
        num_nodes=args.nodes,
        num_features=args.features,
        seed=args.seed,
        output_file=args.output,
    )
