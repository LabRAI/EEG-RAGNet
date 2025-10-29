import torch
import torch.nn as nn
import json
import os
import numpy as np


class GraphRefiner(nn.Module):
    """
    GraphRefiner module:
    Refines the dynamically learned adjacency matrix (A_t) from STGNN using external biomedical
    knowledge retrieved by the KnowledgeRetriever module.
    
    The refinement process fuses learned graph structures with knowledge-informed semantic relationships
    to enhance interpretability and improve temporal stability.
    """

    def __init__(self, threshold: float = 0.6, alpha: float = 0.7, save_path: str = "./results/graph_refinement"):
        """
        Initialize the GraphRefiner.

        Args:
            threshold (float): Confidence threshold for edge pruning.
            alpha (float): Fusion weight between learned A_t and semantic knowledge A_kg.
            save_path (str): Directory to save knowledge-based refinement logs.
        """
        super(GraphRefiner, self).__init__()
        self.threshold = threshold
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.save_path = save_path

        os.makedirs(save_path, exist_ok=True)
        self.support_log_path = os.path.join(save_path, "triplet_supports.json")
        self._support_records = []

    def build_semantic_graph(self, Q: torch.Tensor, retrieved_knowledge: list) -> torch.Tensor:
        """
        Construct a semantic adjacency matrix A_kg based on retrieved knowledge.

        Args:
            Q (torch.Tensor): EEG semantic query embeddings (batch_size, num_nodes, proj_dim)
            retrieved_knowledge (list): KnowledgeRetriever output,
                                        a list of lists containing top-k triplets with similarity scores.

        Returns:
            torch.Tensor: Knowledge-based semantic adjacency matrix A_kg (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = Q.shape
        A_kg = torch.zeros(batch_size, num_nodes, num_nodes, device=Q.device)

        # Construct pairwise semantic relationships based on shared entities or relations
        for b in range(batch_size):
            node_knowledge = retrieved_knowledge  # for single batch; can extend to multi-batch later

            for i in range(num_nodes):
                if i >= len(node_knowledge):
                    continue

                for j in range(num_nodes):
                    if j >= len(node_knowledge) or i == j:
                        continue

                    # Compute relation overlap score between node i and j
                    rel_i = {entry["relation"] for entry in node_knowledge[i] if entry.get("relation")}
                    rel_j = {entry["relation"] for entry in node_knowledge[j] if entry.get("relation")}
                    overlap = len(rel_i.intersection(rel_j))

                    # Compute semantic similarity based on retrieved scores
                    sim_i = np.mean([entry["similarity"] for entry in node_knowledge[i] if "similarity" in entry]) if node_knowledge[i] else 0
                    sim_j = np.mean([entry["similarity"] for entry in node_knowledge[j] if "similarity" in entry]) if node_knowledge[j] else 0
                    semantic_strength = (overlap + 1e-5) * (sim_i + sim_j) / 2.0

                    A_kg[b, i, j] = torch.tensor(semantic_strength, dtype=torch.float32)

        # Normalize A_kg
        A_kg = torch.sigmoid(A_kg)
        return A_kg

    def refine(self, A_t: torch.Tensor, Q: torch.Tensor, retrieved_knowledge: list) -> torch.Tensor:
        """
        Refine the dynamic adjacency matrix using knowledge-based semantic graph.

        Args:
            A_t (torch.Tensor): Original adjacency matrix learned by STGNN (batch_size, num_nodes, num_nodes)
            Q (torch.Tensor): EEG semantic query embeddings (batch_size, num_nodes, proj_dim)
            retrieved_knowledge (list): List of retrieved triplets with similarity values.

        Returns:
            torch.Tensor: Refined adjacency matrix A_refined (batch_size, num_nodes, num_nodes)
        """
        A_kg = self.build_semantic_graph(Q, retrieved_knowledge)

        # Fuse learned graph with knowledge-informed graph
        A_refined = torch.sigmoid(self.alpha * A_t + (1 - self.alpha) * A_kg)

        # Prune edges below confidence threshold
        A_refined[A_refined < self.threshold] = 0.0

        # Save triplet supports for interpretability
        self._save_triplet_supports(retrieved_knowledge)

        return A_refined

    def _save_triplet_supports(self, retrieved_knowledge: list):
        """
        Save retrieved knowledge triplets and their associated similarity scores for interpretability.

        Args:
            retrieved_knowledge (list): Retrieved top-k knowledge triplets per node.
        """
        # Append new retrievals to internal log
        self._support_records.append(retrieved_knowledge)

        with open(self.support_log_path, "w", encoding="utf-8") as f:
            json.dump(self._support_records, f, indent=2, ensure_ascii=False)

    def reset_support_log(self):
        """Clear accumulated triplet support records."""
        self._support_records = []
        if os.path.exists(self.support_log_path):
            os.remove(self.support_log_path)


if __name__ == "__main__":
    # Example test of GraphRefiner
    batch_size = 1
    num_nodes = 5
    proj_dim = 256

    A_t = torch.rand(batch_size, num_nodes, num_nodes)
    Q = torch.rand(batch_size, num_nodes, proj_dim)

    # Example retrieved knowledge (mock data)
    mock_retrieved = [
        [
            {"head": "EEG", "relation": "diagnoses", "tail": "epilepsy", "similarity": 0.88},
            {"head": "MRI", "relation": "supports", "tail": "temporal lobe", "similarity": 0.77}
        ],
        [
            {"head": "Valproate", "relation": "treats", "tail": "seizures", "similarity": 0.92}
        ],
        [
            {"head": "GABA", "relation": "modulates", "tail": "neuronal excitability", "similarity": 0.84}
        ],
        [
            {"head": "EEG", "relation": "records", "tail": "brain activity", "similarity": 0.73}
        ],
        [
            {"head": "Trauma", "relation": "causes", "tail": "epilepsy", "similarity": 0.80}
        ]
    ]

    refiner = GraphRefiner(threshold=0.5, alpha=0.7)
    A_refined = refiner.refine(A_t, Q, mock_retrieved)

    print(f"Original A_t shape: {A_t.shape}")
    print(f"Refined A_refined shape: {A_refined.shape}")
    print("Refinement complete. Saved support log at:", refiner.support_log_path)
