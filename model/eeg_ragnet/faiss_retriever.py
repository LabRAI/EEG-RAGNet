import json
import numpy as np
import faiss
import torch

class KnowledgeRetriever:
    """
    KnowledgeRetriever module: performs high-speed similarity search between EEG semantic queries (Q_t)
    and pre-computed biomedical knowledge embeddings using FAISS.

    Each EEG node query vector retrieves top-k semantically similar triplets (head, relation, tail)
    from the knowledge graph, enabling RAG-based contextual refinement of dynamic brain graphs.
    """

    def __init__(self, index_path: str, kg_json_path: str, k: int = 5, normalize: bool = True):
        """
        Initialize the KnowledgeRetriever.

        Args:
            index_path (str): Path to FAISS index file.
            kg_json_path (str): Path to JSON file containing triplets and embeddings.
            k (int): Number of top results to retrieve per query.
            normalize (bool): Whether to normalize query vectors before search.
        """
        self.index_path = index_path
        self.kg_json_path = kg_json_path
        self.k = k
        self.normalize = normalize

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load structured knowledge entries (list of dict triplets)
        with open(kg_json_path, "r", encoding="utf-8") as f:
            self.kg = json.load(f)

        # Ensure alignment between FAISS vectors and knowledge entries
        if len(self.kg) != self.index.ntotal:
            raise ValueError(
                f"Mismatch: KG entries ({len(self.kg)}) != FAISS index vectors ({self.index.ntotal})"
            )

    def _prepare_query(self, Q: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy and normalize if required.

        Args:
            Q (torch.Tensor): EEG semantic query embeddings (batch_size, num_nodes, proj_dim)

        Returns:
            np.ndarray: Query matrix (N, D)
        """
        if Q.dim() == 3:
            Q = Q.reshape(-1, Q.size(-1))  # Flatten batch and node dims
        Q_np = Q.detach().cpu().numpy().astype("float32")
        if self.normalize:
            faiss.normalize_L2(Q_np)
        return Q_np

    def retrieve(self, Q: torch.Tensor):
        """
        Perform batched FAISS search for the top-k most similar knowledge entries per EEG node query.

        Args:
            Q (torch.Tensor): EEG semantic query embeddings (batch_size, num_nodes, proj_dim)

        Returns:
            List[List[dict]]: For each EEG node query, a list of top-k triplets with similarity scores.
        """
        Q_np = self._prepare_query(Q)
        D, I = self.index.search(Q_np, self.k)  # D: distances, I: indices

        retrieved_all = []
        for node_idx, (distances, indices) in enumerate(zip(D, I)):
            node_results = []
            for d, idx in zip(distances, indices):
                if idx < len(self.kg):
                    entry = self.kg[idx]
                    node_results.append({
                        "head": entry.get("head", None),
                        "relation": entry.get("relation", None),
                        "tail": entry.get("tail", None),
                        "similarity": float(d)
                    })
            retrieved_all.append(node_results)

        return retrieved_all


if __name__ == "__main__":
    # Example test: assumes FAISS index and KG JSON already exist
    index_path = "knowledge/faiss.index"
    kg_json_path = "knowledge/kg_triplets.json"

    # Create a fake query tensor (batch_size=1, num_nodes=5, proj_dim=256)
    Q = torch.randn(1, 5, 256)

    retriever = KnowledgeRetriever(index_path, kg_json_path, k=3)
    results = retriever.retrieve(Q)

    # Print top-1 retrieved relation for each node
    for i, node_results in enumerate(results):
        top1 = node_results[0] if node_results else None
        print(f"Node {i} top-1:", top1)
