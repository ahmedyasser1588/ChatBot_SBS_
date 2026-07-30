import numpy as np


class RAGRetriever:
    """نفس الكلاس بالظبط من worldcubChatbot — بيعمل embedding للسؤال، يطبعه (normalize)،
    يدور في ChromaDB، ويرجع أعلى top_k نتايج بالـ cosine similarity."""

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0, where: dict = None):
        print(f"Retrieving for: '{query}' | top_k={top_k} | threshold={score_threshold} | where={where}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        norm = np.linalg.norm(query_embedding)
        query_norm = query_embedding / norm if norm > 0 else query_embedding

        query_kwargs = {
            "query_embeddings": [query_norm.tolist()],
            "n_results": top_k,
        }
        if where:
            query_kwargs["where"] = where

        results = self.vector_store.collection.query(**query_kwargs)

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                cosine_sim = 1.0 - distance
                if cosine_sim >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": cosine_sim,
                        "rank": i + 1,
                    })

        print(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
