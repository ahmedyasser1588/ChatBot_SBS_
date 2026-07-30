import os
import uuid
from typing import List

import chromadb
import numpy as np


class VectorStore:
    """نفس الكلاس بالظبط اللي في worldcubChatbot، بس اسم الـ collection مختلف
    (spotme_players بدل pdf_documents)."""

    def __init__(self, collection_name="spotme_players", persist_directory="data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "SpotMe player profiles for RAG",
                    "hnsw:space": "cosine",  # explicit cosine similarity
                },
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def count(self) -> int:
        """Return number of documents currently stored. main.py's /api/health calls this."""
        if not self.collection:
            return 0
        return self.collection.count()

    def add_documents(self, texts: List[str], metadatas: List[dict], embeddings):
        if len(texts) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(texts)} documents to vector store...")

        ids, embeddings_list = [], []
        for i, embedding in enumerate(embeddings):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            arr = np.asarray(embedding, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            embeddings_list.append(arr.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts,
            )
            print(f"Successfully added {len(texts)} documents")
            print(f"Total in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
