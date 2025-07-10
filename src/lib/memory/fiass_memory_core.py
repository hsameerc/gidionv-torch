import os
import pickle
import time
from typing import List, Dict, Optional, Callable, Any

import faiss
import numpy as np
from numpy import ndarray

EmbeddingFunction = Callable[[str], ndarray]


class FaissMemoryCore:
    """
    A file-backed memory system using FAISS for high-speed vector search
    and a pickled dictionary for storing metadata and raw vectors.

    This class serves as Gideon's long-term, non-parametric memory.
    """

    def __init__(self, base_path: str = "memory/gideon_ltm", dim: int = 512, index_type: str = 'HNSW'):
        """
        Initializes the memory core, loading from disk if files exist.

        :param base_path: The base name for memory files (e.g., 'gideon_ltm').
                          This will create 'gideon_ltm.index', '.meta', and '.vectors'.
        :param dim: The dimensionality of the vectors (e.g., 512).
        :param index_type: The type of FAISS index. 'FLAT' (exact) or 'HNSW' (fast, approximate).
        """
        if dim <= 0:
            raise ValueError("Vector dimension must be a positive integer.")

        self.index_path = f"{base_path}.index"
        self.meta_path = f"{base_path}.meta"
        self.vectors_path = f"{base_path}.vectors"

        self.dim = dim
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.raw_vectors: List[ndarray] = []

        if os.path.exists(self.index_path):
            self._load()
        else:
            self._create(index_type.upper())

        if self.index and self.index.ntotal != len(self.metadata):
            print(f"CRITICAL WARNING: Mismatch between index size ({self.index.ntotal}), "
                  f"metadata ({len(self.metadata)}), and vectors ({len(self.raw_vectors)}). "
                  f"Consider rebuilding the index.")

    def _load(self):
        """Loads the index, metadata, and raw vectors from disk."""
        print(f"[FAISS] Loading existing memory from {self.index_path}...")
        try:
            self.index = faiss.read_index(self.index_path)
            if self.index.d != self.dim:
                raise ValueError(f"Dimension mismatch on load. Index is {self.index.d}D, expected {self.dim}D.")

            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            with open(self.vectors_path, 'rb') as f:
                self.raw_vectors = pickle.load(f)

            print(f"[FAISS] Load complete. {self.index.ntotal} memories loaded.")
        except Exception as e:
            print(f"Error loading memory files: {e}")
            raise

    def _create(self, index_type: str):
        """Creates a new, empty FAISS index and data stores."""
        print(f"[FAISS] Creating new '{index_type}' index with dimension {self.dim}...")
        if index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif index_type == 'FLAT':
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError(f"Unknown index_type '{index_type}'. Use 'FLAT' or 'HNSW'.")

        self.metadata = {}
        self.raw_vectors = []
        print("[FAISS] New memory core created.")

    @staticmethod
    def _normalize(vectors: ndarray) -> ndarray:
        """Normalizes vectors to unit length for cosine similarity with Inner Product."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (vectors / norms).astype(np.float32)

    def save(self):
        """Atomically saves the index, metadata, and vectors to disk."""
        print(f"[FAISS] Saving state ({len(self)} memories)...")
        try:
            cpu_vectors = [v.get() if hasattr(v, 'get') else v for v in self.raw_vectors]
            faiss.write_index(self.index, self.index_path + '.tmp')
            with open(self.meta_path + '.tmp', 'wb') as f_meta:
                pickle.dump(self.metadata, f_meta)
            with open(self.vectors_path + '.tmp', 'wb') as f_vecs:
                pickle.dump(cpu_vectors, f_vecs)

            os.replace(self.index_path + '.tmp', self.index_path)
            os.replace(self.meta_path + '.tmp', self.meta_path)
            os.replace(self.vectors_path + '.tmp', self.vectors_path)
            print("[FAISS] Save successful.")
        except Exception as e:
            print(f"Error saving memory files: {e}")

    def add_memory(self, text: str, embedding_fn: EmbeddingFunction, prev_memory_id: Optional[int] = None) -> int:
        """Adds a single memory to the store and returns its ID."""
        embedding = embedding_fn(text)
        if embedding.shape != (self.dim,):
            raise ValueError(f"Embedding shape mismatch. Expected ({self.dim},), got {embedding.shape}")

        normalized_embedding = self._normalize(embedding)
        new_id = self.index.ntotal
        self.index.add(normalized_embedding.get().reshape(1, -1))
        self.metadata[new_id] = {'id': new_id, 'timestamp': time.time(), 'raw_text': text,
                                 'temporal_link': prev_memory_id, 'associative_link': None}
        self.raw_vectors.append(embedding)
        self.save()
        return new_id

    def find_similar_memories(self, query_text: str, embedding_fn: EmbeddingFunction, top_k: int = 5) -> List[tuple]:
        """Finds the top-K most similar memories to a query text."""
        if self.index.ntotal == 0: return []
        query_embedding = embedding_fn(query_text)
        normalized_query = self._normalize(query_embedding)
        distances, indices = self.index.search(normalized_query.get().reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                meta = self.metadata.get(int(idx))
                if meta:
                    results.append((float(dist), int(idx), meta['raw_text']))
        return results

    def get_vectors_by_ids(self, ids: List[int]) -> Optional[ndarray]:
        """
         Retrieves the raw vectors for a given list of IDs.
        """
        if not ids or not self.raw_vectors:
            return None
        try:
            vectors_to_return = [self.raw_vectors[i] for i in ids]
            return np.stack(vectors_to_return, axis=0)
        except IndexError:
            print(f"[ERROR] Invalid ID found in list: {ids}. Cannot retrieve vectors.")
            return None

    def add_associative_link(self, source_id: int, target_id: int):
        """
        Creates a directed associative link from one memory to another.

        :param source_id: The ID of the memory from which the link originates.
        :param target_id: The ID of the memory to which the link points.
        """
        if source_id not in self.metadata:
            print(f"Warning: Cannot create link. Source ID {source_id} not found.")
            return
        if target_id not in self.metadata:
            print(f"Warning: Cannot create link. Target ID {target_id} not found.")
            return

        self.metadata[source_id]['associative_link'] = target_id
        print(f"[FAISS] Linked memory {source_id} -> {target_id}")

    def find_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """
        Retrieves a memory's metadata by its unique ID.

        :param memory_id: The ID of the memory to retrieve.
        :return: A dictionary containing the memory's metadata, or None if not found.
        """
        return self.metadata.get(memory_id)

    def get_recent_memories(self, num_to_get: int) -> List[Dict]:
        """
        Retrieves the metadata for the most recent N memories.
        :param num_to_get: The number of recent memories to retrieve.
        :return: A list of metadata dictionaries, sorted from most to least recent.
        """
        if self.index.ntotal == 0:
            return []
        start_id = max(0, self.index.ntotal - num_to_get)
        end_id = self.index.ntotal
        recent_memories = [self.metadata[i] for i in range(start_id, end_id)]
        return sorted(recent_memories, key=lambda x: x['id'], reverse=True)

    def __len__(self) -> int:
        """Returns the total number of memories in the store."""
        return self.index.ntotal

    def close(self):
        """Saves the final state of the memory to disk."""
        self.save()
        print("[FAISS] Memory Core closed and state saved.")
