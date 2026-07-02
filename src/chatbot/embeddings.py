import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.utils.helpers import get_device
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = str(get_device())
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Embedding model loaded on {self.device}")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)

    def compute_similarity(self, query_embedding: np.ndarray,
                           doc_embeddings: np.ndarray) -> np.ndarray:
        return np.dot(doc_embeddings, query_embedding)

    def build_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False):
        import faiss
        dimension = embeddings.shape[1]
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_factory(dimension, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT)
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(index)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        index.add_with_ids(embeddings, np.arange(len(embeddings)))
        return index
