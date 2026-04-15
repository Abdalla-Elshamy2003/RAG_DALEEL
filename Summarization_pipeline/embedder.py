import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from config import config

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        logger.info(f"Loading local BGE-M3 on {config.embed_device}...")
        # تحميل الموديل محلياً باستخدام المسار الموجود في config
        self.model = SentenceTransformer(config.embed_model, device=config.embed_device)

    def embed(self, text: str) -> list[float]:
        if not text: 
            return []
        # تحويل النص إلى Vector
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def cluster_by_similarity(self, doc_embeddings: list[dict]):
        if not doc_embeddings:
            return [], []

        matrix = np.array([d["embedding"] for d in doc_embeddings])
        dist_threshold = 1.0 - config.cluster_similarity_threshold

        # استخدام AgglomerativeClustering للتقسيم لمجموعات
        model = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=dist_threshold
        )
        
        labels = model.fit_predict(matrix)
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(doc_embeddings[idx]["doc_id"])
        
        clusters = [ids for ids in groups.values() if len(ids) >= 2]
        singletons = [ids[0] for ids in groups.values() if len(ids) == 1]
        
        return clusters, singletons