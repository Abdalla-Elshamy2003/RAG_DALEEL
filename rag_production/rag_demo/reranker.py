import torch
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # BGE-Reranker is a Cross-Encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder('BAAI/bge-reranker-v2-m3', device=self.device)
        if self.device == "cuda":
            self.model.model.half() # Use FP16

    def rerank(self, query, documents, top_n=3):
        if not documents:
            return []
        
        # Prepare pairs for the Cross-Encoder
        sentence_pairs = [[query, doc['text']] for doc in documents]
        scores = self.model.predict(sentence_pairs)
        
        for i, score in enumerate(scores):
            documents[i]['rerank_score'] = float(score)
            
        # Sort by score and take top_n
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_n]