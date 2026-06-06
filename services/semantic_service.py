import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from app.config import config
from pydantic import BaseModel

logger = structlog.get_logger()

class SemanticRequest(BaseModel):
    text1: str
    text2: str

class SemanticResponse(BaseModel):
    similarity: float

class SemanticService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            model_name = getattr(config, "SEMANTIC_MODEL", "all-MiniLM-L6-v2")
            device = getattr(config, "WHISPER_DEVICE", "cpu")
            logger.info("Loading semantic transformer model", model=model_name, device=device)
            # Load the sentence transformer model
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("Semantic model loaded successfully")
        except Exception as e:
            logger.error("Failed to load semantic model", error=str(e))
            self.model = None

    async def compute_similarity(self, text1: str, text2: str) -> SemanticResponse:
        if not self.model:
            raise RuntimeError("Semantic model not loaded")
            
        try:
            # Compute embeddings
            embeddings = self.model.encode([text1, text2])
            
            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm1 * norm2))
            
            # In node.js we mapped [-1, 1] to [0, 1] using: (sim + 1) / 2
            normalized_sim = (similarity + 1) / 2
            clamped_sim = float(max(0.0, min(1.0, normalized_sim)))
            
            return SemanticResponse(similarity=clamped_sim)
        except Exception as e:
            logger.error("Failed to compute similarity", error=str(e))
            raise e
