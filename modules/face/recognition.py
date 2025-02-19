# modules/face/recognition.py
import os
import cv2
import numpy as np
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class FaceRecognizer:
    """
    Uses a Triton-based embedding service to compute embeddings,
    then compares to known embeddings stored on disk or in memory.
    """
    def __init__(self, embedding_service, embeddings_dir="data/faces/embeddings", threshold=0.8):
        """
        :param embedding_service: An instance of FaceEmbeddingTritonService
        :param embeddings_dir: Directory where .npy files are stored
        :param threshold: Cosine similarity threshold for recognition
        """
        self.embedding_service = embedding_service
        self.embeddings_dir = embeddings_dir
        self.threshold = threshold
        self.known_embeddings = {}  # name -> embedding vector
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        self.load_known_faces()

    def load_known_faces(self):
        """Loads saved embeddings from disk."""
        self.known_embeddings.clear()
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.npy'):
                name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.embeddings_dir, filename)
                embedding = np.load(file_path)
                self.known_embeddings[name] = embedding
                logger.info(f"Loaded embedding for {name}")

    def recognize(self, face_image):
        """
        Compute embedding (via Triton) for the cropped face_image,
        compare to known embeddings, return best match if above threshold.
        """
        embedding = self.embedding_service.compute_embedding(face_image)
        if embedding is None:
            logger.warning("No embedding returned by Triton.")
            return None

        best_match = None
        best_similarity = -1
        for name, known_emb in self.known_embeddings.items():
            sim = self._cosine_similarity(embedding, known_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_match = name

        logger.info(f"Best similarity: {best_similarity} for {best_match}")
        if best_similarity >= self.threshold:
            logger.info(f"Recognized face as {best_match} with similarity {best_similarity}")
            return {"name": best_match, "similarity": best_similarity}
        else:
            logger.info("Face not recognized")
            return None

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
