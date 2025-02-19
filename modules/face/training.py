# modules/face/training.py
import os
import numpy as np
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class FaceTrainer:
    """
    Uses the same Triton-based embedding service to register new faces
    (compute embedding + save to disk).
    """
    def __init__(self, embedding_service, embeddings_dir="data/faces/embeddings"):
        self.embedding_service = embedding_service
        self.embeddings_dir = embeddings_dir
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)

    def register_new_face(self, name, face_image):
        """
        1) Compute embedding (via Triton).
        2) Save to .npy file.
        """
        embedding = self.embedding_service.compute_embedding(face_image)
        if embedding is None:
            logger.warning("No embedding returned by Triton. Registration failed.")
            return

        file_path = os.path.join(self.embeddings_dir, f"{name}.npy")
        np.save(file_path, embedding)
        logger.info(f"Registered new face for {name} at {file_path}")
