# modules/face/recognition.py
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class FaceRecognizer:
    def __init__(self, embeddings_dir="data/faces/embeddings", device='cpu', threshold=0.8):
        """
        embeddings_dir: Directory where embeddings are stored (as .npy files).
        threshold: Minimum cosine similarity to consider a face a match.
        """
        self.embeddings_dir = embeddings_dir
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        self.device = device
        # Load a state-of-the-art face embedding model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.threshold = threshold
        self.known_embeddings = {}  # Dictionary mapping name -> embedding vector
        self.load_known_faces()

    def load_known_faces(self):
        """Loads saved embeddings from disk."""
        self.known_embeddings = {}
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.npy'):
                name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.embeddings_dir, filename)
                embedding = np.load(file_path)
                self.known_embeddings[name] = embedding
                logger.info(f"Loaded embedding for {name}")

    def compute_embedding(self, face_image):
        """
        Compute the embedding for a cropped face image.
        face_image: Expected to be in BGR (OpenCV) format.
        Returns a 512-dim numpy array.
        """
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Resize to 160x160 (the input size expected by InceptionResnetV1)
        face_resized = cv2.resize(face_rgb, (160, 160))
        # Convert to tensor and normalize to [-1, 1]
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()  # shape: [3, 160, 160]
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.cpu().numpy()[0]

    def recognize(self, face_image):
        """
        Recognize the face in the provided cropped face image.
        Returns a dictionary with the name and similarity if a match is found,
        or None otherwise.
        """
        embedding = self.compute_embedding(face_image)
        best_match = None
        best_similarity = -1
        for name, known_embedding in self.known_embeddings.items():
            # Compute cosine similarity between the embeddings.
            sim = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
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
