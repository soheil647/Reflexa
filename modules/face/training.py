# modules/face/training.py
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class FaceTrainer:
    def __init__(self, embeddings_dir="data/faces/embeddings", device='cpu'):
        """
        embeddings_dir: Directory to store face embeddings.
        """
        self.embeddings_dir = embeddings_dir
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        self.device = device
        # Use the same face embedding model as in FaceRecognizer.
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def compute_embedding(self, face_image):
        """
        Compute the embedding for a cropped face image.
        face_image: Expected to be in BGR (OpenCV) format.
        Returns a 512-dim numpy array.
        """
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()  # shape: [3, 160, 160]
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.cpu().numpy()[0]

    def register_new_face(self, name, face_image):
        """
        Registers (trains) a new face by computing its embedding and saving it.
        """
        embedding = self.compute_embedding(face_image)
        file_path = os.path.join(self.embeddings_dir, f"{name}.npy")
        np.save(file_path, embedding)
        logger.info(f"Registered new face for {name} at {file_path}")
