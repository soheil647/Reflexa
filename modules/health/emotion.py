# modules/health/emotion.py
import torch
import torch.nn as nn
import cv2
import numpy as np

from modules.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionRecognizer:
    def __init__(self, device='cpu'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info("Loading pretrained ResNet18 for emotion (no local checkpoint).")

        # 1) Load pretrained model (initially on CPU)
        self.model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet18', pretrained=True)

        # 2) Replace final FC with 7 outputs
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, 7)

        # 3) Move entire model (including new layer) to device
        self.model.to(self.device)

        # 4) Set to eval mode
        self.model.eval()

        # 7 typical emotion labels
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def predict_emotion(self, roi):
        if roi.size == 0:
            logger.warning("Empty ROI for emotion.")
            return None

        # Convert BGR->RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))

        # Create tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,224,224]
        tensor = tensor.to(self.device)

        # Normalize (ImageNet)
        tensor /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            logits = self.model(tensor)  # => [1,7]
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()
            emotion = self.emotion_labels[idx]
        return emotion
