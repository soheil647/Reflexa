# modules/health/skin.py
import torch
import torch.nn as nn
import cv2
import numpy as np

from modules.utils.logger import get_logger

logger = get_logger(__name__)

class SkinAnalyzer:
    """
    Basic 2-class approach (normal vs. issue).
    No fine-tuning => random out-of-the-box results.
    """
    def __init__(self, device='cpu'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info("Loading pretrained ResNet50 for skin detection (no local checkpoint).")

        self.model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True).to(self.device)
        self.model.eval()
        # Replace final FC with 2 outputs
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, 2)

        # 3) Move entire model (including new layer) to device
        self.model.to(self.device)

        # 4) Set to eval mode
        self.model.eval()

        self.labels = ["normal", "issue"]

    def detect_skin_issue(self, roi):
        if roi.size == 0:
            logger.warning("Empty ROI for skin detection.")
            return None
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224,224))
        tensor = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).to(self.device)
        tensor /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            logits = self.model(tensor)  # => [1,2]
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()
            return self.labels[idx]
