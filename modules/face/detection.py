# modules/face/detection.py
import cv2
from facenet_pytorch import MTCNN
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class FaceDetector:
    def __init__(self, device='cpu'):
        # Initialize MTCNN to detect faces in real time.
        # If you have a GPU, you can set device='cuda'
        self.detector = MTCNN(keep_all=True, device=device)

    def detect_faces(self, frame):
        """
        Detect faces in the provided frame.
        Returns a list of tuples: (x1, y1, x2, y2) for each detected face.
        """
        # Convert BGR (OpenCV) to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.detector.detect(rgb_frame)
        if boxes is None:
            return []
        face_locations = []
        for box in boxes:
            # Round and convert coordinates to integers.
            x1, y1, x2, y2 = [int(b) for b in box]
            face_locations.append((x1, y1, x2, y2))
        logger.debug(f"Detected faces: {face_locations}")
        return face_locations
