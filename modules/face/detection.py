# modules/face/detection.py
from modules.utils.logger import get_logger
logger = get_logger(__name__)

class FaceDetector:
    """
    Now delegates to a Triton-based face detection service
    rather than using MTCNN locally.
    """
    def __init__(self, detection_service):
        """
        :param detection_service: An instance of FaceDetectionTritonService
        """
        self.detection_service = detection_service

    def detect_faces(self, frame):
        """
        Calls the Triton-based detection service to get bounding boxes.
        """
        boxes = self.detection_service.detect_faces(frame)
        logger.debug(f"Detected faces: {boxes}")
        return boxes
