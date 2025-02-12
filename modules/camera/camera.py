# modules/camera/camera.py
import cv2
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Cannot open camera")
            raise Exception("Cannot open camera")
        logger.info("Camera started successfully")
        return self.cap

    def stop(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera stopped")
