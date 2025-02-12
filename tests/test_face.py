# tests/test_face.py
import cv2
from modules.face.detection import FaceDetector
from modules.face.recognition import FaceRecognizer

def test_face_detection():
    # Load a sample image with a face (ensure this file exists in your data folder)
    image = cv2.imread("data/sample_face.jpg")
    detector = FaceDetector()
    faces = detector.detect_faces(image)
    assert len(faces) > 0, "No faces detected in the sample image"

def test_face_recognition():
    image = cv2.imread("data/sample_face.jpg")
    recognizer = FaceRecognizer()
    profile = recognizer.recognize(image)
    # Adjust expectations based on your test images
    assert profile is not None, "Face recognition failed for known face"
