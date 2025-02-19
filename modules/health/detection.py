# modules/health/detection.py
import time
import numpy as np
import cv2

from modules.face.detection import FaceDetector
from modules.health.rppg_tscan import RPPGTSCan
from modules.health.emotion import EmotionRecognizer
from modules.health.stress import StressEstimator
from modules.health.skin import SkinAnalyzer

from modules.utils.logger import get_logger

logger = get_logger(__name__)

class HealthAnalyzer:
    """
    Aggregates advanced signals:
      - Heart Rate (rPPG)
      - SpO2
      - Emotion
      - Stress
      - Skin Health
    using your existing MTCNN face detection + Torch-based modules.
    """

    def __init__(self, sampling_rate=30, window_duration=20, device='cpu'):
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.start_time = None

        # Reuse your existing face detection from modules/face/detection.py
        self.face_detector = FaceDetector(device=device)
        # rPPG (TS-CAN style)
        self.rppg_estimator = RPPGTSCan(device=device)
        # Emotion
        self.emotion_recognizer = EmotionRecognizer(device=device)
        # Stress
        self.stress_estimator = StressEstimator()
        # Skin
        self.skin_analyzer = SkinAnalyzer(device=device)

        self.emotion_buffer = []
        self.skin_buffer = []

    def start_collection(self):
        self.start_time = time.time()
        logger.info("HealthAnalyzer: Started collecting frames.")

    def process_frame(self, frame):
        if self.start_time is None:
            self.start_collection()

        # 1) Detect faces with MTCNN
        faces = self.face_detector.detect_faces(frame)
        if len(faces) == 0:
            return  # No face found

        # If multiple faces, pick the largest
        largest_face = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = largest_face

        # Crop face ROI
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return

        # 2) Feed face ROI to rPPG
        self.rppg_estimator.process_face_frame(face_roi)

        # 3) Emotion
        emotion = self.emotion_recognizer.predict_emotion(face_roi)
        self.emotion_buffer.append(emotion)

        # 4) Stress (heuristic) - we don't have final HR yet, so approximate
        #    We'll do a final pass after we have HR. But let's do partial updates:
        approx_heart_rate = 80  # a placeholder guess for real-time updates
        self.stress_estimator.update_stress(approx_heart_rate, emotion)

        # 5) Skin
        skin_issue = self.skin_analyzer.detect_skin_issues(face_roi)
        self.skin_buffer.append(skin_issue)

    def is_collection_complete(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.window_duration

    def get_results(self):
        """
        Once the collection window is done, finalize the results
        (HR, SpO2, emotion, stress, skin).
        """
        heart_rate = self.rppg_estimator.estimate_heart_rate()
        spo2 = self.rppg_estimator.estimate_spo2()

        # Dominant emotion from the buffer
        if len(self.emotion_buffer) > 0:
            unique, counts = np.unique(self.emotion_buffer, return_counts=True)
            dominant_emotion = unique[np.argmax(counts)]
        else:
            dominant_emotion = None

        # Now that we have a final HR, do a final stress update:
        if heart_rate and dominant_emotion:
            self.stress_estimator.update_stress(heart_rate, dominant_emotion)
        avg_stress = self.stress_estimator.get_average_stress()

        # Skin: any issue across the entire buffer
        final_skin_issue = any(self.skin_buffer)

        results = {
            "heart_rate_bpm": round(float(heart_rate), 2) if heart_rate else None,
            "spo2_percent": round(float(spo2), 2) if spo2 else None,
            "dominant_emotion": dominant_emotion,
            "stress_score": round(float(avg_stress), 3) if avg_stress else None,
            "skin_issues_detected": final_skin_issue
        }

        # Clear buffers for next session
        self._reset()
        return results

    def _reset(self):
        """Clears buffers for next usage."""
        self.rppg_estimator.clear()
        self.emotion_buffer = []
        self.stress_estimator.clear()
        self.skin_buffer = []
        self.start_time = None
