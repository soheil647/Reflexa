# tests/test_unified.py
import cv2
import time

# Camera
from modules.camera.camera import Camera
# Face detection & recognition
from modules.face.detection import FaceDetector
from modules.face.recognition import FaceRecognizer
# Health modules
from modules.health.rppg_tscan import RPPGTSCan
from modules.health.emotion import EmotionRecognizer
from modules.health.skin import SkinAnalyzer


def test_unified_health():
    """
    1) Capture frames from the camera in real time.
    2) Detect & crop largest face.
    3) Pass cropped face ROI to:
       - RPPGTSCan (sliding window) => HR & SpO2
       - EmotionRecognizer
       - SkinAnalyzer
    4) Display bounding box, recognized name, emotion, skin label
    5) Periodically print out HR & SpO2
    6) Press 'q' to exit
    """

    # 1) Initialize camera
    camera = Camera(camera_id=0)
    cap = camera.start()

    # 2) Initialize face detection & recognition
    face_detector = FaceDetector(device='cuda')   # or 'cpu'
    face_recognizer = FaceRecognizer(device='cuda')  # optional

    # 3) Initialize health modules
    rppg_estimator = RPPGTSCan(device='cuda', frames_window=160, sampling_rate=30)
    emotion_model = EmotionRecognizer(device='cuda')
    skin_model = SkinAnalyzer(device='cuda')

    frame_idx = 0
    print("Starting unified health test. Press 'q' to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Detect faces
        faces = face_detector.detect_faces(frame)
        if len(faces) > 0:
            # Pick the largest face by area
            largest_face = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            x1, y1, x2, y2 = largest_face

            # Crop face ROI
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                # (Optional) Face Recognition
                result = face_recognizer.recognize(face_roi)
                if result:
                    name = result['name']
                    sim = result['similarity']
                    cv2.putText(
                        frame, f"{name} ({sim:.2f})",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                # 4a) rPPG: feed cropped face to sliding window
                rppg_estimator.process_face_frame(face_roi)

                # 4b) Emotion
                emotion = emotion_model.predict_emotion(face_roi)

                # 4c) Skin
                skin_label = skin_model.detect_skin_issue(face_roi)

                # Draw bounding box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show emotion & skin text near the face
                offset = 20
                if emotion:
                    cv2.putText(frame, f"Emotion: {emotion}",
                                (x1, y2 + offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    offset += 25

                if skin_label:
                    cv2.putText(frame, f"Skin: {skin_label}",
                                (x1, y2 + offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    offset += 25

        # 5) Periodically compute HR & SpO2 (e.g. every 30 frames => ~1 second)
        if frame_idx % 30 == 0:
            hr = rppg_estimator.estimate_heart_rate()  # won't clear buffer
            spo2 = rppg_estimator.estimate_spo2()
            if hr is not None:
                print(f"[Frame {frame_idx}] HR: {hr:.2f} BPM, SpO2: {spo2}")

        # Show the live feed
        cv2.imshow("Smart Mirror - Unified Health", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

        frame_idx += 1

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_unified_health()
