import cv2
import time

from modules.camera.camera import Camera
from modules.face.detection import FaceDetector
from modules.face.recognition import FaceRecognizer
from modules.health.rppg_tscan import RPPGTSCan

def test_rppg_sliding():
    """
    1) Capture frames from the camera in real time.
    2) Detect & crop largest face.
    3) Pass to RPPGTSCan with a sliding window.
    4) Estimate HR & SpO2 every few frames (or every second) once buffer is full.
    5) Press 'q' to exit.
    """
    camera = Camera(camera_id=0)
    cap = camera.start()

    face_detector = FaceDetector(device='cuda')  # or 'cpu'
    face_recognizer = FaceRecognizer(device='cuda')  # optional
    rppg_estimator = RPPGTSCan(device='cuda', frames_window=160, sampling_rate=30)

    frame_idx = 0
    print("Starting continuous rPPG test (sliding window). Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        faces = face_detector.detect_faces(frame)
        if len(faces) > 0:
            # pick the largest face
            largest_face = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            x1, y1, x2, y2 = largest_face
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                # optional face recognition
                result = face_recognizer.recognize(face_roi)
                if result:
                    name = result['name']
                    sim = result['similarity']
                    cv2.putText(
                        frame, f"{name} ({sim:.2f})",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                # feed to rPPG sliding window
                rppg_estimator.process_face_frame(face_roi)

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Estimate every 30 frames, for example (~1 second at 30 FPS)
        # Once the buffer has at least frames_window frames
        if frame_idx % 30 == 0:
            hr = rppg_estimator.estimate_heart_rate()  # won't clear buffer
            spo2 = rppg_estimator.estimate_spo2()
            if hr is not None:
                print(f"[Frame {frame_idx}] HR: {hr:.2f} BPM, SpO2: {spo2}")

        cv2.imshow("Smart Mirror Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

        frame_idx += 1

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_rppg_sliding()
