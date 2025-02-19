# test_face_pipeline.py

import cv2
import time

# Services
from services.FaceDetectionTritonService import FaceDetectionTritonService
from services.FaceEmbeddingTritonService import FaceEmbeddingTritonService

# Modules (logic)
from modules.face.detection import FaceDetector
from modules.face.recognition import FaceRecognizer
from modules.face.training import FaceTrainer
# Camera module
from modules.camera.camera import Camera

def main():
    # 1) Initialize detection & embedding services
    detect_service = FaceDetectionTritonService(
        model_name="face_detect",
        triton_url="localhost:8001",
        input_name="images",
        output_names=("boxes", "scores", "labels"),
        input_shape=(1,3,640,640),
        score_threshold=0.5
    )
    embed_service = FaceEmbeddingTritonService(
        model_name="face_embedding",
        triton_url="localhost:8001",
        input_name="input",
        output_name="output_embedding",
        input_shape=(1,3,160,160)
    )

    # 2) Wrap them with your module logic
    face_detector = FaceDetector(detect_service)
    face_recognizer = FaceRecognizer(
        embedding_service=embed_service,
        embeddings_dir="data/faces/embeddings",
        threshold=0.8
    )
    face_trainer = FaceTrainer(
        embedding_service=embed_service,
        embeddings_dir="data/faces/embeddings"
    )

    # 3) Start the camera
    camera = Camera(camera_id=0)  # Adjust if you have multiple cameras
    cap = camera.start()

    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break

        # 4) Detect bounding boxes
        boxes = face_detector.detect_faces(frame)

        if boxes:
            # Take the first box for demonstration
            x1, y1, x2, y2 = boxes[0]
            # Ensure the box is within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            if x2 > x1 and y2 > y1:  # valid box
                face_img = frame[y1:y2, x1:x2]

                # 5) Try recognition
                result = face_recognizer.recognize(face_img)
                if result is None:
                    # Register a default name or prompt user, for demonstration we use "TestUser"
                    # In a real system, you'd ask user for name or use a UI
                    name = "TestUser"
                    print(f"No match found. Registering new face as '{name}'...")
                    face_trainer.register_new_face(name, face_img)
                    # Check again
                    result2 = face_recognizer.recognize(face_img)
                    if result2:
                        print(f"Now recognized as {result2['name']} (sim={result2['similarity']:.2f})")
                        label = result2['name']
                    else:
                        print("Still not recognized, something might be off.")
                        label = "Unknown"
                else:
                    print(f"Recognized user: {result['name']} (sim={result['similarity']:.2f})")
                    label = result['name']

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 6) Display the result
        cv2.imshow("Detection & Recognition Test", frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Small delay to reduce CPU usage (optional)
        time.sleep(0.05)

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
