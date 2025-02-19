# tests/test_health.py
import time
import cv2

from modules.camera.camera import Camera
from modules.health.detection import HealthAnalyzer

def test_health_detection():
    # Initialize camera
    camera = Camera(camera_id=0)
    cap = camera.start()

    # Create the health analyzer
    analyzer = HealthAnalyzer(sampling_rate=30, window_duration=20, device='cuda')
    print("Starting health detection. Please stand in front of the mirror for ~20 seconds.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Process the current frame
        analyzer.process_frame(frame)

        # (Optional) Show the feed
        cv2.imshow("Smart Mirror Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early exit.")
            break

        # Check if we've collected enough data
        if analyzer.is_collection_complete():
            print("Collection complete.")
            break

    # Once the window is complete, get final results
    results = analyzer.get_results()
    print("Health Detection Results:", results)

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_health_detection()
