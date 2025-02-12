"""
tests/test_camera_face_ui.py

This test script uses the actual camera to capture a face,
then allows the user to:
  - Register (train) a new face profile.
  - Recognize the face from saved profiles.

It provides a simple UI with two buttons using Tkinter.
Additionally, it displays the captured frame with bounding boxes drawn
around detected faces for testing purposes.
"""

import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2

# Import our project modules
from modules.camera.camera import Camera
from modules.face.detection import FaceDetector
from modules.face.training import FaceTrainer
from modules.face.recognition import FaceRecognizer
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class FaceTestUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Camera & Face Test")

        # Initialize our modules
        self.camera = Camera()  # Uses default camera (camera_id=0)
        self.face_detector = FaceDetector()
        self.face_trainer = FaceTrainer()  # Saves embeddings in data/faces/embeddings
        self.face_recognizer = FaceRecognizer()  # Loads embeddings from data/faces/embeddings

        try:
            self.cap = self.camera.start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open camera: {e}")
            self.master.destroy()
            return

        # Create UI buttons
        self.train_button = tk.Button(master, text="Train Face", command=self.train_face, width=25, pady=5)
        self.train_button.pack(pady=10)

        self.recognize_button = tk.Button(master, text="Recognize Face", command=self.recognize_face, width=25, pady=5)
        self.recognize_button.pack(pady=10)

        self.quit_button = tk.Button(master, text="Quit", command=self.quit, width=25, pady=5)
        self.quit_button.pack(pady=10)

    def capture_frame(self):
        """
        Captures a frame from the camera.
        Returns the frame if successful; otherwise, shows an error message.
        """
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame from camera.")
            return None
        return frame

    def display_frame_with_boxes(self, frame, boxes):
        """
        Draws bounding boxes on the frame and displays it in a window.
        """
        frame_copy = frame.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Detected Faces", frame_copy)
        cv2.waitKey(2000)  # Display for 2000 ms (2 seconds)
        cv2.destroyWindow("Detected Faces")

    def train_face(self):
        """
        Captures a frame from the camera, detects a face,
        shows the frame with the detected face bounding box,
        asks the user for a name, and registers the face.
        """
        frame = self.capture_frame()
        if frame is None:
            return

        # Use the face detector to locate face(s)
        faces = self.face_detector.detect_faces(frame)
        if not faces:
            messagebox.showinfo("Info", "No face detected. Please try again.")
            return

        # Show the frame with bounding boxes for testing purposes.
        self.display_frame_with_boxes(frame, faces)

        # For simplicity, take the first detected face.
        x1, y1, x2, y2 = faces[0]
        face_image = frame[y1:y2, x1:x2]

        # Ask user for a name to register the face
        name = simpledialog.askstring("Input", "Enter your name:")
        if name:
            self.face_trainer.register_new_face(name, face_image)
            messagebox.showinfo("Success", f"Face registered for {name}.")
            logger.info(f"Face for {name} trained successfully.")
            # Refresh known faces for recognition (if running in the same session)
            self.face_recognizer.load_known_faces()
        else:
            messagebox.showinfo("Cancelled", "No name provided. Training cancelled.")

    def recognize_face(self):
        """
        Captures a frame from the camera, detects a face,
        shows the frame with the detected face bounding box,
        and attempts to recognize it using stored face profiles.
        """
        frame = self.capture_frame()
        if frame is None:
            return

        # Detect faces in the frame
        faces = self.face_detector.detect_faces(frame)
        if not faces:
            messagebox.showinfo("Info", "No face detected. Please try again.")
            return

        # Show the frame with bounding boxes for testing purposes.
        self.display_frame_with_boxes(frame, faces)

        # Again, use the first detected face
        x1, y1, x2, y2 = faces[0]
        face_image = frame[y1:y2, x1:x2]

        # Use FaceRecognizer to check for a match
        result = self.face_recognizer.recognize(face_image)
        if result:
            messagebox.showinfo("Recognized", f"Face recognized as: {result['name']}.")
            logger.info(f"Face recognized as {result['name']}.")
        else:
            messagebox.showinfo("Not Recognized", "Face not recognized. Please try training first.")
            logger.info("Face not recognized.")

    def quit(self):
        """Release camera resources and exit the application."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = FaceTestUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
