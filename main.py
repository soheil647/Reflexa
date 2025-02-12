# main.py
import time
from modules.camera.camera import Camera
from modules.face.detection import FaceDetector
from modules.face.recognition import FaceRecognizer
from modules.face.training import FaceTrainer
from modules.wellness.health_metrics import HealthMetricsAnalyzer
from modules.llm.interaction import LLMInteraction
from modules.voice.wakeword import WakeWordDetector
from modules.voice.speech_recognition import SpeechRecognizer
from modules.voice.text_to_speech import TextToSpeech
from modules.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Initialize modules
    camera = Camera()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    face_trainer = FaceTrainer()
    wellness_analyzer = HealthMetricsAnalyzer()
    llm = LLMInteraction()
    wakeword_detector = WakeWordDetector()
    speech_recog = SpeechRecognizer()
    tts = TextToSpeech()

    # Start camera
    cap = camera.start()
    logger.info("Reflexa started. Waiting for wake word...")

    while True:
        # Check for wake word
        if wakeword_detector.detect():
            logger.info("Wake word detected!")
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                continue

            # Detect faces in the frame
            faces = face_detector.detect_faces(frame)
            if not faces:
                tts.speak("No face detected. Please stand in front of me.")
                continue

            # Process each detected face
            for face_location in faces:
                x1, y1, x2, y2 = face_location
                face_image = frame[y1:y2, x1:x2]
                user_profile = face_recognizer.recognize(face_image)
                if user_profile is None:
                    tts.speak("I don't recognize you. Let's create your profile. Please say your name.")
                    name = speech_recog.listen()
                    if name:
                        face_trainer.register_new_face(name, face_image)
                        user_profile = {"name": name}
                        tts.speak(f"Profile created for {name}.")
                    else:
                        tts.speak("Sorry, I didn't catch that.")
                        continue

                # Analyze wellness metrics on the frame
                metrics = wellness_analyzer.analyze(frame)
                # Get personalized recommendation from the LLM
                message = llm.get_recommendation(user_profile, metrics)
                tts.speak(message)

            time.sleep(2)  # Short pause after processing

        time.sleep(0.1)

    cap.release()

if __name__ == '__main__':
    main()
