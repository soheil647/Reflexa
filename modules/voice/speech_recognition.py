# modules/voice/speech_recognition.py
import sounddevice as sd
import numpy as np
from transformers import pipeline
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class SpeechRecognizer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Use a lightweight Whisper model from Hugging Face.
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=-1  # Use CPU; change this if you have GPU available.
        )

    def listen(self, duration=10):
        """
        Records audio for `duration` seconds from the default microphone,
        then transcribes it using the Hugging Face ASR pipeline.
        Returns the recognized text, or None if nothing is recognized.
        """
        logger.info(f"Recording audio for {duration} seconds...")
        try:
            audio = sd.rec(int(duration * self.sample_rate),
                           samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished.
            # Squeeze the array to shape [num_samples]
            audio = np.squeeze(audio)
            logger.info("Audio recording complete. Processing ASR...")
            result = self.asr_pipeline(audio)
            text = result.get("text", "")
            if text.strip():
                logger.info(f"Recognized speech: {text}")
                return text
            else:
                logger.info("No speech recognized.")
                return None
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return None
