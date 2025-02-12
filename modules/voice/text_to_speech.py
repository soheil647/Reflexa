import sounddevice as sd
import numpy as np
from TTS.api import TTS
from modules.utils.logger import get_logger

logger = get_logger(__name__)

class TextToSpeech:
    def __init__(self):
        try:
            logger.info("Initializing Coqui TTS model (VITS, LJSpeech)...")
            # Using the Coqui TTS VITS model for English (LJSpeech)
            self.tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
            # Retrieve the sample rate from the model's synthesizer if available.
            self.sample_rate = self.tts.synthesizer.output_sample_rate
            logger.info(f"TTS model loaded successfully. Sample rate: {self.sample_rate}")
        except Exception as e:
            logger.error(f"Error initializing Coqui TTS: {e}")
            raise e

    def speak(self, text):
        logger.info(f"Speaking: {text}")
        try:
            # Synthesize speech; this returns a NumPy array of the waveform.
            wav = self.tts.tts(text)
            if wav is None or not isinstance(wav, np.ndarray):
                logger.error("No audio was synthesized.")
                return
            logger.info("Playing synthesized audio...")
            sd.play(wav, self.sample_rate)
            sd.wait()  # Wait until playback is finished.
        except Exception as e:
            logger.error(f"Error during TTS synthesis/playback: {e}")
