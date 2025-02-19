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
            # Attempt to synthesize speech
            wav = self.tts.tts(text)
            logger.debug(f"Synthesized output type: {type(wav)}")

            # Log available attributes if possible
            if hasattr(wav, 'shape'):
                logger.debug(f"Waveform shape: {wav.shape}")
            else:
                logger.debug("No shape attribute available for waveform.")

            # Check if the object supports len()
            if hasattr(wav, '__len__'):
                waveform_length = len(wav)
                logger.info(f"Generated waveform length: {waveform_length}")
            elif hasattr(wav, 'shape'):
                waveform_length = wav.shape[0]
                logger.info(f"Generated waveform length (using shape): {waveform_length}")
            else:
                logger.info("Waveform length: unknown (object does not support len() and has no shape)")

            # Play the audio if possible
            sd.play(wav, self.sample_rate)
            sd.wait()  # Wait until playback is finished.
        except Exception as e:
            logger.error(f"Error during TTS synthesis/playback: {e}")
