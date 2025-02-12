"""
tests/test_voice_module.py

This test script demonstrates the advanced voice module:
  - It continuously listens for a wake word using a free, continuous ASR approach with fuzzy matching.
  - Once the wake word is detected, it activates the voice interface.
  - It records user speech for up to 10 seconds and transcribes it using a Whisper model.
  - The recognized text is then translated using googletrans and read back using AI TTS.
  - Then it waits 10 seconds before reactivating the wake word detection.

Dependencies:
  - transformers (pip install transformers)
  - sounddevice (pip install sounddevice)
  - numpy (pip install numpy)
  - googletrans==4.0.0-rc1 (pip install googletrans==4.0.0-rc1)
  - rapidfuzz (pip install rapidfuzz)
"""

import time
import asyncio
from googletrans import Translator
from modules.voice.wakeword import WakeWordDetector
from modules.voice.speech_recognition import SpeechRecognizer
from modules.voice.text_to_speech import TextToSpeech
from modules.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Initialize the free wake word detector with fuzzy matching.
    wakeword_detector = WakeWordDetector(wake_word="hey anita", threshold=80)
    # Separate instance for extended speech capture.
    speech_recognizer = SpeechRecognizer()
    tts = TextToSpeech()  # Now uses the Hugging Face TTS pipeline.
    translator = Translator()

    logger.info("Voice Module Test started. Always listening for wake word...")

    while True:
        logger.info("Waiting for wake word...")
        if wakeword_detector.detect(timeout=3600):
            logger.info("Wake word detected. Activating voice interface.")
            tts.speak("I am awake. Please speak now.")
            recognized_text = speech_recognizer.listen(duration=10)
            if recognized_text:
                logger.info(f"Recognized speech: {recognized_text}")
                # Await the translator if necessary (depending on your googletrans version).
                translated_obj = asyncio.run(translator.translate(recognized_text, dest='en'))
                translated_text = translated_obj.text
                logger.info(f"Translated text: {translated_text}")
                tts.speak(f"You said: {translated_text}")
            else:
                logger.info("No speech detected during the active period.")
                tts.speak("I did not hear any speech.")
            logger.info("Sleeping for 10 seconds before reactivating wake word detection.")
            tts.speak("Sleeping for ten seconds.")
            time.sleep(10)
        else:
            continue

if __name__ == "__main__":
    main()
