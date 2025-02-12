# modules/voice/wakeword.py
import time
from rapidfuzz import fuzz
from modules.voice.speech_recognition import SpeechRecognizer
from modules.utils.logger import get_logger
from config import WAKE_WORD  # e.g. "hey pasta"

logger = get_logger(__name__)

class WakeWordDetector:
    def __init__(self, wake_word=WAKE_WORD, sample_rate=16000, chunk_duration=2, threshold=80):
        """
        wake_word: The desired wake word (e.g., "hey pasta").
        sample_rate: Audio sample rate in Hz.
        chunk_duration: Duration (in seconds) of each audio chunk to analyze.
        threshold: The fuzzy matching score threshold (0-100) to trigger detection.
        """
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.threshold = threshold
        self.recognizer = SpeechRecognizer(sample_rate=sample_rate)

    def detect(self, timeout=None):
        """
        Continuously records audio in short chunks and transcribes each chunk.
        Uses fuzzy matching to compare the transcription to the wake word.
        Returns True as soon as the fuzzy match score is above the threshold.
        If a timeout (in seconds) is provided, returns False after that period.
        """
        logger.info("Starting continuous wake word detection using free ASR with fuzzy matching.")
        start_time = time.time()
        while True:
            # If timeout is set and exceeded, return False.
            if timeout and (time.time() - start_time) > timeout:
                logger.info("Wake word detection timeout reached.")
                return False

            text = self.recognizer.listen(duration=self.chunk_duration)
            if text:
                text_lower = text.lower()
                # Compute a fuzzy matching score using partial_ratio.
                score = fuzz.partial_ratio(text_lower, self.wake_word)
                logger.info(f"Chunk transcription: '{text_lower}' | Wake word: '{self.wake_word}' | Score: {score}")
                if score >= self.threshold:
                    logger.info("Wake word detected via fuzzy matching.")
                    return True
            # Short sleep to avoid a tight loop.
            time.sleep(0.1)
