# tests/test_voice.py
from modules.voice.speech_recognition import SpeechRecognizer
from modules.voice.text_to_speech import TextToSpeech

def test_speech_recognition():
    sr_instance = SpeechRecognizer()
    text = sr_instance.listen()
    assert text is not None, "Speech recognition did not capture any text"

def test_text_to_speech():
    tts = TextToSpeech()
    try:
        tts.speak("This is a test.")
    except Exception as e:
        assert False, f"Text to Speech failed: {e}"
