# config.py
# Configuration file for Reflexa

# Camera settings
CAMERA_ID = 0

# Face detection
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"  # Uses OpenCV’s built-in cascades

# Directories
FACES_DIR = "data/faces/"
PROFILES_DIR = "data/profiles/"
LOGS_DIR = "data/logs/"

# LLM settings (replace YOUR_API_KEY with your actual key)
LLM_API_KEY = "YOUR_API_KEY"
LLM_ENGINE = "text-davinci-003"

# Wake word settings
WAKE_WORD = "hey reflexa"
