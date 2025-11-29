"""
Configuration settings for Facial Emotion Recognition System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = BASE_DIR / "training"

# Model settings
MODEL_PATH = MODELS_DIR / "emotion_model.keras"
MODEL_INPUT_SIZE = (48, 48)  # FER2013 standard size
NUM_CLASSES = 7

# Emotion labels (FER2013 standard)
EMOTIONS = {
    0: "Angry",
    1: "Disgust", 
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Emotion colors for visualization (BGR format)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),      # Red
    "Disgust": (0, 128, 0),    # Green
    "Fear": (128, 0, 128),     # Purple
    "Happy": (0, 255, 255),    # Yellow
    "Sad": (255, 0, 0),        # Blue
    "Surprise": (0, 165, 255), # Orange
    "Neutral": (128, 128, 128) # Gray
}

# Face detection settings
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# Display settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_DISPLAY = True
CONFIDENCE_THRESHOLD = 0.5

# Training settings
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation settings
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "zoom_range": 0.1
}

# Ensure directories exist
def create_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [MODELS_DIR, DATA_DIR, TRAINING_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
