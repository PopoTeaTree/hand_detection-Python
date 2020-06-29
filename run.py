import csv
import cv2
import numpy as np
import tensorflow as tf
from src.hand_tracker import HandTracker

PALM_MODEL_PATH = "models/palm_detection-mediapipe.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark-mediapipe.tflite"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

# interp_palm = tf.lite.Interpreter(PALM_MODEL_PATH)
# interp_palm.allocate_tensors()
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)