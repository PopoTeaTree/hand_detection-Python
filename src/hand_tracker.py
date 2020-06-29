import csv
import cv2
import numpy as np
import tensorflow as tf

class HandTracker():
    def __init__(self, palm_model, joint_model,
                box_enlarge=1.5, box_shift=0.2):
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        # self.interp_joint = tf.lite.Interpreter(joint_model)
        # self.interp_joint.allocate_tensors()