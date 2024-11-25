import os

import dlib
import numpy as np


class LandmarksDetectorIface:
    def detect_landmarks(self, frame, rect):
        raise NotImplementedError

    def convert_to_numpy(self, landmarks):
        raise NotImplementedError


class dlibLandmarks(LandmarksDetectorIface):
    def __init__(self, root='data_unpacked'):
        self.path_5_landmarks = "shape_predictor_5_face_landmarks.dat"
        self.path_68_landmarks = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.shape_predictor(os.path.join(root, self.path_68_landmarks))

    def convert_to_numpy(self, landmarks):
        num_landmarks = 68
        coords = np.zeros((num_landmarks, 2), dtype=int)
        for i in range(num_landmarks):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    def detect_landmarks(self, frame, rect):
        # landmarks detection accept only dlib rectangles to operate on
        if type(rect) != dlib.rectangle:
            (x, y, w, h) = rect
            rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)

        # convert from dlib style to numpy style
        landmarks = self.detector(frame, rect)
        return self.convert_to_numpy(landmarks)
