import os
import time

import cv2
import mediapipe as mp
import numpy as np


# Abstract class / Interface
class FaceDetectorIface:
    def detect_faces(self, frame):
        raise NotImplementedError

class HaarCascadeDetector(FaceDetectorIface):
    def __init__(self, root=None):
        self.path = "haarcascade_frontalface_default.xml"
        if root:
            self.path = os.path.join(root, self.path)

        self.detector = cv2.CascadeClassifier(self.path)

    def detect_faces(self, frame):
        faces = self.detector.detectMultiScale(frame)
        return faces

class DnnDetector(FaceDetectorIface):
    """
        SSD (Single Shot Detectors) based face detection (ResNet-18 backbone(light feature extractor))
    """
    def __init__(self, root=None):
        self.prototxt = "src/face_detector/deploy.prototxt.txt"
        self.model_weights = "src/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

        if root:
            self.prototxt = os.path.join(root, self.prototxt)
            self.model_weights = os.path.join(root, self.model_weights)

        self.detector = cv2.dnn.readNetFromCaffe(prototxt=self.prototxt, caffeModel=self.model_weights)
        self.threshold = 0.1 # to remove weak detections

    @staticmethod
    def sharpen_image(image):
        """
        Apply a sharpening filter to the input image.

        Args:
            image: Input image (NumPy array).

        Returns:
            Sharpened image.
        """
        # Define a sharpening kernel
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])

        # Apply the filter to the image
        sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
        return sharpened_image

    def detect_faces(self, image, padding=0.0):  # 0.45
        """
        Preprocess the input image, run face detection, and adjust cropping so that eyes are centered.
        Eye positions are marked on the cropped face regions.

        Args:
            image: Input image as a NumPy array.
            padding: Fraction of the bounding box size to add as padding.

        Returns:
            List of cropped face regions with eyes centered.
        """
        # Store original dimensions
        original_h, original_w = image.shape[:2]

        # Ensure the image has 3 channels
        if image.shape[-1] != 3:  # If not already 3 channels
            image = cv2.merge([image, image, image])  # Stack grayscale into 3 channels

        # Resize image to 300x300 for the detector
        resized_image = cv2.resize(image, (300, 300))
        # Preprocess the image: create a blob
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0)) #(300, 300)


        # Perform detection
        self.detector.setInput(blob)
        # x = time.time()
        # print(x)
        detections = self.detector.forward()
        # print((time.time() - x) * 1000)
        cropped_faces = []
        for i in range(detections.shape[2]):
            # Extract confidence of detection
            confidence = detections[0, 0, i, 2]
            if confidence < self.threshold:
                continue

            # Extract bounding box, scale back to original image size
            box = detections[0, 0, i, 3:7] * np.array([original_w, original_h, original_w, original_h])
            box = box.astype("int")
            (x1, y1, x2, y2) = box

            # Use a face landmark detector (e.g., MediaPipe) to find eye centers
            face_region = image[y1:y2, x1:x2]
            # cropped_faces.append(face_region)

            landmarks = detect_landmarks_with_mediapipe(face_region)
            if landmarks is None:
                cropped_faces.append(face_region)
                break
            left_eye, right_eye = get_mediapipe_eye_centers(landmarks)

            # Adjust eye coordinates relative to the original image
            left_eye_x = int(left_eye[0] + x1)
            left_eye_y = int(left_eye[1] + y1)
            right_eye_x = int(right_eye[0] + x1)
            right_eye_y = int(right_eye[1] + y1)

            # Find the midpoint of the eyes
            eye_midpoint_x = int((left_eye_x + right_eye_x) // 2)
            eye_midpoint_y = int((left_eye_y + right_eye_y) // 2)
            # landmarks = detect_landmarks_with_mediapipe(face_region)
            # if landmarks is None:
            #     cropped_faces.append(face_region)
            #     break
            #
            # # Get nose landmark instead of eye landmarks
            # nose_center = get_nose_landmark(landmarks)
            #
            # # Adjust nose coordinates relative to the original image
            # nose_x = int(nose_center[0] + x1)
            # nose_y = int(nose_center[1] + y1)
            # Set the box height and width (square region)
            box_size = max(y2 - y1, x2 - x1)

            # Adjust the box to center the eyes
            new_x1 = max(0, int(eye_midpoint_x - box_size / 2))
            # new_x1 = max(0, int(nose_x - box_size / 2))
            new_x2 = min(original_w, int(eye_midpoint_x + box_size / 2))
            # new_x2 = min(original_w, int(nose_x + box_size / 2))
            new_y1 = max(0, int(eye_midpoint_y - box_size / 3))  # Keeping eyes in the upper third of the region
            # new_y1 = max(0, int(nose_y - box_size / 3))  # Keeping eyes in the upper third of the region
            new_y2 = min(original_h, int(new_y1 + box_size))

            # Apply padding
            pad_x = int((new_x2 - new_x1) * padding)
            pad_y = int((new_y2 - new_y1) * padding)
            new_x1 = max(0, new_x1 - pad_x)
            new_y1 = max(0, new_y1 - pad_y)
            new_x2 = min(original_w, new_x2 + pad_x)
            new_y2 = min(original_h, new_y2 + pad_y)

            # Crop the face region
            if new_x2 > new_x1 and new_y2 > new_y1:  # Ensure valid bounding box dimensions
                height = new_y2 - new_y1
                width = new_x2 - new_x1
                ratio = height / width
                new_width = width * ratio
                to_add = (new_width - width) // 2
                new_x1 = max(0, int(new_x1 - to_add))
                new_x2 = min(int(new_x2 + to_add), original_w)
                adjusted_face_region = image[new_y1:new_y2, new_x1:new_x2]
                cropped_faces.append(adjusted_face_region)

        return cropped_faces


def detect_landmarks_with_mediapipe(image):
    """
    Detect facial landmarks using Mediapipe.
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        return np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks.landmark], dtype=np.uint64)


def get_mediapipe_eye_centers(landmarks):
    """
    Returns the centers of the left and right eyes from the landmarks.
    """
    left_eye_landmarks = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]  # Left
    right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # right
    # Eye landmarks based on Mediapipe face mesh model
    right_eye = np.mean([landmarks[i] for i in right_eye_landmarks], axis=0).astype(np.uint64)
    left_eye = np.mean([landmarks[i] for i in left_eye_landmarks], axis=0).astype(np.uint64)
    return left_eye, right_eye


def get_nose_landmark(landmarks):
    return np.array(landmarks[3]).astype(np.uint64)


def display_landmarks(image, landmarks: np.array):
    """
    Displays the detected landmarks on the image.

    Args:
        image (numpy.ndarray): The original image.
        landmarks (list): List of tuples containing landmark coordinates (x, y).
    """
    if landmarks is None:
        print("No landmarks detected.")
        return

    # Make a copy of the image to draw landmarks on
    output_image = image.copy()

    # Draw each landmark as a small circle
    for (x, y) in landmarks:
        cv2.circle(output_image, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)

    # Display the image with landmarks
    cv2.imshow('Landmarks', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
