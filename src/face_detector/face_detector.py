import os

import cv2
import numpy as np

from src.utils import normalization


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
        self.prototxt = "deploy.prototxt.txt"
        self.model_weights = "res10_300x300_ssd_iter_140000.caffemodel"

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

    def detect_faces(self, image):
        """
        Preprocess the input image, run face detection, and display detected faces.

        Args:
            image: Input image as a NumPy array.

        Returns:
            List of detected faces in (x, y, w, h) format.
        """
        # Store original dimensions
        original_h, original_w = image.shape[:2]

        # Ensure the image has 3 channels
        if image.shape[-1] != 3:  # If not already 3 channels
            image = cv2.merge([image, image, image])  # Stack grayscale into 3 channels

        # sharp_image = self.sharpen_image(image)
        # Resize image to 300x300 for the detector
        resized_image = cv2.resize(image, (300, 300))

        cv2.imshow("resized_image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("norm", normalization(resized_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Preprocess the image: create a blob
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Perform detection
        self.detector.setInput(blob)
        detections = self.detector.forward()

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

            # Ensure the bounding box is within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)

            # Crop the face region from the original image
            if x2 > x1 and y2 > y1:  # Ensure valid bounding box dimensions
                cropped_face = image[y1:y2, x1:x2]
                cropped_faces.append(cropped_face)

        return cropped_faces
