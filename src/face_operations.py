import os

import cv2
import dlib
import numpy as np
import torch
from PIL import Image


class FaceDetector:
    def __init__(self, model, processor, threshold=0.9, padding=10):
        self.model = model
        self.processor = processor
        self.threshold = threshold
        self.padding = padding

    def same_detection(self, current_box, found_boxes):
        for already_found_box in found_boxes:
            if sum([x - y for x, y in zip(current_box, already_found_box)]) < 20:
                return True
        return False

    def detect_faces(self, image):
        # Convert the image from BGR to RGB (as Hugging Face models expect RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess the image for the model
        inputs = self.processor(images=image_rgb, return_tensors="pt", do_rescale=False)

        # Run inference
        outputs = self.model(**inputs)

        # Post-process the outputs
        target_sizes = torch.tensor([image_rgb.shape[:2]])  # Height, Width in RGB order
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        # List to hold cropped face images
        face_images = []
        found_faces = []
        for score, box in zip(results["scores"], results["boxes"]):
            if score > self.threshold and not self.same_detection(box.tolist(), found_faces):
                found_faces.append(box.tolist())
                # Convert box coordinates to integers
                box = [int(coord) for coord in box.tolist()]
                x_min, y_min, x_max, y_max = box

                # Apply padding and ensure coordinates stay within image bounds
                x_min_padded = max(x_min - self.padding, 0)
                y_min_padded = max(y_min - self.padding, 0)
                x_max_padded = min(x_max + self.padding, image.shape[1])  # Width
                y_max_padded = min(y_max + self.padding, image.shape[0])  # Height

                # Crop the face from the original BGR image
                face_image = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
                face_images.append(face_image)
                print(
                    f"Detected face with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

        if len(face_images) == 0:
            print("No faces detected.")

        return face_images

    @staticmethod
    def face_align(pil_image, predictor):
        # Convert the PIL image to an OpenCV image (numpy array)
        open_cv_image = np.array(pil_image)

        # Convert RGB to BGR (OpenCV uses BGR)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Detect the face region using dlib's rectangle
        face_rect = dlib.rectangle(0, 0, pil_image.width, pil_image.height)

        # Get the landmarks/parts for the face
        landmarks = predictor(gray, face_rect)

        # Get the coordinates of the eyes and nose
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Calculate the angle to rotate the image
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

        # Get the center of the face for rotation
        eye_center = tuple(np.mean([left_eye, right_eye], axis=0).astype(float))

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

        # Rotate the image
        rotated_cv_image = cv2.warpAffine(open_cv_image, M, (open_cv_image.shape[1], open_cv_image.shape[0]),
                                          flags=cv2.INTER_CUBIC)

        # Get the new position of the nose after rotation
        nose_center = np.dot(M, np.array([nose_tip[0], nose_tip[1], 1])).astype(int)

        # Calculate the translation to move the nose to the center of the image
        image_center = (rotated_cv_image.shape[1] // 2, rotated_cv_image.shape[0] // 2)
        translation_x = image_center[0] - nose_center[0]
        translation_y = image_center[1] - nose_center[1]

        # Apply the translation
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        aligned_cv_image = cv2.warpAffine(rotated_cv_image, translation_matrix,
                                          (rotated_cv_image.shape[1], rotated_cv_image.shape[0]), flags=cv2.INTER_CUBIC)

        # Convert the aligned image back to PIL format
        aligned_pil_image = Image.fromarray(cv2.cvtColor(aligned_cv_image, cv2.COLOR_BGR2RGB))

        return aligned_pil_image
