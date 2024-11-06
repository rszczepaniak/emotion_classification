import os

import cv2
import dlib
import numpy as np
import torch
from PIL import Image


def same_detection(current_box, found_boxes):
    for already_found_box in found_boxes:
        if sum([x - y for x, y in zip(current_box, already_found_box)]) < 20:
            return True
    return False


def detect_faces(image_path, model, processor, threshold=0.9, padding=10):
    """
    Detect faces in an image using the MTCNN model from Hugging Face and return a list of cropped face images.

    Parameters:
        image_path (str): Path to the input image.
        model: Pre-trained MTCNN model.
        processor: Pre-trained MTCNN processor.
        threshold (float): Confidence threshold for face detection.
        padding (int): Padding size for face detection.

    Returns:
        List[Image]: List of cropped images containing only the detected faces.
    """
    # Load the image
    if not os.path.exists(image_path):
        print(f"No image found: {image_path}")
        return []

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    outputs = model(**inputs)

    # Post-process the outputs
    target_sizes = torch.tensor([image.size[::-1]])  # Height, Width
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    # List to hold cropped face images
    face_images = []
    found_faces = []
    for score, box in zip(results["scores"], results["boxes"]):
        if score > threshold and not same_detection(box.tolist(), found_faces):
            found_faces.append(box.tolist())
            # Convert box coordinates to integers
            box = [int(coord) for coord in box.tolist()]
            x_min, y_min, x_max, y_max = box

            x_min_padded = max(x_min - padding, 0)
            y_min_padded = max(y_min - padding, 0)
            x_max_padded = min(x_max + padding, image.width)
            y_max_padded = min(y_max + padding, image.height)

            face_image = image.crop((x_min_padded, y_min_padded, x_max_padded, y_max_padded))
            face_images.append(face_image)
            print(
                f"Detected face with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    if len(face_images) == 0:
        print("No faces detected.")

    return face_images


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
