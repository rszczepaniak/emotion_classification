import os

import dlib

from parsing_data import get_images_data
from face_operations import detect_faces, face_align
from configuration import FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME, UNPACKED_DATA_DIR, FACE_DETECTION_MODEL_NAME, SHAPE_PREDICTOR_NAME

from transformers import DetrForObjectDetection, DetrImageProcessor


if __name__ == "__main__":
    parsed_images = get_images_data(os.path.join(UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME.format("01")))
    model = DetrForObjectDetection.from_pretrained(FACE_DETECTION_MODEL_NAME)
    processor = DetrImageProcessor.from_pretrained(FACE_DETECTION_MODEL_NAME)

    num_images = 0
    found_faces = []
    for image in parsed_images:
        if not os.path.exists(image.get("path_to_image")):
            continue
        if num_images >= 3:
            break
        num_images += 1
        found_faces.extend(detect_faces(image.get("path_to_image"), model, processor))

    predictor = dlib.shape_predictor(os.path.join(UNPACKED_DATA_DIR, SHAPE_PREDICTOR_NAME))
    for x, image in enumerate(found_faces):
        image.show()
        aligned = face_align(image, predictor)
        aligned.show()
