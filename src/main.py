import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from transformers import DetrForObjectDetection, DetrImageProcessor

from ArgumentParser import ArgumentParser
from configuration import FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME, UNPACKED_DATA_DIR, FACE_DETECTION_MODEL_NAME, \
    IMAGES_DIR
from face_frontalization import frontalize, facial_feature_detector, camera_calibration
from face_operations import detect_faces
from parsing_data import get_images_data


def get_current_image_index() -> int:
    if os.listdir(IMAGES_DIR):
        return max(int(face.replace(".png", "").split("_")[-1]) for face in os.listdir(IMAGES_DIR))
    return 0


def main(args):
    parsed_images = get_images_data(
        os.path.join(UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME.format("01")))
    model = DetrForObjectDetection.from_pretrained(FACE_DETECTION_MODEL_NAME)
    processor = DetrImageProcessor.from_pretrained(FACE_DETECTION_MODEL_NAME)

    num_images = 0
    found_faces = []
    for image in parsed_images:
        if not os.path.exists(image.get("path_to_image")):
            continue
        if num_images >= int(args.num_pictures_to_process):
            break
        num_images += 1
        found_faces.extend(detect_faces(image.get("path_to_image"), model, processor))

    for x, image in enumerate(found_faces):
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        model3D = frontalize.ThreeD_Model("face_frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV BGR format

        plt.figure()
        plt.imshow(img[:, :, ::-1])  # Convert BGR to RGB for displaying with matplotlib
        plt.axis('off')
        plt.savefig(f"images/original_face_{x}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        lmarks = facial_feature_detector.get_landmarks(img)
        if lmarks is None:
            continue
        proj_matrix, camera_matrix, rmat, tvec = camera_calibration.estimate_camera(model3D, lmarks[0])
        eyemask = np.asarray(io.loadmat('face_frontalization/frontalization_models/eyemask.mat')['eyemask'])
        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

        plt.figure()
        # plt.imshow(frontal_raw[:, :, ::-1])
        plt.imshow(frontal_sym[:, :, ::-1])
        plt.axis('off')
        plt.savefig(f"images/frontalized_face_{x}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse()
    main(arguments)
