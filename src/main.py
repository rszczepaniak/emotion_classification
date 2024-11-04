import os

from PIL import ImageShow
from transformers import DetrForObjectDetection, DetrImageProcessor

from ArgumentParser import ArgumentParser
from configuration import FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME, UNPACKED_DATA_DIR, FACE_DETECTION_MODEL_NAME, \
    IMAGES_DIR
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

    # predictor = dlib.shape_predictor(os.path.join(UNPACKED_DATA_DIR, SHAPE_PREDICTOR_NAME))
    for x, image in enumerate(found_faces):
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        # current_img_idx = get_current_image_index()
        # image.save(os.path.join(IMAGES_DIR, f"face_{datetime.now().strftime(DATE_FORMAT)}_{current_img_idx + 1}.png"))
        image.show()


class DisplayViewer(ImageShow.Viewer):
    def get_command(self, file, **options):
        return f"display {file}"


if __name__ == "__main__":
    ImageShow.register(DisplayViewer())

    parser = ArgumentParser()
    arguments = parser.parse()
    main(arguments)
