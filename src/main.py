import os

import cv2
import numpy as np
import torch
from scipy import io

from configuration import IMAGES_DIR, UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME
from face_frontalization import frontalize, facial_feature_detector, camera_calibration
from src.ArgumentParser import ArgumentParser
from src.datasets.fer_dataset import create_train_dataloader
from src.face_detector.face_detector import DnnDetector, HaarCascadeDetector
from src.parsing_data import get_images_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_current_image_index() -> int:
    if os.listdir(IMAGES_DIR):
        return max(int(face.replace(".png", "").split("_")[-1]) for face in os.listdir(IMAGES_DIR))
    return 0


def get_faces_fddb(args, face_detector):
    parsed_images = get_images_data(
        os.path.join(UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME.format("01")))

    found_faces = []
    for image in parsed_images:
        if not os.path.exists(image.get("path_to_image")):
            continue
        img = cv2.imread(image.get("path_to_image"))

        for x, y, w, h in face_detector.detect_faces(img):
            found_faces.extend(img) # [y:y + h, x:x + w]
            if len(found_faces) >= int(args.num_pictures_to_process):
                break
        if len(found_faces) >= int(args.num_pictures_to_process):
            break
    return found_faces


def get_faces_fer(args):
    found_faces = []
    dataloader = create_train_dataloader()
    for i in range(len(dataloader)):
        face, emotion = dataloader.dataset[i]
        found_faces.append(face)
        if len(found_faces) >= int(args.num_pictures_to_process):
            break
    return found_faces


def main(args):
    if args.haar:
        face_detector = HaarCascadeDetector("src/face_detector")
    else:
        face_detector = DnnDetector("src/face_detector")
    # found_faces = get_faces_fddb(args, face_detector)
    found_faces = get_faces_fer(args)

    # if not os.path.exists(IMAGES_DIR):
    #     os.makedirs(IMAGES_DIR)

    for x, img in enumerate(found_faces):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (160, 160))
        model3D = frontalize.ThreeD_Model("face_frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
        # img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV BGR format
        cv2.imshow("original", img)
        cv2.waitKey(0)

        lmarks = facial_feature_detector.get_landmarks(img)
        if lmarks is None:
            continue
        proj_matrix, camera_matrix, rmat, tvec = camera_calibration.estimate_camera(model3D, lmarks[0])
        eyemask = np.asarray(io.loadmat('face_frontalization/frontalization_models/eyemask.mat')['eyemask'])
        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

        cv2.imshow("Frontalized Raw", frontal_raw)
        cv2.waitKey(0)

        # Display the frontalized symmetric face image
        cv2.imshow("Frontalized Symmetric", frontal_sym)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.figure()
        # # plt.imshow(frontal_raw[:, :, ::-1])
        # plt.imshow(frontal_sym[:, :, ::-1])
        # plt.axis('off')
        # plt.savefig(f"images/frontalized_face_{x}.png", bbox_inches='tight', pad_inches=0)
        # plt.close()
    #
    # mini_xception = Mini_Xception().to(device)
    # mini_xception.eval()
    # face_alignment = FaceAlignment()
    # if args.haar:
    #     face_detector = HaarCascadeDetector("src/face_detector")
    # else:
    #     face_detector = DnnDetector("src/face_detector")
    #
    # while args.image:
    #     frame = cv2.imread(args.path)
    #
    #     if args.path:
    #         frame = cv2.resize(frame, (640, 480))
    #
    #     # faces
    #     faces = face_detector.detect_faces(frame)
    #
    #     for face in faces:
    #         (x, y, w, h) = face
    #
    #         # preprocessing
    #         # cv2.imshow("pre", frame)
    #         input_face = face_alignment.frontalize_face(face, frame)
    #         # cv2.imshow("post", input_face)
    #
    #         input_face = cv2.resize(input_face, (48, 48))
    #
    #         # input_face = histogram_equalization(input_face)
    #         cv2.imshow('input face', cv2.resize(input_face, (120, 120)))
    #         cv2.waitKey(0)  # Waits for a key press to close the window
    #         cv2.destroyAllWindows()
    #         break
    #         input_face = transforms.ToTensor()(input_face).to(device)
    #         input_face = torch.unsqueeze(input_face, 0)
    #
    #         with torch.no_grad():
    #             input_face = input_face.to(device)
    #             emotion = mini_xception(input_face)
    #             # print(f'\ntime={(time.time()-t) * 1000 } ms')
    #
    #             torch.set_printoptions(precision=6)
    #             softmax = torch.nn.Softmax()
    #             emotions_soft = softmax(emotion.squeeze()).reshape(-1, 1).cpu().detach().numpy()
    #             emotions_soft = np.round(emotions_soft, 3)
    #             for i, em in enumerate(emotions_soft):
    #                 em = round(em.item(), 3)
    #                 # print(f'{get_label_emotion(i)} : {em}')
    #
    #             emotion = torch.argmax(emotion)
    #             percentage = round(emotions_soft[emotion].item(), 2)
    #             emotion = emotion.squeeze().cpu().detach().item()
    #             emotion = get_label_emotion(emotion)
    #
    #             frame[y - 30:y, x:x + w] = (50, 50, 50)
    #             cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200))
    #             cv2.putText(frame, str(percentage), (x + w - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                         (200, 200, 0))
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #
    #     cv2.putText(frame, str("fps"), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    #     cv2.imshow("Image", frame)


if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse()
    main(arguments)
    pass
