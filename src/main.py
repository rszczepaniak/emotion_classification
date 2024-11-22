import os

import cv2
import numpy as np
import torch
from scipy import io
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, RandomHorizontalFlip

from face_frontalization import frontalize, facial_feature_detector, camera_calibration
from src.ArgumentParser import ArgumentParser
from src.configuration import IMAGES_DIR, UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME
from src.datasets.fer_dataset import create_train_dataloader, FER2013
from src.face_detector.face_detector import HaarCascadeDetector, DnnDetector
from src.model.cnn import EmotionCNN
from src.parsing_data import get_images_data
from src.plot import insert_data_into_json, plot_all_models


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
            found_faces.extend(img)  # [y:y + h, x:x + w]
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


def apply_black_mask(img, landmarks, expansion_factor=0.1):
    """
    Create a mask for the face based on landmarks, enlarge the area, and black out everything else.
    """
    # Create a black mask of the same size as the image
    mask = np.zeros_like(img)

    # Convert landmarks to a NumPy array and ensure integer type
    landmarks = np.array(landmarks, dtype=np.int32)

    # Calculate the center of the landmarks
    center = np.mean(landmarks, axis=0)

    # Enlarge each landmark point away from the center
    enlarged_landmarks = (landmarks - center) * expansion_factor + center
    enlarged_landmarks = enlarged_landmarks.astype(np.int32)

    # Create a convex hull for the enlarged landmarks
    face_region = cv2.convexHull(enlarged_landmarks.reshape(-1, 1, 2))

    # Fill the face region with white on the mask
    cv2.fillConvexPoly(mask, face_region, (255, 255, 255))

    # Apply the mask to the image
    face_only = cv2.bitwise_and(img, mask)
    return face_only


class NumpyToTensorTransform:
    def __call__(self, image):
        # Convert the NumPy array to a tensor and scale to range [0, 1]
        tensor_image = torch.tensor(image, dtype=torch.float32)
        # Normalize if the data is in range [0, 255]
        if tensor_image.max() > 1:
            tensor_image = tensor_image / 255.0
        return tensor_image


def frontalize_faces(args):
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


def display_tensor_with_opencv(tensor_image):
    """
    Display a PyTorch tensor image using OpenCV.
    Args:
        tensor_image (torch.Tensor): Tensor image of shape (C, H, W), normalized in [0, 1].
    """
    # Convert to NumPy array
    image_np = tensor_image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    # Scale to 0–255 if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("Image", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")

    # Ensure model and data are on the same device
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Transformations for input data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Prepare datasets and loaders
    train_dataset = FER2013("data_unpacked", mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    eval_dataset = FER2013("data_unpacked", mode='val', transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_epochs = 100
    train_losses, train_accuracies, eval_accuracies = [], [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_predictions, total_predictions = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            running_loss += loss.item()

        # Training metrics
        epoch_accuracy = 100 * correct_predictions / total_predictions
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(epoch_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Evaluation
        model.eval()
        eval_correct_predictions, eval_total_predictions = 0, 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct_predictions += (predicted == labels).sum().item()
                eval_total_predictions += labels.size(0)

        # Evaluation metrics
        eval_accuracy = 100 * eval_correct_predictions / eval_total_predictions
        eval_accuracies.append(eval_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Evaluation Accuracy: {eval_accuracy:.2f}%')

    insert_data_into_json("FER", "Cropped_eyes_24_48_rotated", num_epochs, train_accuracies, eval_accuracies)


def print_plots(dataset_name, num_epochs, name_prefix):
    plot_all_models(dataset_name, num_epochs, name_prefix)


def main(args):
    # train_model(args)
    print_plots("FER", 100, "Cropped_eyes")

if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse()
    main(arguments)
    pass
