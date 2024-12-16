import gc
import gc
import json
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights

from face_frontalization import frontalize, facial_feature_detector, camera_calibration
from src.configuration import IMAGES_DIR, UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME
from src.datasets.datasets import create_train_dataloader, RAFD, prepare_dataframe
from src.face_detector.face_detector import HaarCascadeDetector, DnnDetector
from src.parsing_data import get_images_data
from src.plot import insert_data_into_json, plot_emotion_accuracies, plot_heatmaps


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


def split_indexes_randomly(indexes, ratio=0.9, seed=42):
    """
    Splits a list of indexes into two random subsets: one with a specified ratio and the other with the complement.
    """
    # Shuffle the indexes to ensure randomness
    if seed is not None:
        random.seed(seed)  # Set the random seed for reproducibility
    random.shuffle(indexes)

    # Determine the split point
    split_point = int(ratio * len(indexes))

    # Split into two subsets
    first_subset = indexes[:split_point]
    second_subset = indexes[split_point:]

    return first_subset, second_subset


def test_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(weights=None)
    num_classes = 8  # Replace with the number of emotion classes
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),  # First new fully connected layer
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),  # Second new fully connected layer
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),  # Output layer
        nn.Softmax(dim=1)  # For multi-class classification
    )
    model.load_state_dict(torch.load(f"emotion_classification_model_{model_name}.pth"))
    model = model.to(device)
    model.eval()

    # Store detailed classification results
    overall_eval_accuracies = defaultdict(lambda: defaultdict(int))  # Nested dictionary

    for chunk_idx, chunk in enumerate(pd.read_csv("data_unpacked/rafd.csv", chunksize=536), 1):
        dataset = RAFD(prepare_dataframe(chunk))
        print("Processing chunk {}/15".format(chunk_idx))
        _, eval_dataset = train_test_split(dataset, test_size=0.10, random_state=42)

        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=False, num_workers=4)

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs = inputs.permute(0, 3, 1, 2)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Update classification results
                for actual, pred in zip(labels, predicted):
                    actual = actual.item()
                    pred = pred.item()
                    overall_eval_accuracies[actual][pred] += 1

        del eval_loader, eval_dataset, dataset, chunk
        torch.cuda.empty_cache()

    # Save the results in a JSON file
    with open("validation_results.json", 'r') as fh:
        data = json.load(fh)

    data.append({
        "model_name": model_name.capitalize(),
        "results": {k: dict(v) for k, v in overall_eval_accuracies.items()}  # Convert defaultdict to dict
    })

    with open("validation_results.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")

    # model = EmotionCNN(num_classes=7).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Freeze the feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_classes = 8  # Replace with the number of emotion classes
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),  # First new fully connected layer
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),  # Second new fully connected layer
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),  # Output layer
        nn.Softmax(dim=1)  # For multi-class classification
    )
    # model.load_state_dict(torch.load("emotion_classification_model.pth"))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    # Transformations for input data
    # transform = transforms.Compose([
    #     # transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor()
    # ])

    # Prepare datasets and loaders
    # train_dataset = FER2013("data_unpacked", mode='train', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # eval_dataset = FER2013("data_unpacked", mode='val', transform=transform)
    # eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    num_epochs = 20
    num_chunks = 15
    overall_train_accuracies = {chunk_id: [] for chunk_id in range(num_chunks)}
    overall_eval_accuracies = {chunk_id: [] for chunk_id in range(num_chunks)}

    start_time = time.time()
    for chunk_idx, chunk in enumerate(pd.read_csv("data_unpacked/rafd.csv", chunksize=536), 1):
        dataset = RAFD(prepare_dataframe(chunk))
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.10, random_state=42)
        # skip_for_eval, skip_for_train = split_indexes_randomly(list(range(len(os.listdir("data_unpacked/rafd/")))))
        # train_dataset = RAFD_DYNAMIC(skip_for_eval)
        # eval_dataset = RAFD_DYNAMIC(skip_for_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=False, num_workers=4)

        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=False, num_workers=4)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_correct_per_class = np.zeros(num_classes)
            train_total_per_class = np.zeros(num_classes)

            for inputs, labels in train_loader:
                inputs = inputs.permute(0, 3, 1, 2)  # From [batch_size, height, width, channels] to [batch_size, channels, height, width]

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)

                for i in range(num_classes):
                    train_correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                    train_total_per_class[i] += (labels == i).sum().item()

            accuracies = []
            for i in range(num_classes):
                accuracies.append(
                    100 * train_correct_per_class[i] / train_total_per_class[i] if train_total_per_class[i] > 0 else 0)
            overall_train_accuracies[chunk_idx - 1].append(accuracies) 

            print(
                f"Chunk: {chunk_idx}/{num_chunks}, Epoch [{epoch + 1}/{num_epochs}], Train Accuracies Per Class: {overall_train_accuracies[chunk_idx - 1][epoch]}")

            # Evaluation
            model.eval()
            eval_correct_per_class = np.zeros(num_classes)
            eval_total_per_class = np.zeros(num_classes)

            with torch.no_grad():
                for inputs, labels in eval_loader:
                    inputs = inputs.permute(0, 3, 1, 2)
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    for i in range(num_classes):
                        eval_correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                        eval_total_per_class[i] += (labels == i).sum().item()

                # Calculate mean accuracies for this epoch
            accuracies = []
            for i in range(num_classes):
                accuracies.append(
                    100 * eval_correct_per_class[i] / eval_total_per_class[i] if eval_total_per_class[i] > 0 else 0)
            overall_eval_accuracies[chunk_idx - 1].append(accuracies)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Eval Accuracies Per Class: {overall_eval_accuracies[chunk_idx - 1][epoch]}")
        train_loader = None
        eval_loader = None
        train_loader = None
        eval_loader = None
        dataset = None
        chunk = None
        del train_loader, eval_loader, train_dataset, eval_dataset, dataset, chunk
        gc.collect()
        torch.cuda.empty_cache()

    time_of_training = time.time() - start_time
    torch.save(model.state_dict(), "emotion_classification_model_raw_image_no_norm.pth")
    insert_data_into_json("RAFD", "Raw_image_no_norm", num_epochs, overall_train_accuracies, overall_eval_accuracies,
                          time_of_training)


def print_names(dataset_name):
    with open("models_training_results.json", "r") as json_file:
        data = json.load(json_file)
    names = [x["name"] for x in data if x["dataset_name"] == dataset_name]
    return set(names)


def print_accuracy_per_emotion(model_name):
    emotion_map = {
        "happy": 0,
        "angry": 1,
        "sad": 2,
        "contemptuous": 3,
        "disgusted": 4,
        "neutral": 5,
        "fearful": 6,
        "surprised": 7
    }

    with open("validation_results.json", "r") as json_file:
        data = json.load(json_file)

    found = None
    for d in data:
        if d["model_name"] == model_name:
            found = d
    if found is None:
        return

    for idx, emotion in enumerate(emotion_map.keys(), 0):
        print(f'{emotion}: {found["results"][str(idx)][0]/found["results"][str(idx)][1]*100:.2f}%')


def main():
    # train_model()
    # for model_name in ["raw_image", "crop_eyes", "crop_face", "frontalized", "raw_image_hist_eq"]:
    test_model("raw_image_no_norm")
    # print_accuracy_per_emotion("crop_face")
    # plot_emotion_accuracies("RAFD", 20, "Frontalized", 8, False, True)
    # plot_heatmaps("Raw_image")
    # plot_emotion_first_last_chunks("RAFD", 20, "Raw_image", 8, False, False)
    # plot_single_model("RAFD", 20, "Raw_image_hist_eq")
    # plot_all_models("RAFD", 20, name_prefixes=[], exact_names=[], filters=[])#, exact_names=["Raw_image", "Cropped_eyes_24_48", "Cropped_eyes_24_48_rotated", "Frontalized"])
    # print(print_names("FER"))


if __name__ == "__main__":
    main()
    pass
