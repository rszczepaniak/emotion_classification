import gc
import gc
import json
import os
import random
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights, resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights, vgg16, \
    InceptionOutputs
from tqdm import tqdm

from face_frontalization import frontalize, facial_feature_detector, camera_calibration
from src.configuration import IMAGES_DIR, UNPACKED_DATA_DIR, FDDB_IMAGE_DATASET, FDDB_DATASET_FILE_NAME
from src.datasets.datasets import create_train_dataloader, RAFD, prepare_dataframe, preprocess_image
from src.face_detector.face_detector import HaarCascadeDetector, DnnDetector
from src.model.feature_extraction import build_gabor_kernels, apply_gabor_filters, extract_lbp_features, \
    euclidean_distance, mahalanobis_distance, chi_square_distance
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


def test_model(model_name, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 8  # Replace with the number of emotion classes

    # Model selection
    if model_name == "vgg16":
        model = get_vgg16(num_classes=num_classes, device=device)
    elif model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, device=device)
    elif model_name == "inception_v3":
        model = get_inception_v3(num_classes=num_classes, device=device)
    else:
        print("Invalid model name")
        return

    # Load the model weights
    model.load_state_dict(torch.load(f"emotion_classification_{model_name}.pth"))
    model = model.to(device)
    model.eval()

    # Store detailed classification results
    overall_eval_accuracies = defaultdict(lambda: defaultdict(int))  # Nested dictionary

    # Process data in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv("data_unpacked/rafd_crop_face.csv", chunksize=536), 1):
        dataset = RAFD(prepare_dataframe(chunk, image_size))
        print(f"Processing chunk {chunk_idx}/15")
        _, eval_dataset = train_test_split(dataset, test_size=0.10, random_state=42)

        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=False, num_workers=4)

        with torch.no_grad():
            for inputs, labels in eval_loader:
                # Transform inputs from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                inputs = inputs.permute(0, 3, 1, 2)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # Handle InceptionV3 outputs
                if model_name == "inception_v3" and isinstance(outputs, InceptionOutputs):
                    outputs = outputs.logits  # Use primary logits

                _, predicted = torch.max(outputs, 1)

                # Update classification results
                for actual, pred in zip(labels, predicted):
                    actual = actual.item()
                    pred = pred.item()
                    overall_eval_accuracies[actual][pred] += 1

        # Clean up memory to avoid leaks
        del eval_loader, eval_dataset, dataset, chunk
        torch.cuda.empty_cache()

    # Save the results in a JSON file
    results_path = "validation_results.json"
    try:
        with open(results_path, 'r') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = []

    data.append({
        "model_name": model_name.capitalize(),
        "results": {k: dict(v) for k, v in overall_eval_accuracies.items()}  # Convert defaultdict to dict
    })

    with open(results_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_vgg16(num_classes=8, device=torch.device("cpu")):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
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
    model = model.to(device)
    return model


def get_resnet50(num_classes=8, device=torch.device("cpu")):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    return model


def get_inception_v3(num_classes=8, device=torch.device("cpu")):
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model.aux_logits = True
    model.AuxLogits.fc = nn.Sequential(
        nn.Linear(model.AuxLogits.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)
    return model


def train_model(model_name, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 8  # Replace with the number of emotion classes

    if model_name == "vgg16":
        model = get_vgg16(num_classes=num_classes, device=device)
        parameters = model.classifier.parameters()  # Use classifier parameters
    elif model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, device=device)
        parameters = model.fc.parameters()  # Use fc parameters
    elif model_name == "inception_v3":
        model = get_inception_v3(num_classes=num_classes, device=device)
        parameters = model.fc.parameters()  # Use fc parameters
    else:
        print("Invalid model name")
        return

        # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=0.0001)

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
    num_epochs = 50
    num_chunks = 15
    overall_train_accuracies = {chunk_id: [] for chunk_id in range(num_chunks)}
    overall_eval_accuracies = {chunk_id: [] for chunk_id in range(num_chunks)}

    start_time = time.time()
    for chunk_idx, chunk in enumerate(pd.read_csv("data_unpacked/rafd_crop_face.csv", chunksize=536), 1):
        dataset = RAFD(prepare_dataframe(chunk, image_size))
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
                # Transform inputs from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                inputs = inputs.permute(0, 3, 1, 2)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # Handle InceptionV3 outputs
                if model_name == "inception_v3" and isinstance(outputs, InceptionOutputs):
                    primary_output = outputs.logits  # Primary output
                    loss = criterion(primary_output, labels)

                    # Optional: Add loss from auxiliary logits
                    if model.aux_logits:
                        aux_loss = criterion(outputs.aux_logits, labels)
                        loss += 0.4 * aux_loss  # Weighted auxiliary loss
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Handle predictions for accuracy calculation
                if isinstance(outputs, InceptionOutputs):
                    outputs = outputs.logits  # Use primary logits

                _, predicted = torch.max(outputs, 1)

                for i in range(num_classes):
                    train_correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                    train_total_per_class[i] += (labels == i).sum().item()

            # Calculate accuracies for this epoch
            accuracies = [
                100 * train_correct_per_class[i] / train_total_per_class[i] if train_total_per_class[i] > 0 else 0
                for i in range(num_classes)
            ]
            overall_train_accuracies[chunk_idx - 1].append(accuracies)

            print(
                f"Chunk: {chunk_idx}/{num_chunks}, Epoch [{epoch + 1}/{num_epochs}], Train Accuracies Per Class: {overall_train_accuracies[chunk_idx - 1][epoch]}"
            )

            # Evaluation
            model.eval()
            eval_correct_per_class = np.zeros(num_classes)
            eval_total_per_class = np.zeros(num_classes)

            with torch.no_grad():
                for inputs, labels in eval_loader:
                    inputs = inputs.permute(0, 3, 1, 2)
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)

                    # Handle InceptionV3 outputs
                    if isinstance(outputs, InceptionOutputs):
                        outputs = outputs.logits  # Use primary logits

                    _, predicted = torch.max(outputs, 1)

                    for i in range(num_classes):
                        eval_correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                        eval_total_per_class[i] += (labels == i).sum().item()

            # Calculate accuracies for this epoch
            accuracies = [
                100 * eval_correct_per_class[i] / eval_total_per_class[i] if eval_total_per_class[i] > 0 else 0
                for i in range(num_classes)
            ]
            overall_eval_accuracies[chunk_idx - 1].append(accuracies)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Eval Accuracies Per Class: {overall_eval_accuracies[chunk_idx - 1][epoch]}"
            )
        train_loader = None
        eval_loader = None
        train_loader = None
        eval_loader = None
        dataset = None
        chunk = None
        del train_loader, eval_loader, train_dataset, eval_dataset, dataset, chunk
        gc.collect()
        print(f"End of batch, uncollected objects: {gc.garbage}")
        torch.cuda.empty_cache()

    time_of_training = time.time() - start_time
    torch.save(model.state_dict(), f"emotion_classification_{model_name}.pth")
    insert_data_into_json("RAFD", f"Crop_face_{model_name}", num_epochs, overall_train_accuracies,
                          overall_eval_accuracies,
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
            break
    if found is None:
        print(f"Model '{model_name}' not found.")
        return

    results = found["results"]
    total_accuracies = []

    # Calculate and print per-emotion accuracy
    print(f"Per-Emotion Accuracy for model '{model_name}':")
    for emotion, idx in emotion_map.items():
        actual_total = sum(results[str(idx)].values())  # Total occurrences of the current emotion
        correct_predictions = results[str(idx)].get(str(idx), 0)  # Correct predictions for this emotion

        if actual_total > 0:
            accuracy = correct_predictions / actual_total * 100
        else:
            accuracy = 0.0

        total_accuracies.append(accuracy)
        print(f'{emotion}: {accuracy:.2f}%')

    # Calculate and print mean accuracy
    mean_accuracy = np.mean(total_accuracies)
    std_deviation = np.std(total_accuracies)

    print(f'\nMean Accuracy: {mean_accuracy:.2f}%')
    print(f'Accuracy Variation (Standard Deviation): {std_deviation:.2f}%')


def train_svm(features_file, kernel='linear', C=1.0, gamma='scale'):
    """
    Train an SVM with specified hyperparameters and log the results.

    Parameters:
        features_file (str): Path to the file containing features and labels.
        kernel (str): Kernel type for SVM ('linear', 'rbf', 'poly', 'sigmoid').
        C (float): Regularization parameter.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

    Returns:
        None
    """
    # Load data
    data = np.load(features_file)
    x = data['x']
    y = data['y']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    # Load existing JSON file or initialize an empty list
    try:
        with open("svm_training.json", 'r') as json_file:
            results = json.load(json_file)
    except FileNotFoundError:
        results = []

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results.append({
        "features_file": features_file,
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON compatibility
    })

    # Save the updated results back to the JSON file
    with open("svm_training.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)


def train_mdc(features_file, distance_metric, centroid_method="mean", normalization=None, regularization=1e-5):
    # Load data
    data = np.load(features_file)
    x = data['x']
    y = data['y']

    # Normalize features if specified
    if normalization == "standard":
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    elif normalization == "minmax":
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

    # Remove zero-variance features
    x = x[:, np.var(x, axis=0) > 1e-6]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Compute centroids
    class_labels = np.unique(y_train)
    if centroid_method == "mean":
        centroids = {label: X_train[y_train == label].mean(axis=0) for label in class_labels}
    elif centroid_method == "median":
        centroids = {label: np.median(X_train[y_train == label], axis=0) for label in class_labels}
    else:
        raise ValueError("Unsupported centroid method. Choose 'mean' or 'median'.")

    # Compute covariance matrix for Mahalanobis distance
    covariance_matrix = None
    if distance_metric.__name__ == "mahalanobis_distance":
        covariance_matrix = np.cov(X_train, rowvar=False) + np.eye(X_train.shape[1]) * regularization

    # Predict labels for the test set
    y_pred = []
    for sample in X_test:
        distances = {label: distance_metric(sample, centroid, covariance_matrix) for label, centroid in
                     centroids.items()}
        y_pred.append(min(distances, key=distances.get))

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Log results to JSON
    result = {
        "features_file": features_file,
        "distance_metric": distance_metric.__name__,
        "centroid_method": centroid_method,
        "normalization": normalization,
        "regularization": regularization if covariance_matrix is not None else None,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }

    # Save results
    try:
        with open("mdc_training.json", "r") as json_file:
            results = json.load(json_file)
    except FileNotFoundError:
        results = []

    results.append(result)

    with open("mdc_training.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


def extract_features_chunk(chunk, method, kernels=None, radius=1, n_points=8):
    x_chunk = []
    y_chunk = []
    for _, row in chunk.iterrows():
        face = preprocess_image(row["pixels"])
        if method == "gabor":
            features = apply_gabor_filters(face, kernels)
        elif method == "lbp":
            features = extract_lbp_features(face, radius, n_points)
        else:
            raise ValueError("Unsupported method. Choose 'gabor' or 'lbp'.")
        x_chunk.append(features)
        y_chunk.append(row['emotion'])  # Use the emotion label
    return x_chunk, y_chunk


def extract_features_to_file(file_name, method="gabor", ksize=31, radius=1, n_points=8):
    chunk_size = 536
    file_path = "data_unpacked/rafd_crop_face.csv"

    # Create Gabor kernels if using Gabor method
    kernels = build_gabor_kernels(ksize=ksize) if method == "gabor" else None

    x_all = []
    y_all = []

    chunks = pd.read_csv(file_path, dtype={"emotion": "str", "pixels": "str"}, chunksize=chunk_size)
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for header row

    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        with Pool(processes=6) as pool:
            process_chunk = partial(
                extract_features_chunk, method=method, kernels=kernels, radius=radius, n_points=n_points
            )
            for x_chunk, y_chunk in pool.imap(process_chunk, chunks):
                x_all.extend(x_chunk)
                y_all.extend(y_chunk)
                pbar.update(len(x_chunk))

    # Save all rows to a single .npz file
    np.savez(file_name, x=np.array(x_all), y=np.array(y_all))


def test_svm_hiperparameters():
    features_files = [
        "gabor_64x64s_21.npz",
        "gabor_64x64s_31.npz",
        "gabor_64x64s_41.npz",
        "lbp_64x64s_1_8.npz",
        "lbp_64x64s_2_16.npz",
        "lbp_64x64s_4_32.npz",
        "lbp_64x64s_1_16.npz",
        "lbp_64x64s_4_8.npz"
    ]

    hyperparameter_sets = [
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 5},
        {"kernel": "linear", "C": 10.0},
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 0.1, "gamma": "auto"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": "auto"},
        {"kernel": "rbf", "C": 5.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 5.0, "gamma": "auto"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "auto"},
        {"kernel": "rbf", "C": 1.0, "gamma": 0.1},
        {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 2},
        {"kernel": "poly", "C": 1.0, "gamma": "auto", "degree": 2},
        {"kernel": "poly", "C": 5.0, "gamma": "scale", "degree": 2},
        {"kernel": "poly", "C": 5.0, "gamma": "auto", "degree": 2},
        {"kernel": "poly", "C": 10.0, "gamma": "scale", "degree": 3},
        {"kernel": "poly", "C": 10.0, "gamma": "auto", "degree": 3},
        {"kernel": "sigmoid", "C": 0.1, "gamma": 0.1},
        {"kernel": "sigmoid", "C": 1, "gamma": 0.1},
        {"kernel": "sigmoid", "C": 5, "gamma": 0.1},
        {"kernel": "sigmoid", "C": 10.0, "gamma": "auto"},
        {"kernel": "sigmoid", "C": 10.0, "gamma": "scale"}
    ]

    # Iterate through feature files and hyperparameter sets
    for idx_f, features_file in enumerate(features_files, 1):
        print(f"file: {idx_f}/{len(features_files)}")
        for params in tqdm(hyperparameter_sets):
            kernel = params["kernel"]
            C = params["C"]
            gamma = params.get("gamma", "scale")  # Default to 'scale'
            train_svm(features_file, kernel=kernel, C=C, gamma=gamma)
            gc.collect()
        gc.collect()


def test_mdc_hiperparameters():
    """
    Test Minimum Distance Classifier (MDC) with multiple hyperparameter combinations.
    Calls `train_mdc` for each configuration, relying on it to log results.
    """
    # Feature files to test
    features_files = [
        "gabor_21.npz",
        "gabor_31.npz",
        "gabor_41.npz",
        "lbp_1_8.npz",
        "lbp_2_16.npz",
        "lbp_4_32.npz",
        "lbp_1_16.npz",
        "lbp_4_8.npz"
    ]

    # Distance metrics to test
    distance_metrics = [
        {"name": "chi_square", "function": chi_square_distance},
        {"name": "euclidean", "function": euclidean_distance},
        {"name": "mahalanobis", "function": mahalanobis_distance}
    ]

    # Centroid methods to test
    centroid_methods = ["mean", "median"]

    # Normalization methods to test
    normalizations = ["none", "standard", "minmax"]

    # Regularization for Mahalanobis distance
    regularizations = [0, 1e-5, 1e-3, 1e-1]

    # Iterate through all feature files
    for idx_f, features_file in enumerate(features_files, 1):
        print(f"Testing MDC with features file: {features_file} ({idx_f}/{len(features_files)})")

        # Test each distance metric
        for metric in tqdm(distance_metrics, desc="Testing distance metrics"):
            metric_name = metric["name"]
            metric_function = metric["function"]

            # Test each centroid method
            for centroid_method in centroid_methods:

                # Test each normalization method
                for normalization in normalizations:

                    # Handle Mahalanobis-specific regularization
                    if metric_name == "mahalanobis":
                        for regularization in regularizations:
                            print(
                                f"  Metric: {metric_name}, Centroid: {centroid_method}, Normalization: {normalization}, Reg: {regularization}")

                            # Call `train_mdc` with the configuration
                            train_mdc(
                                features_file=features_file,
                                distance_metric=metric_function,
                                centroid_method=centroid_method,
                                normalization=normalization,
                                regularization=regularization
                            )
                            gc.collect()
                    else:
                        # For non-Mahalanobis metrics
                        print(f"  Metric: {metric_name}, Centroid: {centroid_method}, Normalization: {normalization}")

                        # Call `train_mdc` with the configuration
                        train_mdc(
                            features_file=features_file,
                            distance_metric=metric_function,
                            centroid_method=centroid_method,
                            normalization=normalization
                        )
                        gc.collect()


def main():
    # for file_name, method, ksize, radius, n_points in [
    #     ["gabor_64x64l_21.npz", "gabor", 21, None, None],
    #     ["gabor_64x64l_31.npz", "gabor", 31, None, None],
    #     ["gabor_64x64l_41.npz", "gabor", 41, None, None],
    #     ["lbp_64x64l_1_8.npz", "lbp", None, 1, 8],
    #     ["lbp_64x64l_2_16.npz", "lbp", None, 2, 16],
    #     ["lbp_64x64l_4_32.npz", "lbp", None, 4, 32],
    #     ["lbp_64x64l_1_16.npz", "lbp", None, 1, 16],
    #     ["lbp_64x64l_4_8.npz", "lbp", None, 4, 8]
    # ]:
    #     extract_features_to_file(file_name, method=method, ksize=ksize, radius=radius, n_points=n_points)
    #     gc.collect()
    # for model_name, shape in [
    #     # ("vgg16", (320, 320))
    #     ("resnet50", (224, 224)),
    #     # ("inception_v3", (299, 299)),
    # ]:
    #     train_model(model_name, shape)
    # test_svm_hiperparameters()
    # test_mdc_hiperparameters()
    # data = np.load("gabor_21.npz")
    # print(data['x'].shape, data['y'].shape)
    # print(np.unique(data['y'], return_counts=True))  # Check class distribution
    # print(np.isnan(data['x']).any(), np.isnan(data['y']).any())  # Check for NaNs
    # print(np.all(data['x'] == 0))  # Check if all features are zeros
    # with open("svm_training.json", 'r') as json_file:
    #     data = json.load(json_file)
    # for d in data:
    #     print(f'accuracy for {d["features_file"]}: {d["classification_report"]["accuracy"]}')
    # train_svm("gabor_31.npz", kernel="linear", C=1)
    # train_svm()
    # train_mdc(euclidean_distance) # mahalanobis_distance, chi_square_distance
    # train_model()
    # for model_name in ["raw_image", "crop_eyes", "crop_face", "frontalized", "raw_image_hist_eq"]:
    # test_model("resnet50", (224, 224))
    # print_accuracy_per_emotion("Crop_face_")
    # for name in ["Raw_image", "Raw_image_no_norm", "Raw_image_hist_eq", "Raw_image_clahe", "Crop_face", "Crop_eyes", "Frontalized"]:
    #     plot_emotion_accuracies("RAFD", 20, name, 8, False, True)
    # plot_heatmaps("Crop_face", "VGG16")
    # plot_emotion_first_last_chunks("RAFD", 20, "Raw_image", 8, False, False)
    plot_emotion_accuracies("RAFD", 50, "Crop_face_resnet50", plot_train=False, save_plot=True, plot_title="Wykres jakości klasyfikacji dla każdej z emocji\nResNet50")
    # plot_all_models("RAFD", 20, name_prefixes=[], exact_names=[], filters=[])#, exact_names=["Raw_image", "Cropped_eyes_24_48", "Cropped_eyes_24_48_rotated", "Frontalized"])
    # print(print_names("FER"))
    # with open("mdc_training.json", 'r') as json_file:
    #     data = json.load(json_file)
    #
    # gabor_data = [entry for entry in data if "lbp" in entry["features_file"]]
    #
    # sorted_gabor_data = sorted(gabor_data, key=lambda x: x["accuracy"], reverse=True)
    #
    # # Retrieve the 4 best and 1 worst accuracies
    #
    # best_4_gabor = sorted_gabor_data[:4]
    #
    # worst_gabor = sorted_gabor_data[-1]
    #
    #
    # # Display results
    # for i, g in zip(range(4), best_4_gabor):
    #     print(f"place: {i+1} - file name: {g['features_file']} - accuracy: {g['accuracy']}, distance_metric: {g['distance_metric']}, centroid_method: {g['centroid_method']}")
    # print(f"place: {len(sorted_gabor_data)} - file name: {worst_gabor['features_file']} - accuracy: {worst_gabor['accuracy']}, distance_metric: {worst_gabor['distance_metric']}, centroid_method: {worst_gabor['centroid_method']}")


if __name__ == "__main__":
    main()
    pass
