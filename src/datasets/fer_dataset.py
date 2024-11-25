import argparse
import os
from collections import Counter
from functools import lru_cache

import cv2
import dlib
import mediapipe as mp
import numpy as np
import pandas as pd
import torchvision.transforms.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io
from torch.utils.data import Dataset, DataLoader

import face_frontalization.camera_calibration as calib
from face_frontalization import frontalize
from src.face_alignment import FaceAlignment
from src.landmarks_detector import dlibLandmarks
from src.utils import get_label_emotion, standerlization, normalize_dataset_mode_255, \
    get_transforms


def detect_landmarks(image):
    """
    Detects facial landmarks using dlib.
    """
    height, width = image.shape
    landmarks_detector = dlibLandmarks()
    rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
    landmarks = landmarks_detector.detect_landmarks(image, rect)
    return landmarks


def convert_pixels_to_image(pixels_str):
    """
    Converts a pixel string to a 48x48 image.
    """
    face = np.fromstring(pixels_str, sep=' ').reshape(48, 48).astype(np.uint8)
    return face


def resize_image(image, size):
    """
    Resizes an image to the given size.
    """
    return cv2.resize(image, size)


def detect_landmarks_with_mediapipe(image):
    """
    Detect facial landmarks using Mediapipe.
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        return [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks.landmark]


def display_landmarks(image, landmarks):
    """
    Displays the detected landmarks on the image.

    Args:
        image (numpy.ndarray): The original image.
        landmarks (list): List of tuples containing landmark coordinates (x, y).
    """
    if landmarks is None:
        print("No landmarks detected.")
        return

    # Make a copy of the image to draw landmarks on
    output_image = image.copy()

    # Draw each landmark as a small circle
    for (x, y) in landmarks:
        cv2.circle(output_image, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)

    # Display the image with landmarks
    cv2.imshow('Landmarks', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_eye_alignment(image, left_eye, right_eye):
    """
    Draws eye landmarks, the line connecting both eyes, and a horizontal alignment line.

    Args:
        image (numpy.ndarray): Input image.
        left_eye (tuple): Coordinates of the left eye center (x, y).
        right_eye (tuple): Coordinates of the right eye center (x, y).
        angle (float): Rotation angle used for alignment.

    Returns:
        numpy.ndarray: Image with the visualization.
    """
    output_image = image.copy()

    # Draw eye centers
    if left_eye is not None:
        cv2.circle(output_image, (int(left_eye[0]), int(left_eye[1])), 5, (255, 0, 0), -1)  # Blue for left eye
    if right_eye is not None:
        cv2.circle(output_image, (int(right_eye[0]), int(right_eye[1])), 5, (0, 0, 255), -1)  # Red for right eye

    return output_image


def display_eye_centers(image, left_eye, right_eye):
    """
    Visualizes the centers of the left and right eyes on the image.

    Args:
        image (numpy.ndarray): The original image.
        left_eye (tuple): Coordinates of the left eye center (x, y).
        right_eye (tuple): Coordinates of the right eye center (x, y).
    """
    # Make a copy of the image to draw on
    output_image = image.copy()

    # Draw the left eye center (blue)
    cv2.circle(output_image, (int(left_eye[0]), int(left_eye[1])), radius=5, color=(255, 0, 0), thickness=-1)

    # Draw the right eye center (red)
    cv2.circle(output_image, (int(right_eye[0]), int(right_eye[1])), radius=5, color=(0, 0, 255), thickness=-1)

    # Display the image with the eye centers
    cv2.imshow('Eye Centers', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_mediapipe_eye_centers(landmarks):
    """
    Returns the centers of the left and right eyes from the landmarks.
    """
    left_eye_landmarks = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]  # Left
    right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # right
    # Eye landmarks based on Mediapipe face mesh model
    right_eye = np.mean([landmarks[i] for i in left_eye_landmarks], axis=0).astype(np.uint8)
    left_eye = np.mean([landmarks[i] for i in right_eye_landmarks], axis=0).astype(np.uint8)
    return left_eye, right_eye


def compute_rotation_angle(left_eye, right_eye):
    """
    Computes the angle of rotation required to align the eyes.
    """
    dx = int(right_eye[0]) - int(left_eye[0])
    dy = int(right_eye[1]) - int(left_eye[1])
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def rotate_image(image, angle, center):
    """
    Rotates the image around a given center and angle.
    """
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image


def rotate_eye_centers(left_eye, right_eye, angle, center):
    """
    Rotates the eye centers around a given center point and angle.

    Args:
        left_eye (tuple): Coordinates of the left eye center (x, y).
        right_eye (tuple): Coordinates of the right eye center (x, y).
        angle (float): The angle of rotation in degrees (positive is counterclockwise).
        center (tuple): The center of rotation (cx, cy).

    Returns:
        tuple: Rotated left_eye and right_eye as new coordinates.
    """
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Convert eye centers to numpy arrays for matrix operations
    left_eye_point = np.array([left_eye[0], left_eye[1], 1]).reshape(3, 1)  # Homogeneous coordinates
    right_eye_point = np.array([right_eye[0], right_eye[1], 1]).reshape(3, 1)

    # Apply rotation
    rotated_left_eye = np.dot(rotation_matrix, left_eye_point).flatten()
    rotated_right_eye = np.dot(rotation_matrix, right_eye_point).flatten()

    # Return the rotated eye coordinates
    return (rotated_left_eye[:2], rotated_right_eye[:2])


def crop_image_around_eyes(image, left_eye, right_eye):
    """
    Crops the image so that the eyes are level and centered at the middle height.
    """
    height, width = image.shape
    x_min = max(0, int(min(left_eye[0], right_eye[0]) - width * 0.1))
    x_max = min(width, int(max(left_eye[0], right_eye[0]) + width * 0.1))
    y_center = (left_eye[1] + right_eye[1]) // 2
    y_min = max(0, int(y_center - height * 0.2))
    y_max = min(height, int(y_center + height * 0.2))

    cropped_face = image[y_min:y_max, x_min:x_max]
    return cropped_face if cropped_face.size > 0 else None

def visualize_ref_U_3D(ref_U):
    x, y, z = ref_U[:, :, 0], ref_U[:, :, 1], ref_U[:, :, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the arrays for 3D scatter plotting
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=z.flatten(), cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of ref_U')
    plt.show()


def visualize_ref_U_as_image(ref_U):
    ref_U_normalized = (ref_U - ref_U.min()) / (ref_U.max() - ref_U.min())  # Normalize to [0, 1]
    plt.imshow(ref_U_normalized)
    plt.title('ref_U Visualized as an Image')
    plt.axis('off')
    plt.show()


def filter_and_visualize_2d(ref_U):
    # Simulate that channel 1 (Y) < 0 is set to zero for 2D visualization
    y_channel = ref_U[:, :, 1]  # Assuming the second channel represents 'Y'
    mask = np.ones_like(y_channel)  # Start with all ones (keep all values)
    mask[y_channel < 0] = 0  # Set to zero where Y < 0

    # Apply the mask to the original ref_U for visualization
    filtered_ref_U = np.multiply(ref_U, mask[:, :, None])  # Apply along all channels
    return np.sum(filtered_ref_U, axis=2)  # Reduce to 2D by summing across channels for visualization



def frontalize_face(face):
    model3D = frontalize.ThreeD_Model("face_frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
    model3D.ref_U[model3D.ref_U.any() < 0] = 0
    # lmarks = np.array(detect_landmarks_with_mediapipe(face))
    lmarks = detect_landmarks(face)
    if len(face.shape) == 2:  # Grayscale (single-channel)
        face = np.stack((face, face, face), axis=-1)
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks)
    eyemask = np.asarray(io.loadmat('face_frontalization/frontalization_models/eyemask.mat')['eyemask'])
    _frontal_raw, frontal_sym = frontalize.frontalize(face, proj_matrix, model3D.ref_U, eyemask)
    return frontal_sym


def crop_and_rotate_eyes(face):
    landmarks = detect_landmarks_with_mediapipe(face)
    if landmarks is None:
        return None

    left_eye, right_eye = get_mediapipe_eye_centers(landmarks)

    angle = compute_rotation_angle(left_eye, right_eye)
    face = rotate_image(face, angle, center=(face.shape[1] // 2, face.shape[0] // 2))
    left_eye, right_eye = rotate_eye_centers(left_eye, right_eye, angle,
                                             center=(face.shape[1] // 2, face.shape[0] // 2))

    face = crop_image_around_eyes(face, left_eye, right_eye)
    return face


@lru_cache(maxsize=10000)
def preprocess_image(pixels_str):
    try:
        face = convert_pixels_to_image(pixels_str)
        face = resize_image(face, (320, 320))
        cv2.imshow("face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        face = frontalize_face(face)
        cv2.imshow("frontalized", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if face is not None:
            face = resize_image(face, (48, 48))  # 48, 24
        return face
    except Exception as e:
        print(f"SOMETHING WENT WRONG: {e}")
        return None


def process_row(pixels):
    return preprocess_image(pixels)


def prepare_dataframe(csv_path, mode="train"):
    df = pd.read_csv(csv_path)
    mode_map = {'train': 'Training', 'val': 'PrivateTest', 'test': 'PublicTest'}
    df = df[df['Usage'] == mode_map[mode]]

    # with Pool(cpu_count()) as pool:
    #     df['face'] = list(tqdm(pool.imap(process_row, df['pixels']), total=len(df['pixels'])))

    df['face'] = df['pixels'].apply(preprocess_image)
    df = df.dropna(subset=['face'])
    return df[['emotion', 'face']]


class FER2013(Dataset):
    """
    FER2013 format:
        index   emotion     pixels      Usage

    index: id of series
    emotion: label (from 0 - 6)
    pixels: 48x48 pixel value (uint8)
    Usage: [Training, PrivateTest, PublicTest]    
    """

    def __init__(self, root='data_unpacked', mode='train', transform=None):
        self.root = root
        self.transform = transform
        assert mode in ['train', 'val', 'test']

        self.csv_path = os.path.join(self.root, 'fer2013.csv')
        self.df = prepare_dataframe(self.csv_path, mode)
        # print(self.df)

    def __getitem__(self, index: int):
        data_series = self.df.iloc[index]
        emotion = data_series['emotion']
        face = data_series['face']

        # Convert to PIL Image
        face = Image.fromarray(face)
        face = self.transform(face)
        return face, emotion

    def __len__(self) -> int:
        return self.df.index.size


def create_train_dataloader(root='data_unpacked', batch_size=32):
    dataset = FER2013(root, mode='train', transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader


def create_val_dataloader(root='data_unpacked', batch_size=2):
    dataset = FER2013(root, mode='val', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader


def create_test_dataloader(root='data_unpacked', batch_size=1):
    # transform = transforms.ToTensor()
    transform = get_transforms()
    dataset = FER2013(root, mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader


def calculate_dataset_mean_std(dataset: FER2013):
    n = len(dataset)
    means = []
    stds = []
    for i in range(n):
        image, _ = dataset[i]
        # image = image/ 255
        mean = np.mean(image)
        std = np.std(image)
        means.append(mean)
        stds.append(std)
        print(f'i={i}, mean = {mean}, std = {std}')
    mean = np.mean(means)
    std = np.mean(stds)
    print(f'\n\t Mean = {mean} ... Std = {std}\n')


def test_dataloader_main():
    dataloader = create_test_dataloader()
    for image, label in dataloader:
        image = image.squeeze().numpy()
        cv2.imshow('img', image)
        print(image.shape)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='train', help='dataset mode')
    parser.add_argument('--datapath', type=str, default='data_unpacked')
    parser.add_argument('--mean_std', action='store_true', help='calculate mean std of dataset')
    parser.add_argument('--test', action='store_true', help='test augumentation')

    args = parser.parse_args()

    if args.test:
        test_dataloader_main()
        exit(0)

    dataset = FER2013(args.datapath, args.mode)
    print(f'dataset size = {len(dataset)}')

    if args.mean_std:
        calculate_dataset_mean_std(dataset)
        exit(0)

    for i in range(len(dataset)):

        face, emotion = dataset[i]
        # print('emotion',emotion)
        # print('shape',face.shape)
        face = np.copy(face)
        print(f"before min:{np.min(face)}, max:{np.max(face)}, mean:{np.mean(face)}, std:{np.std(face)}")
        face1 = normalize_dataset_mode_255(face)
        # face1 = normalization(face)
        face2 = standerlization(face)
        print(f"after min:{np.min(face)}, max:{np.max(face)}, mean:{np.mean(face)}, std:{np.std(face)}\n")

        face = cv2.resize(face, (200, 200))
        cv2.putText(face, get_label_emotion(emotion), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('face', face)
        fa = FaceAlignment()
        print(len(face))
        break
        new_face = fa.frontalize_face(face, 0)
        cv2.imshow('new_face', new_face)

        # face1 = cv2.resize(face1, (200, 200))
        # face2 = cv2.resize(face2, (200, 200))
        # cv2.imshow('normalization', face1)
        # cv2.imshow('standerlization', face2)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

    # df = pd.DataFrame({
    #     "name": ["amr",'ELSERSY', 'sersy'],
    #     'salary': [100,20,3000],
    #     'job': ['software', 'ray2', 'mech']
    # }, index=[0,1,4])

    # print(df.index.size)

    # print(df)

    # ser = df[df['salary'] > 50]
    # print(type(ser)) # dataframe
    # print(ser)

    # salary = df[df['salary'] < 200]  # dataframe
    # print(type(salary.iloc[1])) # series
    # print(salary.iloc[1],'\n')
    # print(salary.iloc[0]['name'])
    # print(salary.shape)
    # print(salary.iloc[0]['name'])

    # df = df[df['salary'] == 3000] 
    # print(df[['name', "salary"]])

    # df = df[df['salary'] < 500]
    # print(df.index)
    # print(df.iloc[0].values)
    # print(type(df.values))
    # print('==========================')
    # # print(df.count()) # count of each series in the dataframe
    # print(df.iloc[0].count())
