import argparse
import gc
import os
import re
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io
from scipy.special.cython_special import spherical_jn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import face_frontalization.camera_calibration as calib
from face_frontalization import frontalize
from src.datasets.parse_mp_to_dlib import convert_landmarks_mediapipe_to_dlib
from src.face_alignment import FaceAlignment
from src.face_detector.face_detector import DnnDetector, detect_landmarks_with_mediapipe, get_mediapipe_eye_centers
from src.landmarks_detector import dlibLandmarks
from src.utils import get_label_emotion, standerlization, normalize_dataset_mode_255, \
    get_transforms, histogram_equalization, apply_clahe, normalization


def detect_landmarks(face):
    """
    Detects facial landmarks using dlib.
    """
    # height, width = image.shape[:2]
    # face = DnnDetector().detect_faces(image)[0]
    cv2.imshow("face1", face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    height, width = face.shape[:2]
    landmarks_detector = dlibLandmarks()
    rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
    landmarks = landmarks_detector.detect_landmarks(face, rect)
    return landmarks


def convert_pixels_to_image(pixels_str, shape, rgb=True):
    """
    Converts a pixel string to a 2D (grayscale) or 3D (RGB) image.

    Parameters:
        pixels_str (str): A space-separated string of pixel values.
        shape (tuple): A tuple representing the shape of the image (height, width).
        rgb (bool): If True, converts the pixel string into an RGB image (3 channels).
                    If False, converts it into a grayscale image (1 channel).

    Returns:
        np.ndarray: The reconstructed image as a numpy array.
    """
    # Convert the pixel string into a numpy array
    pixels = np.fromstring(pixels_str, sep=' ').astype(np.uint8)

    if rgb:
        # Reshape into (Height, Width, 3) for RGB
        face = pixels.reshape(shape[0], shape[1], 3)
    else:
        # Reshape into (Height, Width) for grayscale
        face = pixels.reshape(shape[0], shape[1])
        face = np.stack((face,) * 3, axis=-1)
    return face


def resize_image(image, size):
    """
    Resizes an image to the given size.
    """
    return cv2.resize(image, size)


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



def compute_rotation_angle(left_eye, right_eye):
    """
    Computes the angle of rotation required to align the eyes.
    """
    dx = abs(int(right_eye[0]) - int(left_eye[0]))
    dy = abs(int(right_eye[1]) - int(left_eye[1]))
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def rotate_image(image, angle, center):
    """
    Rotates the image around a given center and angle.
    """
    h, w = image.shape[:2]
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

    # Homogeneous coordinates for matrix operations
    left_eye_point = np.array([left_eye[0], left_eye[1], 1])
    right_eye_point = np.array([right_eye[0], right_eye[1], 1])

    # Apply the rotation
    rotated_left_eye = np.dot(rotation_matrix, left_eye_point.T).astype(int)
    rotated_right_eye = np.dot(rotation_matrix, right_eye_point.T).astype(int)

    return rotated_left_eye[:2], rotated_right_eye[:2]


def crop_image_around_eyes(image, left_eye, right_eye, margin=0.1):
    """
    Crops the image so that the eyes are level and centered at the middle height.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or RGB).
        left_eye (tuple): (x, y) coordinates of the left eye.
        right_eye (tuple): (x, y) coordinates of the right eye.
        margin (float): Fraction of the image dimensions to include as margin around the eyes.

    Returns:
        numpy.ndarray or None: Cropped face region or None if invalid crop.
    """
    height, width = image.shape[:2]

    # Validate input eye coordinates
    left_eye = (max(0, min(left_eye[0], width - 1)), max(0, min(left_eye[1], height - 1)))
    right_eye = (max(0, min(right_eye[0], width - 1)), max(0, min(right_eye[1], height - 1)))

    # Calculate cropping boundaries with margins
    x_min = max(0, int(min(left_eye[0], right_eye[0]) - width * margin))
    x_max = min(width, int(max(left_eye[0], right_eye[0]) + width * margin))

    y_min = max(0, int(min(left_eye[1], right_eye[1]) - height * margin))
    y_max = min(height, int(max(left_eye[1], right_eye[1]) + height * margin))

    # Ensure the crop area is valid
    if x_max > x_min and y_max > y_min:
        cropped_face = image[y_min:y_max, x_min:x_max]
        return cropped_face
    return None


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


def frontalize_face(face):
    model3D = frontalize.ThreeD_Model("face_frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
    # lmarks = np.array(detect_landmarks_with_mediapipe(face))
    # lmarks = detect_landmarks(face)
    lmarks = detect_landmarks_with_mediapipe(face)
    lmarks = convert_landmarks_mediapipe_to_dlib(lmarks)
    if len(face.shape) == 2:  # Grayscale (single-channel)
        face = np.stack((face, face, face), axis=-1)
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks)
    eyemask = np.asarray(io.loadmat('face_frontalization/frontalization_models/eyemask.mat')['eyemask'])
    _frontal_raw, frontal_sym = frontalize.frontalize(face, proj_matrix, model3D.ref_U, eyemask)
    return frontal_sym


def crop_middle_dynamic(image, crop_size):
    """
    Crops the middle square of the specified size from the image using slicing.

    Args:
        image (numpy.ndarray): Input image.
        crop_size (int): Size of the square crop.

    Returns:
        numpy.ndarray: Cropped image.
    """
    height, width = image.shape[:2]

    # Calculate the top-left corner for the crop
    start_y = max(0, (height - crop_size) // 2)
    start_x = max(0, (width - crop_size) // 2)

    # Calculate the bottom-right corner for the crop
    end_y = start_y + crop_size
    end_x = start_x + crop_size

    # Perform the crop
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def preprocess_image(pixels_str, shape=(320, 320), need_convert=True):
    try:
        if need_convert:
            face = convert_pixels_to_image(pixels_str, shape, True)
            pixels_str = None
            gc.collect()
        else:
            face = pixels_str

        # cv2.imwrite("face.jpg", face)

        # idx = len(os.listdir("images"))
        # cv2.imwrite(f"images/{idx}_orig_rafd.jpg", orig_face)
        # face = change_background_to_black(face)
        # face = DnnDetector().detect_faces(face)[0]
        # face = resize_image(face, (320, 320))
        # lmarks = detect_landmarks_with_mediapipe(face)
        # left_eye, right_eye = get_mediapipe_eye_centers(lmarks)
        # face = crop_image_around_eyes(face, left_eye, right_eye)
        # cv2.imshow("face", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # face = resize_image(face, (512, 340))
        # cv2.imwrite("cropped_face.jpg", face)
        # cv2.imshow("cropped", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(f"images/{idx}_crop_rafd.jpg", face)
        # face = crop_and_rotate_eyes(face)
        # face = frontalize_face(face)
        # face = face[60:260, 60:260]
        # face = resize_image(face, (320, 320))
        # cv2.imwrite(f"images/{idx}_frontalized_rafd.jpg", front_face)
        # save_images_side_by_side(f"images/_cropped_face.jpg", orig_face, face)
        # cv2.imshow("face", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # face = apply_clahe(face)
        # face_eq = histogram_equalization(face)
        # face = None
        # del face
        # gc.collect()
        # cv2.imshow("origface", orig_face)
        # cv2.imshow("face", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # face_norm = normalization(face)
        # face = None
        # del face
        # gc.collect()
        # lmarks = detect_landmarks_with_mediapipe(face)
        # left_eye, right_eye = get_mediapipe_eye_centers(lmarks)
        # face = crop_image_around_eyes(face, left_eye, right_eye)
        #
        # if face is not None:
            # face = resize_image(face, (256, 170))  # 48, 24 \ 1024, 681
            # cv2.imshow("pre-processes", face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return face
    except Exception as e:
        print(f"SOMETHING WENT WRONG: {e}")
        # raise Exception("koniec")
        return None


def save_images_side_by_side(output_path, *images):
    """Helper function to save images side by side to a file."""
    # Resize all images to have the same dimensions
    height, width = 320, 320  # Set a fixed height and width for uniformity

    # Concatenate images horizontally
    concatenated_image = np.hstack(tuple(cv2.resize(image, (width, height)) for image in images))

    # Save the concatenated image to the specified file path
    cv2.imwrite(output_path, concatenated_image)
    print(f"Image saved successfully at: {output_path}")


def process_row(pixels):
    return preprocess_image(pixels)


def prepare_dataframe(df, mode="train"):
    with Pool(cpu_count()) as pool:
        df['face'] = list(tqdm(pool.imap(process_row, df['pixels']), total=len(df['pixels'])))
    # df['face'] = df['pixels'].apply(preprocess_image)
    # Drop NaN rows and reset index to compact memory
    df = df.dropna(subset=['face']).reset_index(drop=True)

    # Drop 'pixels' column to save memory
    df = df.drop(columns=['pixels'])

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
        # face = Image.fromarray(face)
        face = self.transform(face)
        return face, emotion

    def __len__(self) -> int:
        return self.df.index.size


class RAFD(Dataset):
    """
    RAFD format:
        emotion     pixels

    emotion: label (from 0 - 7)
    pixels: 1024x681 pixel value (uint8)
    """

    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
        # self.chunk_iterator = iter(self.df)
        # print(self.df)

    def __getitem__(self, index: int):
        data_series = self.df.iloc[index]
        emotion = int(data_series['emotion'])
        face = data_series['face']

        # Convert to PIL Image
        # face = Image.fromarray(face)
        if self.transform is not None:
            face = self.transform(face)

        face_tensor = torch.from_numpy(face) if not isinstance(face, torch.Tensor) else face
        emotion_tensor = torch.tensor(emotion, dtype=torch.long)  # Ensure it's a long tensor for classification

        return face_tensor, emotion_tensor

    def __len__(self) -> int:
        return len(self.df)


class RAFD_DYNAMIC(Dataset):
    """
    RAFD format:
        emotion     pixels

    emotion: label (from 0 - 7)
    pixels: 1024x681 pixel value (uint8)
    """
    EMOTION_MAP = {
        "happy": 0,
        "angry": 1,
        "sad": 2,
        "contemptuous": 3,
        "disgusted": 4,
        "neutral": 5,
        "fearful": 6,
        "surprised": 7
    }
    def __init__(self, skip_indexes: list, transform=None):
        self.transform = transform
        self.paths = [f"data_unpacked/rafd/{file_name}" for idx, file_name in enumerate(os.listdir("data_unpacked/rafd"), 0) if idx not in skip_indexes]

    def __getitem__(self, index: int):
        path = self.paths[index]
        face = cv2.imread(path)
        face = preprocess_image(face, need_convert=False)
        emotion = re.search(r'(happy|angry|sad|contemptuous|disgusted|neutral|fearful|surprised)', path).group(0)

        if self.transform is not None:
            face = self.transform(face)
        return face, self.EMOTION_MAP[emotion]

    def __len__(self) -> int:
        return len(self.paths)


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
