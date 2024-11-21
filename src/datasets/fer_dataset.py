import argparse
import os
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import cv2
import dlib
import numpy as np
import pandas as pd
import torchvision.transforms.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.face_alignment import FaceAlignment
from src.landmarks_detector import dlibLandmarks
from src.utils import get_label_emotion, standerlization, normalize_dataset_mode_255, \
    get_transforms, apply_clahe, histogram_equalization


def crop_eyes(image):
    image = cv2.resize(image, (300, 300))
    height, width = image.shape
    landmarks_detector = dlibLandmarks()
    rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
    landmarks = landmarks_detector.detect_landmarks(image, rect)  # Assuming landmarks are in (x, y) format

    right_eye = ((landmarks[0] + landmarks[1]) // 2).astype(np.uint8)
    left_eye = ((landmarks[2] + landmarks[3]) // 2).astype(np.uint8)

    # Use safe conversion and clamping
    x_min = max(0, int(min(left_eye[0], right_eye[0]) - width * 0.1))
    x_max = min(width, int(max(left_eye[0], right_eye[0]) + width * 0.1))
    y_min = max(0, int(min(left_eye[1], right_eye[1]) - height * 0.1))
    y_max = min(height, int(max(left_eye[1], right_eye[1]) + height * 0.1))

    # Crop the region
    eyes_crop = image[y_min:y_max, x_min:x_max]
    if eyes_crop.any():
        return cv2.resize(eyes_crop, (100, 50))
    return None


@lru_cache(maxsize=10000)
def preprocess_image(pixels_str):
    try:
        # Convert string to numpy array
        face = np.fromstring(pixels_str, sep=' ').reshape(48, 48).astype(np.uint8)

        # Resize and process
        face = cv2.resize(face, (300, 300))
        # cv2.imshow("face", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face = apply_clahe(face)
        # face = histogram_equalization(face)
        # cv2.imshow("face eq", face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # height, width = face.shape
        # landmarks_detector = dlibLandmarks()
        # rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
        # landmarks = landmarks_detector.detect_landmarks(face, rect)
        #
        # right_eye = ((landmarks[0] + landmarks[1]) // 2).astype(np.uint8)
        # left_eye = ((landmarks[2] + landmarks[3]) // 2).astype(np.uint8)
        #
        # x_min = max(0, int(min(left_eye[0], right_eye[0]) - width * 0.1))
        # x_max = min(width, int(max(left_eye[0], right_eye[0]) + width * 0.1))
        # y_min = max(0, int(min(left_eye[1], right_eye[1]) - height * 0.1))
        # y_max = min(height, int(max(left_eye[1], right_eye[1]) + height * 0.1))
        #
        # face = face[y_min:y_max, x_min:x_max]
        if face.any():
            face = cv2.resize(face, (48, 48))  # (48, 24)
            # cv2.imshow("processed", face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return face
        return  None
    except Exception as e:
        print(f"SOMETHING WENT WRONG: {e}")
        return None

def process_row(pixels):
    return preprocess_image(pixels)

def prepare_dataframe(csv_path, mode="train"):
    df = pd.read_csv(csv_path)
    mode_map = {'train': 'Training', 'val': 'PrivateTest', 'test': 'PublicTest'}
    df = df[df['Usage'] == mode_map[mode]]

    with Pool(cpu_count()) as pool:
        df['face'] = list(tqdm(pool.imap(process_row, df['pixels']), total=len(df['pixels'])))

    # df['face'] = df['pixels'].apply(preprocess_image)
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
