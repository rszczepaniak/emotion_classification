"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils functions
"""
import cv2
import numpy as np
# import seaborn as sn
from torchvision.transforms.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms.transforms import ToTensor, ToPILImage


def random_rotation(image_in):
    image = np.copy(image_in)
    h, w = image.shape[0:2]
    center = (w // 2, h // 2)
    angle = int(np.random.randint(-10, 10))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, image.shape)
    return np.copy(image)


def get_transforms():
    # transform = Compose([random_rotation, ToPILImage(), RandomCrop(46), Resize((48,48)),
    #                          RandomHorizontalFlip(0.5), ToTensor()])
    transform = Compose([ToPILImage(), RandomHorizontalFlip(0.5), ToTensor()])
    return transform


def get_label_emotion(label: int) -> str:
    label_emotion_map = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    return label_emotion_map[label]


def tensor_to_numpy(image):
    if type(image) != np.ndarray:
        return image.cpu().squeeze().numpy()
    return image


def histogram_equalization(image):
    # Step 1: Convert the image to grayscale (since it's grayscale, we can use just one channel)
    gray_image = image[:, :, 0]  # Grayscale, so all channels are identical
    gray_image = np.clip(gray_image * 255, 0, 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8

    # Step 2: Apply histogram equalization to the grayscale image
    equalized_gray = cv2.equalizeHist(gray_image)

    # Step 3: Upscale back to 3 channels (replicate the grayscale channel)
    equalized_image = np.stack([equalized_gray] * 3, axis=-1)

    return equalized_image


def apply_clahe(image):
    # Convert to grayscale if image is in color (optional)
    gray_image = image[:, :, 0]  # Grayscale, so all channels are identical
    gray_image = np.clip(gray_image * 255, 0, 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8

    # Create CLAHE object with a clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE
    clahe_image = clahe.apply(gray_image)

    return clahe_image


def histogram_stretching(image):
    # Ensure the image is grayscale but in 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Extract the first channel (which is identical across r, g, b)
        image = image[:, :, 0]

    # Ensure the image is in uint8 format and in the range [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Compute the minimum and maximum pixel values in the image
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    print(f"Original min pixel value: {min_pixel}")
    print(f"Original max pixel value: {max_pixel}")

    # Check if the image has no contrast (max_pixel == min_pixel)
    if min_pixel == max_pixel:
        print("Image has no contrast, returning original image")
        return image

    # Apply min-max normalization to stretch the range to 0-255
    stretched_image = (image - min_pixel) * 255 / (max_pixel - min_pixel)

    # Ensure the pixel values are clipped to [0, 255] and converted to uint8
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return stretched_image


def gamma_correction(image, gamma=1.0):
    # Normalize the image to range [0, 1]
    image_normalized = image / 255.0

    # Apply gamma correction
    corrected_image = np.power(image_normalized, gamma)

    # Rescale back to the range [0, 255]
    corrected_image = np.clip(corrected_image * 255, 0, 255)

    return np.uint8(corrected_image)


def z_score_normalization(image):
    # Convert to float32 for the calculations
    image_float = image.astype(np.float32)

    # Compute the mean and standard deviation
    mean = np.mean(image_float)
    std = np.std(image_float)

    # Apply Z-score normalization
    normalized_image = (image_float - mean) / std

    # Rescale the image to the range 0-255 and convert to uint8
    normalized_image = np.clip(
        (normalized_image - np.min(normalized_image)) * 255 / (np.max(normalized_image) - np.min(normalized_image)), 0,
        255)

    return np.uint8(normalized_image)


def apply_canny(image):
    # Convert to grayscale if it's in color
    gray_image = image[:, :, 0]  # Grayscale, so all channels are identical
    gray_image = np.clip(gray_image * 255, 0, 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8

    # Ensure it's in uint8
    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    return edges


def gaussian_blur(image, kernel_size=(5, 5)):
    # Ensure the image is in uint8 format
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred_image


def normalization(face):
    face = tensor_to_numpy(face)
    # [-1,1] range
    # face = np.clip(face, 0, 255).astype(np.float32)
    mean = np.mean(face)
    std = np.std(face)

    # if black_image
    # if int(mean) == 0 and int(std) == 0:
    #     return face

    face = (face - mean) / std
    face = face.astype(np.float32)
    # print(f'mean = {mean}, std={std}')
    # normalization will change mean/std but will have overflow in max/min values
    face = np.clip(face, -1, 1)
    # convert from [-1,1] range to [0,1]
    face = (face + 1) / 2
    # face = (face * 255).astype(np.uint8)
    return face.astype(np.float32)


def standerlization(image):
    image = tensor_to_numpy(image)

    # standerlization .. convert it to 0-1 range
    min_img = np.min(image)
    max_img = np.max(image)
    image = (image - min_img) / (max_img - min_img)
    return image.astype(np.float32)


# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/

def is_black_image(face):
    # training dataset contains 10 images black & 1 in val dataset
    face = tensor_to_numpy(face)
    mean = np.mean(face)
    std = np.std(face)
    if int(mean) == 0 and int(std) == 0:
        return True
    return False


def normalize_dataset_mode_1(image):
    mean = 0.5077425080522144
    std = 0.21187228780099732
    image = (image - mean) / std
    return image


def normalize_dataset_mode_255(image):
    mean = 129.47433955331468
    std = 54.02743338925431
    image = (image - mean) / std
    return image

# def visualize_confusion_matrix(confusion_matrix):
#     df_cm = pd.DataFrame(confusion_matrix, range(7), range(7))
#     sn.set(font_scale=1.1) # for label size
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#     plt.show()
