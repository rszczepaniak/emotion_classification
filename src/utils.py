"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils functions
"""
import cv2
import numpy as np
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
    """
    Applies histogram equalization to each channel of an RGB image or directly to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input image, can be grayscale or RGB.

    Returns:
        numpy.ndarray: Image after histogram equalization.
    """
    # Check if the image is grayscale or RGB
    if len(image.shape) == 3:  # RGB image
        equalized_image = cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(image)])
    else:  # Grayscale image
        equalized_image = cv2.equalizeHist(image)
    return equalized_image


def apply_clahe(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to each channel of an RGB image or
    directly to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input image, can be grayscale or RGB.

    Returns:
        numpy.ndarray: Image after CLAHE.
    """
    # Create CLAHE object with a clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Check if the image is grayscale or RGB
    if len(image.shape) == 3:  # RGB image
        # Split the image into its R, G, B channels
        channels = cv2.split(image)
        # Apply CLAHE to each channel
        clahe_channels = [clahe.apply(channel) for channel in channels]
        # Merge the channels back into an RGB image
        clahe_image = cv2.merge(clahe_channels)
    else:  # Grayscale image
        # Apply CLAHE directly
        clahe_image = clahe.apply(image)

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


def gaussian_blur(image, kernel_size=(5, 5)):
    # Ensure the image is in uint8 format
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred_image


def normalization(face):
    """
    Normalizes a grayscale or RGB image. Converts the pixel range to [0, 255] after normalization.

    Parameters:
        face (numpy.ndarray): Input image, either grayscale or RGB.

    Returns:
        numpy.ndarray: Normalized image with pixel values in the range [0, 255].
    """
    face = tensor_to_numpy(face)  # Ensure the input is a numpy array

    # Handle black images or near-zero standard deviation
    epsilon = 1e-6  # Small value to avoid division by zero

    if len(face.shape) == 3:  # RGB image
        # Normalize each channel independently
        face = cv2.merge([(np.clip((channel - np.mean(channel)) / max(np.std(channel), epsilon), -1, 1) + 1) / 2 for channel in cv2.split(face)])
    else:  # Grayscale image
        face = (np.clip((face - np.mean(face)) / max(np.std(face), epsilon), -1, 1) + 1) / 2  # Scale to [0, 1]

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
