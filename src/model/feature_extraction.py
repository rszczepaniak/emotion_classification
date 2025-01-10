import cv2
import numpy as np
from skimage.feature import local_binary_pattern

#
# def build_gabor_kernels(ksize):
#     kernels = []
#     for theta in np.arange(0, np.pi, np.pi / 16):  # Orientations (e.g., 8 orientations)
#         for sigma in (0.1, 0.5, 1, 3, 5):  # Scale of the filter
#             for lamda in np.pi / np.array([0.2, 0.8, 2, 4]):  # Wavelength
#                 for gamma in (0.7, 1, 1.5):  # Aspect ratio
#                     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
#                     kernels.append(kernel)
#     return kernels
#
#
def build_gabor_kernels(ksize):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 8):  # Fewer orientations
        for sigma in (1, 3):  # Fewer scales
            for lamda in np.pi / np.array([0.8, 2]):  # Fewer wavelengths
                for gamma in (0.7, 1):  # Fewer aspect ratios
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels


# def build_gabor_kernels(ksize):
#     kernels = []
#     for theta in np.arange(0, np.pi, np.pi / 4):  # Fewer orientations
#         for sigma in (0.5, 1.5):  # Fewer scales
#             for lamda in np.pi / np.array([0.8, 1.3]):  # Fewer wavelengths
#                 for gamma in (0.7, 1):  # Fewer aspect ratios
#                     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
#                     kernels.append(kernel)
#     return kernels


def apply_gabor_filters(image, kernels):
    """
    Apply Gabor filters to a normalized grayscale image and extract mean and variance as features.

    Parameters:
        image (np.ndarray): Input image (grayscale, normalized in the range [0, 1]).
        kernels (list): List of Gabor kernels.

    Returns:
        np.ndarray: Concatenated feature vector containing mean and variance for each kernel.
    """
    # Ensure the image is normalized in the range [0, 1]
    if image.max() > 1.0 or image.min() < 0.0:
        raise ValueError("Input image should be normalized in the range [0, 1].")

    # Scale the image to [0, 255] and convert to uint8 for compatibility with OpenCV
    scaled_image = (image * 255).astype(np.uint8)

    # Initialize a list to store features
    features = []

    # Apply each Gabor kernel to the grayscale image
    for kernel in kernels:
        filtered_img = cv2.filter2D(scaled_image, cv2.CV_32F, kernel)  # Use CV_32F for precision

        # Extract mean and variance as features
        features.append(filtered_img.mean())  # Mean as a feature
        features.append(filtered_img.var())  # Variance as a feature

    # Convert features to a numpy array
    features = np.array(features)

    # Normalize the feature vector (L2 normalization)
    features /= (np.linalg.norm(features) + 1e-6)  # Prevent division by zero

    return features

def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) features from a grayscale image.

    Parameters:
        image (np.ndarray): A 2D grayscale image normalized to [0, 1].
        radius (int): Radius for LBP (default is 1).
        n_points (int): Number of points for LBP (default is 8).

    Returns:
        np.ndarray: The LBP histogram features.
    """
    # Ensure the input is 2D (grayscale)
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Scale the image from [0, 1] to [0, 255] and convert to integers
    image = (image * 255).astype(np.uint8)

    # Compute the LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # CHECK?
    # bins = n_points + 2
    # hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, bins + 1), range=(0, bins))

    # Compute the histogram of the LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize to sum to 1

    return hist

def euclidean_distance(vector1, vector2, _covariance_matrix=None):
    """
    Compute the Euclidean distance between two vectors.
    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.
    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def mahalanobis_distance(sample, centroid, covariance_matrix):
    """
    Compute the Mahalanobis distance with regularization for stability.

    Parameters:
        sample (np.ndarray): A single sample.
        centroid (np.ndarray): Centroid of the class.
        covariance_matrix (np.ndarray): Covariance matrix.

    Returns:
        float: Mahalanobis distance.
    """
    regularization = 1e-5  # Small value to add to the diagonal
    regularized_covmat = covariance_matrix + np.eye(covariance_matrix.shape[0]) * regularization

    try:
        inv_covmat = np.linalg.inv(regularized_covmat)
    except np.linalg.LinAlgError:
        raise ValueError("Regularized covariance matrix is still singular. Check your data.")

    diff = sample - centroid
    return np.sqrt(np.dot(np.dot(diff, inv_covmat), diff.T))

def chi_square_distance(vector1, vector2, _covariance_matrix=None):
    """
    Compute the Chi-Square distance between two vectors (e.g., histograms).
    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.
    Returns:
        float: Chi-Square distance.
    """
    epsilon = 1e-10  # Small constant to prevent division by zero
    return np.sum(((vector1 - vector2) ** 2) / (vector1 + vector2 + epsilon))
