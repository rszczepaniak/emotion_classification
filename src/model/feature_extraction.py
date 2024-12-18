import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def build_gabor_kernels(ksize):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 8):  # Orientations (e.g., 8 orientations)
        for sigma in (1, 3, 5):  # Scale of the filter
            for lamda in np.pi / np.array([4, 8]):  # Wavelength
                for gamma in (0.5, 0.8):  # Aspect ratio
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels


def apply_gabor_filters(image, kernels):
    features = []
    for kernel in kernels:
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        features.append(filtered_img.mean())  # Mean as a feature
        features.append(filtered_img.var())  # Variance as a feature
    return np.array(features)


def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract LBP features from an image.
    Args:
        image: Input image as a 2D array (grayscale).
        radius: Radius of the circular LBP pattern.
        n_points: Number of points to consider in the LBP pattern.
    Returns:
        LBP histogram (1D feature vector).
    """
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    # Compute the histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize to ensure sum of histogram is 1
    return hist


def euclidean_distance(vector1, vector2, _covariance_matrix):
    """
    Compute the Euclidean distance between two vectors.
    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.
    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((vector1 - vector2) ** 2))


def mahalanobis_distance(vector1, vector2, covariance_matrix):
    """
    Compute the Mahalanobis distance between two vectors.
    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.
        covariance_matrix (np.ndarray): Covariance matrix of the dataset.
    Returns:
        float: Mahalanobis distance.
    """
    diff = vector1 - vector2
    inv_covmat = np.linalg.inv(covariance_matrix)
    return np.sqrt(np.dot(np.dot(diff.T, inv_covmat), diff))


def chi_square_distance(vector1, vector2, _covariance_matrix):
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
