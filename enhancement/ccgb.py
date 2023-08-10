import cv2
import numpy as np


def get_mask(image: np.ndarray):
    h, w = image.shape[:2]
    mask = np.zeros_like(image)
    cv2.circle(mask, (w // 2, h // 2), min(w // 2, h // 2), (255, 255, 255), -1)
    return mask


def apply_mask(image: np.ndarray):
    return cv2.bitwise_and(image, get_mask(image))


def apply_gaussian(image: np.ndarray, sigmaX: float):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaX)


def CCGB(
    image: np.ndarray,
    sigmaX: float = 15,
    alpha: float = 5,
    beta: float = -5,
    gamma: float = 128,
) -> np.ndarray:
    # Crop, Circle and Gaussian Blur
    masked = apply_mask(image)
    smoothed = apply_gaussian(masked, sigmaX)
    enhanced = cv2.addWeighted(masked, alpha, smoothed, beta, gamma)
    return enhanced
