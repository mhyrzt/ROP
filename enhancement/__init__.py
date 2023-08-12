import cv2
import numpy as np

from .ccgb import CCGB
from .clahe import CLAHE
from .agccpf import AGCCPF
from .retinex import Retinex


def load_image(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def per_pixel_average(image: np.ndarray, b_min=120) -> np.ndarray:
    b = np.sqrt(np.mean(image))
    if b >= b_min:
        return image
    beta = b_min - b
    alpha = b_min / b
    adjusted = alpha * image + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)
