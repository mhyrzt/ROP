# Green Channel
# GrayScale
# RGB
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


class CLAHE:
    def __init__(
        self, image: np.ndarray, clipLimit: float, tileGridSize: tuple[int] = None
    ) -> None:
        self.update_clahe(clipLimit, tileGridSize)
        self.image = np.copy(image)
        self.green = np.copy(self.image[:, :, 1])
        self.grays = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def update_clahe(self, clipLimit: float, tileGridSize: tuple[int] = None) -> None:
        if tileGridSize is None:
            self.clahe = cv2.createCLAHE(clipLimit)
        else:
            self.clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        return self

    def apply_green(self):
        return self.clahe.apply(self.green)

    def apply_hsv(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = self.clahe.apply(image[:, :, 2])
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def apply_gray(self):
        return self.clahe.apply(self.grays)
