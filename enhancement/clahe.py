# Green Channel
# GrayScale
# RGB
import cv2
import numpy as np


class CLAHE:
    def __init__(
        self, image: np.ndarray, clipLimit: float = ..., tileGridSize: tuple[int] = ...
    ) -> None:
        self.image = np.copy(image)
        self.green = np.copy(self.image[:, :, 1])
        self.grays = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    def update_clahe(self, clipLimit: float, tileGridSize: tuple[int] = ...) -> None:
        self.clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        return self

    def apply_green(self):
        return self.clahe.apply(self.green)

    def apply_all(self):
        image = np.zeros_like(self.image)
        for i in range(image.shape[-1]):
            image[:, :, i] = self.clahe.apply(self.image[:, :, i])
        return image

    def apply_gray(self):
        return self.clahe.apply(self.grays)
