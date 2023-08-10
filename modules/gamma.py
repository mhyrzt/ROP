# AGCCPF (Adaptive Gamma Correction Color Preserving Framework)
import numpy as np
from image_enhancement import image_enhancement


def AGCCPF(image: np.ndarray, alpha: float):
    return image_enhancement.IE(image).AGCCPF(alpha)
