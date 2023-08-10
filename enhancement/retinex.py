import cv2
import numpy as np


class Retinex:
    def __init__(self, image: np.ndarray) -> None:
        self.image = np.copy(image)
        self.original = np.copy(image)

    def get_kernel_size(self, sigma: float) -> int:
        return int(((sigma - 0.8) / 0.15) + 2.0)

    def get_gaussian_blur(self, sigma: float, kernel_size: int = None):
        kernel_size = kernel_size or self.get_kernel_size(sigma)
        separate_kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(separate_kernel, separate_kernel)
        return cv2.filter2D(self.image, -1, kernel)

    def single_scale(self, sigma: float):
        return np.log10(self.image) - np.log10(self.get_gaussian_blur(sigma) + 1.0)

    def normalize(self, image: np.ndarray):
        return cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_GRAY2BGR
        )

    def multi_scale(self, sigmas: list[float] = [15, 80, 250], normalize: bool = False):
        msr = np.zeros_like(self.image)
        for u in sigmas:
            msr += self.single_scale(u) / len(sigmas)
        if normalize:
            return self.normalize(msr)
        return msr

    def create_lut(self, li, hi):
        return np.array(
            [
                0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255))
                for i in np.arange(0, 256)
            ],
            dtype="uint8",
        )

    def contrast_stretch(self, channel, low_count, high_count):
        cum_hist_sum = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            return channel
        return cv2.LUT(channel, self.create_lut(li, hi))

    def get_channels(self):
        if len(self.image.shape) == 2:
            return [self.image]
        return cv2.split(self.image)

    def get_low_high_counts(self, low_per, high_per):
        tot_pix = self.image.shape[1] * self.image.shape[0]
        low_count = tot_pix * low_per / 100
        high_count = tot_pix * (100 - high_per) / 100
        return low_count, high_count

    def color_balance(self, low_per: float, high_per: float):
        lc, hc = self.get_low_high_counts(low_per, high_per)
        balanced = [self.contrast_stretch(ch, lc, hc) for ch in self.get_channels()]
        if len(balanced) == 1:
            return np.squeeze(balanced)
        return cv2.merge(balanced)

    def color_restore(
        self,
        alpha: float,
        beta: float,
    ):
        return beta * (
            np.log10(alpha * self.image)
            - np.log10(np.sum(self.image, axis=2, keepdims=True))
        )

    def multi_scale_color_restoration(
        self,
        sigma_scales: list[float] = [15, 80, 250],
        alpha: float = 125,
        beta: float = 46,
        G: float = 192,
        b: float = -30,
        low_per: float = 1,
        high_per: float = 1,
    ):
        self.image = self.image.astype("float64") + 1.0
        self.image = self.normalize(
            G
            * (
                self.multi_scale(sigma_scales)
                * self.color_restore(sigma_scales, alpha, beta)
                - b
            )
        )
        msrcr = self.color_balance(low_per, high_per)
        self.image = np.copy(self.original)
        return msrcr

    def multi_scale_color_preservation(
        self,
        sigma_scales: list[float] = [15, 80, 250],
        low_per: float = 1,
        high_per: float = 1,
    ):
        self.image = np.sum(self.image, axis=2) / self.image.shape[2] + 1.0
        intensity = np.copy(self.image)
        self.image = self.multi_scale(sigma_scales)
        self.image = self.color_balance(low_per, high_per)
        A = np.min(
            np.array(
                [256.0 / (np.max(self.image, axis=2) + 1.0), self.image / intensity]
            ),
            axis=0,
        )
        msrcp = np.clip(np.expand_dims(A, 2) * self.original, 0.0, 255.0)
        self.image = np.copy(self.original)
        return msrcp
