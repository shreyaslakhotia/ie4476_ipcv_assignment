"""Fixed, label-free Haar approximation features for preprocessed grayscale data."""

import numbers
import numpy as np
import pywt


class HaarDWT:
    """Multilevel 2D Haar LL coefficients with half-sample symmetric extension.

    Input is (N,H,W) scaled grayscale images or (N,H*W) row-major features.
    No fitting, labels, centering, whitening or coefficient rescaling is used.
    Transform returns float64 (N, n_LL_coefficients), flattened row-major.
    At odd axis lengths, symmetric extension repeats the end sample before
    the next filtering/downsampling stage. Dimensions follow PyWavelets.
    """

    wavelet = 'haar'
    mode = 'symmetric'

    def __init__(self, level=1, image_shape=(112, 92)):
        shape = tuple(image_shape)
        if len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1 for v in shape):
            raise ValueError('image_shape must contain two positive integer dimensions (H,W)')
        maximum = pywt.dwtn_max_level(shape, self.wavelet)
        if isinstance(level, bool) or not isinstance(level, numbers.Integral) or not 1 <= level <= maximum:
            raise ValueError(f'level must be an integer in [1, {maximum}] for image_shape={shape}')
        self.level, self.image_shape = int(level), shape

    def approximation(self, X):
        """Return the raw LL arrays (N,H_level,W_level), useful for inspection."""
        if np.iscomplexobj(X):
            raise ValueError('Expected real grayscale data')
        images = np.asarray(X, dtype=np.float64)
        if images.ndim == 2 and images.shape[1] == np.prod(self.image_shape):
            images = images.reshape(len(images), *self.image_shape)
        if images.ndim != 3 or len(images) == 0 or images.shape[1:] != self.image_shape or not np.isfinite(images).all():
            raise ValueError('Expected nonempty finite images/features matching image_shape')
        approximations = np.stack([pywt.wavedec2(image, self.wavelet, mode=self.mode,
                                                level=self.level, axes=(0, 1))[0] for image in images])
        if not np.isfinite(approximations).all():
            raise ValueError('Nonfinite Haar coefficients; check image magnitudes')
        return approximations

    def transform(self, X):
        approximation = self.approximation(X)
        return approximation.reshape(len(approximation), -1)
