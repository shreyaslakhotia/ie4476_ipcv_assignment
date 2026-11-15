"""Haar coefficient checks and training/validation-only experiment integration."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from src.ablations import read_validation_results
from src.classification import Euclidean1NN
from src.experiments import evaluate_dwt, DWT_LEVELS
from src.partition import DEFAULT_OUTPUT, load_partitions
from src.wavelets import HaarDWT


def manual_haar(images, level):
    """Independent Haar LL: symmetric end extension, 2×2 sums divided by 2."""
    result = np.asarray(images, dtype=np.float64)
    for _ in range(level):
        result = np.pad(result, ((0, 0), (0, result.shape[1] % 2), (0, result.shape[2] % 2)), mode='symmetric')
        result = (result[:, 0::2, 0::2] + result[:, 0::2, 1::2]
                  + result[:, 1::2, 0::2] + result[:, 1::2, 1::2]) / 2
    return result


class DWTTests(unittest.TestCase):
    def test_coefficients_order_and_symmetric_odd_boundaries(self):
        images = np.arange(2*5*7, dtype=float).reshape(2, 5, 7) / 70
        original = images.copy()
        for level in (1, 2):
            model = HaarDWT(level, image_shape=(5, 7))
            expected = manual_haar(images, level)
            assert_allclose(model.approximation(images), expected, atol=1e-13)
            assert_allclose(model.transform(images.reshape(2, -1)), expected.reshape(2, -1), atol=1e-13)
            assert_array_equal(model.transform(images), model.transform(images))
        assert_array_equal(images, original)

    def test_actual_orl_dimensions_and_display_scaling(self):
        image = np.ones((1, 112, 92)) * .4
        for level, shape in zip(DWT_LEVELS, ((56, 46), (28, 23), (14, 12), (7, 6))):
            model = HaarDWT(level)
            LL = model.approximation(image)
            self.assertEqual(LL.shape, (1, *shape))
            self.assertEqual(model.transform(image).shape, (1, np.prod(shape)))
            assert_allclose(LL, .4*2**level, atol=1e-13)
            assert_allclose(LL/2**level, .4, atol=1e-13)
            self.assertEqual(vars(model), dict(level=level, image_shape=(112, 92)))
        self.assertFalse(hasattr(HaarDWT(), 'fit'))  # No labels or learned parameters.

    def test_invalid_parameters_and_data(self):
        for level in (0, -1, 7, 1.5, True):
            with self.assertRaises(ValueError):
                HaarDWT(level)
        for shape in ((0, 92), (112,), (112, 92, 1), (112.0, 92), (True, 92)):
            with self.assertRaises(ValueError):
                HaarDWT(image_shape=shape)
        for data in (np.zeros((0, 112, 92)), np.zeros((2, 92, 112)), np.ones((1, 20)),
                     np.full((1, 112, 92), np.nan), np.ones((1, 112, 92), dtype=complex)):
            with self.assertRaises(ValueError):
                HaarDWT().transform(data)

    def test_orl_reference_predictions_and_reproducibility(self):
        parts = load_partitions(names=('train', 'validation'))
        self.assertEqual(set(parts), {'train', 'validation'})
        train, validation = parts['train'], parts['validation']
        _, summary = read_validation_results(DEFAULT_OUTPUT)
        fit = Euclidean1NN.fit
        calls = []

        def checked_fit(model, features, labels):
            self.assertEqual(len(features), 240)
            assert_array_equal(labels, train['y'])
            calls.append(features.shape[1])
            return fit(model, features, labels)

        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(Euclidean1NN, 'fit', checked_fit):
                first = evaluate_dwt(train, validation, summary)
            second = evaluate_dwt(train, validation, summary)
        self.assertEqual(calls, [2576, 644, 168, 42])
        self.assertEqual(first['rows'], second['rows'])
        for level in DWT_LEVELS:
            assert_array_equal(first['predictions'][level], second['predictions'][level])
            # Check every real feature against NumPy arithmetic, then evaluate
            # the same features with an independent 1-NN implementation.
            expected_train = manual_haar(train['X'].reshape(-1, 112, 92), level).reshape(240, -1)
            expected_validation = manual_haar(validation['X'].reshape(-1, 112, 92), level).reshape(80, -1)
            assert_allclose(HaarDWT(level).transform(train['X']), expected_train, atol=1e-12)
            assert_allclose(HaarDWT(level).transform(validation['X']), expected_validation, atol=1e-12)
            reference = KNeighborsClassifier(n_neighbors=1, metric='euclidean', algorithm='brute').fit(expected_train, train['y'])
            assert_array_equal(first['predictions'][level], reference.predict(expected_validation))


if __name__ == '__main__':
    unittest.main()
