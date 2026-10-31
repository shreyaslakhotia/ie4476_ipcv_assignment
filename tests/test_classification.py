import unittest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from src.classification import Euclidean1NN, classification_metrics
from src.experiments import best_configurations, METHODS


class ClassificationTests(unittest.TestCase):
    def test_matches_sklearn_at_low_and_high_dimension(self):
        rng = np.random.default_rng(123)
        for d in (1, 39, 239, 10304):
            X, V = rng.normal(size=(12, d)), rng.normal(size=(6, d))
            y = np.arange(12) % 4 + 1
            actual = Euclidean1NN().fit(X, y).predict(V)
            reference = KNeighborsClassifier(n_neighbors=1, metric='euclidean', algorithm='brute').fit(X, y).predict(V)
            assert_array_equal(actual, reference)

    def test_ties_use_first_training_row(self):
        X = np.asarray([[-1., 0], [1., 0], [-1., 0]])
        y = np.asarray([9, 1, 3])
        V = np.asarray([[0., 0], [-1., 0]])
        actual = Euclidean1NN().fit(X, y).predict(V)
        assert_array_equal(actual, [9, 9])
        reference = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='euclidean').fit(X, y).predict(V)
        assert_array_equal(actual, reference)

    def test_direct_differences_avoid_large_offset_cancellation(self):
        X = np.asarray([[1e12 + 3], [1e12 + 1]])
        assert_array_equal(Euclidean1NN().fit(X, ['far', 'near']).predict([[1e12]]), ['near'])

    def test_fit_copies_input_and_predict_is_read_only(self):
        X = np.asarray([[0.], [2.]])
        y = np.asarray([1, 2])
        model = Euclidean1NN().fit(X, y)
        X[:] = 100
        y[:] = 10
        assert_array_equal(model.predict([[.1], [1.9]]), [1, 2])
        assert_array_equal(model.predict([[.1], [1.9]]), [1, 2])
        assert_array_equal(model.X_, [[0], [2]])

    def test_metrics_and_config_ties(self):
        metrics = classification_metrics([1, 2, 3, 4], [1, 2, 4, 3])
        self.assertEqual(metrics, dict(accuracy=.5, error_rate=.5, n_correct=2, n_samples=4))
        rows = [dict(method=method, n_correct=70, retained_dimension=d) for method in METHODS for d in (20, 5, 10)]
        self.assertTrue(all(row['retained_dimension'] == 5 for row in best_configurations(rows).values()))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            Euclidean1NN().predict([[1]])
        for X, y in (([], []), ([[np.nan]], [1]), ([[1], [2]], [1]), ([[1j]], [1])):
            with self.assertRaises(ValueError):
                Euclidean1NN().fit(X, y)
        model = Euclidean1NN().fit([[0, 1]], [1])
        with self.assertRaises(ValueError):
            model.predict([[0]])
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([1], [1, 2])


if __name__ == '__main__':
    unittest.main()
