"""Weighted-scatter mathematics, eta=1 control and ORL extension checks."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.ablations import read_validation_results
from src.partition import DEFAULT_OUTPUT, load_partitions
from src.dimensionality_reduction import PCA, SPCA
from src.experiments import evaluate_spca, SPCA_WEIGHTS, PCA_DIMENSIONS


class SPCAMathematicsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        # Unbalanced priors detect unweighted class averaging or nc-1 divisors.
        self.y = np.repeat([1, 2, 3], [5, 9, 14])
        self.X = rng.normal(size=(28, 6)) + rng.normal(size=(3, 6))[self.y-1] * 2

    def test_weighted_scatter_eigenproblem_and_orthogonality(self):
        X, y = self.X, self.y
        mean = X.mean(0)
        Sw, Sb = np.zeros((6, 6)), np.zeros((6, 6))
        for label in np.unique(y):
            group = X[y == label]
            prior, class_mean = len(group)/len(X), group.mean(0)
            Sw += prior * np.cov(group, rowvar=False, bias=True)
            delta = class_mean - mean
            Sb += prior * np.outer(delta, delta)
        assert_allclose(Sw+Sb, np.cov(X, rowvar=False, bias=True), atol=1e-13)
        for eta in (0, 1, 2, 5, 10):
            model = SPCA(4, eta=eta).fit(X, y)
            covariance = Sw + eta*Sb
            values, vectors = np.linalg.eigh(covariance)
            assert_allclose(model.eigenvalues_, values[::-1], rtol=1e-11)
            assert_allclose(model.components_ @ model.components_.T, np.eye(6), atol=1e-13)
            assert_allclose(covariance @ model.components_.T, model.components_.T * model.eigenvalues_, atol=1e-11)
            assert_allclose(model.class_priors_, [5/28, 9/28, 14/28])
            self.assertTrue(np.all(np.diff(model.eigenvalues_) <= 0))
            W = model.components_[:4].T
            assert_allclose(W @ W.T, vectors[:, -4:] @ vectors[:, -4:].T, atol=1e-11)

    def test_eta_one_pca_equivalence(self):
        pca, spca = PCA().fit(self.X), SPCA(eta=1).fit(self.X, self.y)
        assert_allclose(spca.eigenvalues_, pca.explained_variance_ * 27/28, atol=1e-12)
        for d in (1, 2, 4, 6):
            rotation = pca.components_[:d] @ spca.components_[:d].T
            assert_allclose(rotation @ rotation.T, np.eye(d), atol=1e-12)
            assert_allclose(spca.transform(self.X, d), pca.transform(self.X, d) @ rotation, atol=1e-11)

    def test_determinism_immutable_transform_and_fit_transform(self):
        model = SPCA(3, eta=5)
        first = model.fit_transform(self.X, self.y)
        second = SPCA(3, eta=5).fit_transform(self.X, self.y)
        assert_array_equal(first, second)
        snapshot = copy.deepcopy(vars(model))
        probe = np.random.default_rng(30).normal(size=(7, 6)) + 100
        self.assertEqual(model.transform(probe).shape, (7, 3))
        self.assertEqual(model.transform(probe, 1).shape, (7, 1))
        for name, old in snapshot.items():
            if isinstance(old, np.ndarray):
                assert_array_equal(getattr(model, name), old)
            else:
                self.assertEqual(getattr(model, name), old)

    def test_invalid_parameters_inputs_and_rank(self):
        for eta in (-1, np.nan, np.inf, True, '2'):
            with self.assertRaises(ValueError):
                SPCA(eta=eta).fit(self.X, self.y)
        for d in (0, -1, 7, True, 2.5):
            with self.assertRaises(ValueError):
                SPCA(d).fit(self.X, self.y)
        for X, y in ((self.X, self.y[:-1]), (self.X, np.ones(28)),
                     (np.full((28, 6), np.nan), self.y), (self.X, np.full(28, np.nan))):
            with self.assertRaises(ValueError):
                SPCA().fit(X, y)
        with self.assertRaises(ValueError):
            SPCA().transform(self.X)
        with self.assertRaises(ValueError):
            SPCA().fit(self.X, self.y).transform(np.ones((2, 7)))
        X = np.column_stack((np.arange(6.), np.arange(6.) * 2))
        self.assertEqual(SPCA().fit(X, [1, 1, 1, 2, 2, 2]).rank_, 1)
        with self.assertRaises(ValueError):
            SPCA(2).fit(X, [1, 1, 1, 2, 2, 2])


class SPCAIntegrationTests(unittest.TestCase):
    def test_training_only_full_grid_control_and_reproducibility(self):
        parts = load_partitions(names=('train', 'validation'))
        self.assertEqual(set(parts), {'train', 'validation'})
        train, validation = parts['train'], parts['validation']
        rows, summary = read_validation_results(DEFAULT_OUTPUT)
        original_fit, original_pca = SPCA.fit, PCA.fit
        calls = []

        def checked_fit(model, X, y):
            assert_array_equal(X, train['X'])
            assert_array_equal(y, train['y'])
            calls.append(model.eta)
            return original_fit(model, X, y)

        def checked_pca(model, X):
            assert_array_equal(X, train['X'])
            return original_pca(model, X)

        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(SPCA, 'fit', checked_fit), patch.object(PCA, 'fit', checked_pca):
                first = evaluate_spca(train, validation, rows, summary)
            repeated = evaluate_spca(train, validation, rows, summary)
        self.assertEqual(calls, list(SPCA_WEIGHTS))
        self.assertEqual(first['rows'], repeated['rows'])
        self.assertEqual(first['selected'], repeated['selected'])
        self.assertEqual(len(first['predictions']), 64)
        self.assertEqual(len(first['rows']), 80)
        for key in first['predictions']:
            assert_array_equal(first['predictions'][key], repeated['predictions'][key])
        for d in PCA_DIMENSIONS:
            assert_array_equal(first['predictions'][(1, d)], first['control_predictions'][d])
        for eta in SPCA_WEIGHTS:
            assert_array_equal(first['predictions'][(eta, 239)], first['control_predictions'][239])
        best = min((r for r in first['rows'] if r['method'] == 'SPCA'),
                   key=lambda r: (-r['n_correct'], r['retained_dimension'], r['supervision_weight']))
        self.assertEqual(first['selected'], best)
        self.assertTrue(all(rank == 239 for rank in first['ranks'].values()))


if __name__ == '__main__':
    unittest.main()
