"""Training/validation integration, leakage guards and numerical investigation."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.linalg import eigh
from sklearn.neighbors import KNeighborsClassifier

from src.classification import Euclidean1NN
from src.partition import load_partitions
from src.dimensionality_reduction import PCA, LDA
from src.experiments import evaluate, LDA_DIMENSIONS


class ExperimentTests(unittest.TestCase):
    def test_reproducibility_training_only_fits_and_independent_pipeline(self):
        parts = load_partitions(names=('train', 'validation'))
        self.assertEqual(set(parts), {'train', 'validation'})
        train, validation = parts['train'], parts['validation']
        X, y, V = train['X'], train['y'], validation['X']
        calls = []
        original_pca_fit, original_lda_fit = PCA.fit, LDA.fit

        def pca_fit(model, features):
            assert_array_equal(features, X)
            calls.append(('PCA', features.shape))
            return original_pca_fit(model, features)

        def lda_fit(model, features, labels):
            self.assertEqual(len(features), 240)
            assert_array_equal(labels, y)
            if features.shape[1] == 10304:
                assert_array_equal(features, X)
            calls.append(('LDA', features.shape))
            return original_lda_fit(model, features, labels)

        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(PCA, 'fit', pca_fit), patch.object(LDA, 'fit', lda_fit):
                rows, predictions, diagnostics = evaluate(train, validation)
            repeated_rows, repeated_predictions, repeated_diagnostics = evaluate(train, validation)
        self.assertEqual(calls, [('PCA', (240, 10304)), ('LDA', (240, 10304)), ('LDA', (240, 200))])
        self.assertEqual(rows, repeated_rows)
        self.assertEqual(diagnostics, repeated_diagnostics)
        self.assertEqual(len(rows), 37)
        for key in predictions:
            assert_array_equal(predictions[key], repeated_predictions[key])
        # All positive-rank PCs span training differences, preserving 1-NN.
        assert_array_equal(predictions['PCA:239'], predictions['Unreduced:10304'])
        reference = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='euclidean').fit(X, y)
        assert_array_equal(predictions['Unreduced:10304'], reference.predict(V))

        # Investigate m=200 without changing m, regularization, or scaling.
        # Solve the same Fisher problem by an independent dense generalized
        # eigensolver in 200D and compare every already-requested configuration.
        pca = PCA(200).fit(X)
        Z, ZV = pca.transform(X), pca.transform(V)
        means = np.asarray([Z[y == label].mean(0) for label in y])
        residual = Z - means
        Sw = residual.T @ residual / len(X)
        B = np.asarray([np.sqrt(np.mean(y == label)) * (Z[y == label].mean(0) - Z.mean(0))
                        for label in np.unique(y)]).T
        Sb = B @ B.T
        eigenvalues, vectors = eigh(Sb, Sw)
        W = vectors[:, -39:][:, ::-1]
        relative_residual = np.linalg.norm(Sb @ W - (Sw @ W) * eigenvalues[-39:][::-1]) / np.linalg.norm(Sb @ W)
        self.assertLess(relative_residual, 1e-8)
        for d in LDA_DIMENSIONS:
            reference_prediction = Euclidean1NN().fit((Z - Z.mean(0)) @ W[:, :d], y).predict((ZV - Z.mean(0)) @ W[:, :d])
            assert_array_equal(predictions[f'PCA→LDA:{d}'], reference_prediction)
        print(f'\nIndependent PCA(200)→LDA generalized-eigenproblem: all 10 validation predictions arrays match; '
              f'relative residual={relative_residual:.3g}')


if __name__ == '__main__':
    unittest.main()
