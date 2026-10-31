"""Ablation configuration, paired-statistics and training-only reproducibility."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from src.ablations import M_VALUES, analyse, paired_comparison, read_validation_results, validate_m_values, variance_thresholds
from src.partition import DEFAULT_OUTPUT, load_partitions
from src.dimensionality_reduction import PCA, LDA


class AblationUnitTests(unittest.TestCase):
    def test_configuration_and_threshold_handling(self):
        self.assertEqual(validate_m_values(M_VALUES, 200), M_VALUES)
        for values in ((), (39,), (201,), (60, 40), (40, 40), (40.0,), (True,)):
            with self.assertRaises(ValueError):
                validate_m_values(values, 200)
        pca = PCA()
        pca.cumulative_explained_variance_ = np.asarray([.5, .9, .95, .99, 1.0])
        self.assertEqual(variance_thresholds(pca), {.9: 2, .95: 3, .99: 4})
        with self.assertRaises(ValueError):
            variance_thresholds(pca, (1.1,))

    def test_exact_paired_counts_and_no_discordance(self):
        truth = np.ones(5, dtype=int)
        a, b = np.asarray([0, 0, 0, 1, 1]), truth.copy()
        pair = paired_comparison(truth, a, b)
        self.assertEqual(pair, dict(both_correct=2, both_wrong=0, corrected=[0, 1, 2], new_errors=[], pvalue=.25))
        reverse = paired_comparison(truth, b, a)
        self.assertEqual(reverse['pvalue'], pair['pvalue'])
        self.assertEqual(reverse['new_errors'], pair['corrected'])
        self.assertEqual(paired_comparison(truth, a, a)['pvalue'], 1.0)
        with self.assertRaises(ValueError):
            paired_comparison(truth, a[:-1], b)


class AblationIntegrationTests(unittest.TestCase):
    def test_full_grid_training_only_fitting_and_reproducibility(self):
        parts = load_partitions(names=('train', 'validation'))
        self.assertEqual(set(parts), {'train', 'validation'})
        train, validation = parts['train'], parts['validation']
        rows, source = read_validation_results(DEFAULT_OUTPUT)
        pca_fit, lda_fit = PCA.fit, LDA.fit
        calls = []

        def checked_pca(model, X):
            assert_array_equal(X, train['X'])
            calls.append(('PCA', X.shape[1], None))
            return pca_fit(model, X)

        def checked_lda(model, X, y):
            self.assertEqual(len(X), 240)
            assert_array_equal(y, train['y'])
            if X.shape[1] == 10304:
                assert_array_equal(X, train['X'])
            else:
                self.assertEqual(model.ridge, 0)
                self.assertEqual(model.n_components, 39)
            calls.append(('LDA', X.shape[1], model.ridge))
            return lda_fit(model, X, y)

        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(PCA, 'fit', checked_pca), patch.object(LDA, 'fit', checked_lda):
                first = analyse(train, validation, rows, source)
            repeated = analyse(train, validation, rows, source)
        self.assertEqual(calls, [('PCA', 10304, None)] + [('LDA', m, 0) for m in M_VALUES] + [('LDA', 10304, 1e-3)])
        for key in ('variance', 'thresholds', 'ablation', 'selected', 'statistics', 'pairs'):
            self.assertEqual(first[key], repeated[key])
        for name in first['predictions']:
            assert_array_equal(first['predictions'][name], repeated['predictions'][name])
        for m in M_VALUES:
            assert_array_equal(first['ablation_predictions'][m], repeated['ablation_predictions'][m])
            self.assertEqual(first['ablation_predictions'][m].shape, (80,))
        best = min(first['ablation'], key=lambda r: (-r['n_correct'], r['intermediate_pca_dimension']))
        self.assertEqual(first['selected'], best)
        for metrics in first['statistics'].values():
            self.assertEqual(metrics['n_correct'] + metrics['n_incorrect'], 80)
            self.assertLessEqual(metrics['ci_low'], metrics['accuracy'])
            self.assertGreaterEqual(metrics['ci_high'], metrics['accuracy'])
        for m in (40, 60):
            self.assertEqual(next(r['n_correct'] for r in first['ablation'] if r['intermediate_pca_dimension'] == m), 78)
        self.assertEqual(first['ablation'][-1]['n_correct'], 36)


if __name__ == '__main__':
    unittest.main()
