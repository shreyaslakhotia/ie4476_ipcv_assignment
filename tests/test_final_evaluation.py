"""Final evaluation safeguards using synthetic queries only; never read real test images."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import final_evaluation as final


class FinalEvaluationTests(unittest.TestCase):
    def prepare(self, directory):
        results = Path(directory)
        for name in final.SOURCES:
            shutil.copyfile(final.DEFAULT_OUTPUT/name, results/name)
        return results

    def test_freeze_uses_saved_validation_only_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            results = self.prepare(directory)
            with patch.object(final.Image, 'open', side_effect=AssertionError('No images during freeze')):
                protocol = final.freeze(results)
                self.assertEqual(protocol['configurations'], final.CONFIGURATIONS)
                self.assertEqual(protocol['training_count'], 240)
                self.assertEqual(protocol['configurations'][4]['eta'], 2)
                with self.assertRaises(FileExistsError):
                    final.freeze(results)

    def test_synthetic_run_orders_fits_before_test_and_blocks_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            results = self.prepare(directory)
            final.freeze(results)
            X, y = np.zeros((240, 10304), dtype=np.float32), np.ones(240, dtype=np.int64)
            Q, truth = np.zeros((80, 10304), dtype=np.float32), np.ones(80, dtype=np.int64)
            events, calls = [], []

            class Projection:
                def __init__(self, d):
                    self.d = d

                def transform(self, queries):
                    return np.zeros((len(queries), self.d))

            class Classifier:
                def __init__(self, d):
                    self.d = d

                def predict(self, queries):
                    self_test.assertEqual(events, ['fit_all', 'load_test'])
                    self_test.assertEqual(queries.shape, (80, self.d))
                    calls.append(self.d)
                    return truth.copy()

            self_test = self

            def fit(training, labels, configs):
                self.assertIs(training, X)
                self.assertIs(labels, y)
                self.assertEqual(len(training), 240)
                self.assertEqual(events, [])
                events.append('fit_all')
                return [(None if c['method'] == 'Unreduced' else Projection(c['d']),
                         Classifier(c['d'])) for c in configs]

            def load(*args, **kwargs):
                self.assertEqual(events, ['fit_all'])
                self.assertEqual(kwargs['names'], ('train', 'test'))
                events.append('load_test')
                return dict(train=dict(X=X, y=y),
                            test=dict(X=Q, y=truth, indices=np.arange(80),
                                      paths=np.asarray([f'synthetic/{i}.pgm' for i in range(80)])))

            with patch.object(final, 'load_training', return_value=(X, y)), \
                    patch.object(final, 'fit_models', side_effect=fit), \
                    patch.object(final, 'load_partitions', side_effect=load), \
                    contextlib.redirect_stdout(io.StringIO()):
                final.evaluate(Path(directory), results)
                self.assertEqual(calls, [c['d'] for c in final.CONFIGURATIONS])
                with self.assertRaises(FileExistsError):
                    final.evaluate(Path(directory), results)
            # Reports regenerate without entering any image-loading or fitting path.
            with patch.object(final.Image, 'open', side_effect=AssertionError('No images during render')), \
                    patch.object(final, 'fit_models', side_effect=AssertionError('No fitting during render')):
                rows = final.render(results)
            self.assertTrue(all(r['n_correct'] == 80 and r['n_incorrect'] == 0 for r in rows))
            self.assertEqual(len(calls), 6)
            audit = json.loads((results/'final_test_audit.json').read_text(encoding='utf-8'))
            audit['predictions']['PCA']['labels'][0] = 2
            final.write_json(results/'final_test_audit.json', audit)
            with self.assertRaisesRegex(ValueError, 'prediction record'):
                final.render(results)


if __name__ == '__main__':
    unittest.main()
