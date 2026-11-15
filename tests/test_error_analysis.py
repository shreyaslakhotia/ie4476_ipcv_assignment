"""Focused checks for neighbor ranks, selection preservation and test isolation."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image

from src.error_analysis import selected_configurations, load_review_partitions, analyse, neighbors
from src.partition import DEFAULT_ROOT, DEFAULT_OUTPUT


class ErrorAnalysisTests(unittest.TestCase):
    def test_stable_neighbor_ranks_and_correct_class(self):
        training = np.asarray([[-1.], [1.], [2.]])
        result = neighbors(training, [0.], np.asarray([1, 2, 1]), true_label=2)
        self.assertEqual(result, dict(nearest=0, correct=1, wrong=0, nearest_distance=1.,
                                     correct_distance=1., wrong_distance=1., correct_rank=2))

    def test_selected_predictions_reproduce_without_opening_test_images(self):
        split_file = DEFAULT_OUTPUT/'split_indices.json'
        saved = json.loads(split_file.read_text(encoding='utf-8'))
        forbidden = {(DEFAULT_ROOT/path).resolve() for path in saved['partitions']['test']['paths']}
        configurations, metadata = selected_configurations(DEFAULT_OUTPUT)
        self.assertEqual(configurations['PCA']['d'], 50)
        self.assertEqual(configurations['PCA→LDA']['m'], 40)
        self.assertEqual(configurations['DWT']['level'], 3)
        self.assertEqual(configurations['Supervised SPCA']['eta'], 2)
        original_open = Image.open
        opened = []

        def checked_open(path, *args, **kwargs):
            self.assertNotIn(path.resolve(), forbidden)
            opened.append(path.resolve())
            return original_open(path, *args, **kwargs)

        with patch.object(Image, 'open', checked_open):
            parts = load_review_partitions(DEFAULT_ROOT, split_file, metadata)
        self.assertEqual(len(opened), 320)
        self.assertEqual(len(set(opened)), 320)
        self.assertEqual(set(parts), {'train', 'validation'})
        with contextlib.redirect_stdout(io.StringIO()):
            first = analyse(parts, configurations, metadata, DEFAULT_OUTPUT)
            repeated = analyse(parts, configurations, metadata, DEFAULT_OUTPUT)
        for name in first['predictions']:
            assert_array_equal(first['predictions'][name], repeated['predictions'][name])
        self.assertEqual(first['details'], repeated['details'])
        self.assertEqual(first['top'], ['PCA', 'SPCA control', 'Supervised SPCA'])
        self.assertEqual(first['next_group'], ['Direct ridge LDA', 'PCA→LDA', 'DWT'])
        self.assertEqual([str(parts['validation']['paths'][i]) for i in first['illustrated']], ['s5/10.pgm', 's1/10.pgm'])
        self.assertEqual([str(parts['validation']['paths'][i]) for i in first['errors']['PCA']], ['s5/10.pgm'])
        self.assertEqual(first['focal'], 'Direct ridge LDA')


if __name__ == '__main__':
    unittest.main()
