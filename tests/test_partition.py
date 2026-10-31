"""Small protocol tests independent of source image files."""

import unittest
import numpy as np
from src.partition import make_split_indices, validate_indices


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(1, 41), 10)
        self.parts = make_split_indices(self.y)

    def test_protocol_and_repeatability(self):
        original = self.y.copy()
        repeated = make_split_indices(self.y, 42)
        np.testing.assert_array_equal(self.y, original)
        for name, count in (("train", 6), ("validation", 2), ("test", 2)):
            np.testing.assert_array_equal(self.parts[name], repeated[name])
            self.assertEqual(len(self.parts[name]), 40 * count)
            for label in range(1, 41):
                self.assertEqual(np.count_nonzero(self.y[self.parts[name]] == label), count)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(self.parts.values()))), np.arange(400))
        np.testing.assert_array_equal(self.parts['train'][:6], [5, 6, 0, 7, 3, 2])
        self.assertFalse(np.array_equal(self.parts['train'], make_split_indices(self.y, 43)['train']))

    def test_overlap_rejected_even_with_correct_class_counts(self):
        self.parts['validation'][0] = self.parts['train'][0]
        with self.assertRaisesRegex(ValueError, 'unique, disjoint'):
            validate_indices(self.y, self.parts)

    def test_out_of_bounds_rejected(self):
        self.parts['test'][0] = 400
        with self.assertRaisesRegex(ValueError, 'bounds'):
            validate_indices(self.y, self.parts)

    def test_wrong_input_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ten samples'):
            make_split_indices(self.y[:-1])

    def test_duplicate_within_partition_rejected(self):
        self.parts['train'][0] = self.parts['train'][1]
        with self.assertRaisesRegex(ValueError, 'unique, disjoint'):
            validate_indices(self.y, self.parts)


if __name__ == '__main__':
    unittest.main()
