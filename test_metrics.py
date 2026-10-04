"""Small checks for error counting and frame-label alignment."""
import unittest
from common import match, metrics, dataset
from compare import score, validate_labels

class MatchingTests(unittest.TestCase):

    def test_duplicate_detection_is_false_positive(self):
        self.assertEqual(match([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], [[0, 0, 10, 10]]), {'tp': 1, 'fp': 1, 'fn': 0})

    def test_wrong_location_counts_both_errors(self):
        self.assertEqual(match([[20, 20, 30, 30, 0.9]], [[0, 0, 10, 10]]), {'tp': 0, 'fp': 1, 'fn': 1})

    def test_one_prediction_cannot_match_two_objects(self):
        self.assertEqual(match([[0, 0, 10, 10, 0.9]], [[0, 0, 10, 10], [1, 1, 9, 9]]), {'tp': 1, 'fp': 0, 'fn': 1})

    def test_empty_predictions_miss_objects(self):
        self.assertEqual(match([], [[0, 0, 10, 10]])['fn'], 1)

    def test_absent_pen_false_alarm(self):
        self.assertEqual(score([[0, 0, 10, 10, 0.9]], {'present': False}, 'presence'), {'tp': 0, 'fp': 1, 'fn': 0, 'tn': 0})

    def test_present_pen_missed(self):
        self.assertEqual(score([], {'present': True}, 'presence')['fn'], 1)

    def test_undefined_precision(self):
        self.assertIsNone(metrics({'tp': 0, 'fp': 0, 'fn': 2})['precision'])

    def test_missing_frame_index_rejected(self):
        with self.assertRaises(ValueError):
            validate_labels({'label_type': 'presence', 'frames': [{'index': 1, 'present': True}]})

    def test_ambiguous_truth_rejected(self):
        with self.assertRaises(ValueError):
            validate_labels({'label_type': 'presence', 'frames': [{'index': 0, 'present': None}]})

    def test_dataset_has_disjoint_author_groups(self):
        _, counts = dataset()
        self.assertEqual(counts, {'low': 16, 'high': 79, 'val': 16, 'test': 29})
if __name__ == '__main__':
    unittest.main()
