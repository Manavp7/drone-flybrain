import unittest

from experiments.evaluate_people_checks import match_people


class MatchingTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_inflate_recalled_people(self):
        box = [0,0,10,10]
        self.assertEqual(len(match_people([box], [box,box])), 1)

    def test_augmenting_path_recovers_match_a_greedy_choice_would_lose(self):
        people = [[0,0,10,10], [4,0,14,10]]
        predictions = [[2,0,12,10], [0,0,7,10]]
        pairs = match_people(people,predictions)
        self.assertEqual({(p['annotation_index'],p['prediction_index']) for p in pairs}, {(0,1),(1,0)})

    def test_empty_or_disjoint_detections_are_misses(self):
        self.assertEqual(match_people([[0,0,10,10]], []), [])
        self.assertEqual(match_people([[0,0,10,10]], [[20,20,30,30]]), [])

    def test_invalid_boxes_or_threshold_are_rejected(self):
        with self.assertRaises(ValueError):
            match_people([[0,0,float('nan'),10]], [[0,0,10,10]])
        with self.assertRaises(ValueError):
            match_people([], [], threshold=0)


if __name__ == '__main__':
    unittest.main()
