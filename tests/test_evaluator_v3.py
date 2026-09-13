import json
import unittest
from stress.evaluate import journal_matches


class JournalTests(unittest.TestCase):
    def test_complete_reordered_json_records_accept_tuple_serialization(self):
        rows = [{"trial_id": 1, "position": (1., 2., 3.), "outcome": "mission_complete"},
                {"trial_id": 2, "position": (4., 5., 6.), "outcome": "collision"}]
        journal = json.loads(json.dumps(list(reversed(rows))))
        self.assertTrue(journal_matches(journal, rows))

    def test_missing_duplicate_or_changed_result_is_rejected(self):
        rows = [{"trial_id": 1, "outcome": "mission_complete"},
                {"trial_id": 2, "outcome": "collision"}]
        self.assertFalse(journal_matches(rows[:1], rows))
        self.assertFalse(journal_matches(rows + rows[:1], rows))
        self.assertFalse(journal_matches([rows[0], {"trial_id": 2, "outcome": "mission_complete"}], rows))
