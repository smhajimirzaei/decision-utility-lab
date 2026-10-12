"""Check temporal packing against the published execution inverse."""

import unittest

import numpy as np

from trust_imagination.action_groups import pack_actions


class ActionGroupingTests(unittest.TestCase):
    def test_chronological_inverse_and_candidate_separation(self):
        actions = np.linspace(-1, 1, 2 * 10 * 4, dtype=np.float32).reshape(2, 10, 4)
        mean = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        std = np.array([0.5, 0.8, 1.0, 0.9], dtype=np.float32)
        groups = pack_actions(actions, mean, std)
        self.assertEqual(groups.shape, (2, 2, 20))
        for candidate in range(2):
            executed = groups[:, candidate].reshape(10, 4) * std + mean
            np.testing.assert_allclose(executed, actions[candidate], atol=1e-7)
        np.testing.assert_allclose(groups[0, 1, 4:8], (actions[1, 1] - mean) / std)

    def test_refuses_silent_clipping_or_partial_group(self):
        for actions in (
            np.zeros((1, 6, 4)),
            np.full((1, 5, 4), 1.5),
            np.full((1, 5, 4), np.nan),
        ):
            with self.assertRaises(ValueError):
                pack_actions(actions, np.zeros(4), np.ones(4))


if __name__ == "__main__":
    unittest.main()
