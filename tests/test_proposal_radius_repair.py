import numpy as np
import pytest
from trust_imagination.proposal_contract import pool_contract
from trust_imagination.proposal_radius_repair import radius_repair_pool


def test_original_banks_and_prefix_identity_are_byte_preserved():
    raw = np.random.default_rng(18).uniform(-1.2, 1.2, (8, 50, 4)).astype(np.float32)
    before = raw.copy()
    old = pool_contract(raw, 58)
    new = radius_repair_pool(raw, 58)
    np.testing.assert_array_equal(raw, before)
    assert new["logical"][:12].tobytes() == old["logical"].tobytes()
    for bank in ("policy", "mixed"):
        assert (
            new["unique"][new["banks"][bank]].tobytes()
            == old["unique"][old["banks"][bank]].tobytes()
        )
    assert new["banks"]["repair"][:4] == new["banks"]["mixed"][:4]
    for candidate in new["logical"][12:]:
        np.testing.assert_array_equal(candidate[:, 3], new["logical"][0, :, 3])
        assert (
            np.max(np.linalg.norm(candidate[:, :3] - new["logical"][0, :, :3], axis=1))
            <= 0.250001
        )
    assert np.abs(new["logical"]).max() <= 1
    assert new["logical"].tobytes() == radius_repair_pool(raw, 58)["logical"].tobytes()


def test_radius_changes_only_magnitude_before_clipping_and_dedup_keeps_logical_weights():
    raw = np.zeros((8, 50, 4), np.float32)
    d = radius_repair_pool(raw, 59)
    np.testing.assert_allclose(
        d["logical"][12:, :, :3], 2.5 * d["logical"][8:12, :, :3], atol=3e-8
    )
    assert len(d["logical"]) == 16 and len(d["unique"]) == 9
    assert d["banks"]["policy"] == [0] * 8
    assert len(d["banks"]["repair"]) == 8
    assert d["banks"]["repair"][:4] == [0] * 4
    np.testing.assert_array_equal(d["unique"][d["mapping"]], d["logical"])


def test_boundary_clipping_can_eliminate_a_direction_without_removing_a_logical_candidate():
    raw = np.ones((8, 50, 4), np.float32)
    d = radius_repair_pool(raw, 19)
    assert len(d["banks"]["repair"]) == 8
    assert np.abs(d["unique"]).max() <= 1
    assert np.all(d["logical"][12:, :, 3] == 1)
    for raw_bad in [np.zeros((8, 10, 4)), np.full((8, 50, 4), np.nan)]:
        with pytest.raises(ValueError):
            radius_repair_pool(raw_bad, 1)
