from __future__ import annotations

import numpy as np

from lap.verifiers.cover.protocol import build_bootstrap_indices
from lap.verifiers.cover.protocol import build_phrase_manifest
from lap.verifiers.cover.protocol import select_training_phrase


def test_phrase_manifest_has_exact_approved_bank():
    manifest = build_phrase_manifest()
    assert len(manifest["phrases"]) == 16
    assert manifest["phrases"][0] == "reach to the hole and insert the circular peg"
    assert manifest["phrases"][-1] == "place the square peg into the hole"
    assert len(set(manifest["phrases"])) == 16


def test_language_selection_is_deterministic_and_shape_specific():
    first = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    second = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    assert first == second
    assert "circular" in first


def test_bootstrap_indices_use_fixed_shape_and_seed():
    indices = build_bootstrap_indices([f"episode_{i}" for i in range(8)], replicates=4)
    expected = np.array(
        [[7, 2, 0, 5, 5, 7, 7, 6], [2, 7, 3, 1, 5, 0, 5, 0], [7, 1, 1, 3, 6, 1, 1, 2], [5, 1, 4, 6, 0, 3, 5, 6]]
    )
    assert indices.shape == (4, 8)
    assert np.array_equal(indices, expected)
