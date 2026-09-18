import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.preprocessor import Preprocessor


def test_canonical_mapping_deterministic():
    """Verify that mapping derived from the training set is 100% deterministic across calls."""
    preprocessor = Preprocessor(length=10, timeout=float("inf"))
    train_file = "examples/data/hdfs_train"

    if not os.path.exists(train_file):
        pytest.skip("Training file not present.")

    _, _, _, mapping1 = preprocessor.text(train_file, verbose=False)
    _, _, _, mapping2 = preprocessor.text(train_file, verbose=False)

    assert mapping1 == mapping2, "Canonical training mapping must be completely deterministic."
    assert len(mapping1) > 0, "Mapping must contain non-empty event dictionary."
    # Check that NO_EVENT (-1337) is mapped
    assert preprocessor.NO_EVENT in mapping1.values(), "NO_EVENT must be present in canonical mapping."


def test_mapping_reuse_across_splits():
    """Verify that passing the canonical mapping to test data preserves key-to-event mappings."""
    preprocessor = Preprocessor(length=10, timeout=float("inf"))
    train_file = "examples/data/hdfs_train"
    test_file = "examples/data/hdfs_test_normal"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        pytest.skip("Dataset splits not present.")

    _, _, _, train_mapping = preprocessor.text(train_file, verbose=False)
    # Load first 100 lines with training mapping
    _, _, _, test_mapping = preprocessor.text(test_file, nrows=100, mapping=train_mapping, verbose=False)

    # All keys in train_mapping must be identical in test_mapping
    for k, v in train_mapping.items():
        assert test_mapping[k] == v, f"Mapping mismatch for key {k}: {test_mapping[k]} vs {v}"
