import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.preprocessor import Preprocessor


def test_preprocessor_shapes():
    """Verify that Preprocessor creates expected context and event tensor shapes."""
    context_length = 10
    preprocessor = Preprocessor(length=context_length, timeout=float("inf"))

    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training file not present.")

    X, y, _, mapping = preprocessor.text(train_file, nrows=50, verbose=False)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == context_length, f"Context length must be {context_length}"
    assert X.shape[0] == y.shape[0], "Number of contexts must equal number of target events"


def test_preprocessor_no_event_padding():
    """Verify that early events in a sequence are padded with NO_EVENT."""
    context_length = 10
    preprocessor = Preprocessor(length=context_length, timeout=float("inf"))

    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training file not present.")

    X, y, _, mapping = preprocessor.text(train_file, nrows=1, verbose=False)

    # Inverted mapping to find index of NO_EVENT
    id_to_canonical = {v: k for k, v in mapping.items()}
    no_event_idx = id_to_canonical[preprocessor.NO_EVENT]

    # For the very first event of a block, its context should be 100% NO_EVENT
    first_ctx = X[0].tolist()
    assert all(idx == no_event_idx for idx in first_ctx), "First event must have all NO_EVENT padding in context."


def test_preprocessor_no_writable_numpy_warning():
    """Verify that Preprocessor executes without PyTorch non-writable NumPy warnings."""
    import warnings
    preprocessor = Preprocessor(length=10, timeout=float("inf"))
    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training file not present.")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        preprocessor.text(train_file, nrows=20, verbose=False)
        for w in record:
            assert "The given NumPy array is not writable" not in str(w.message)

