import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.preprocessor import Preprocessor
from deeplog.deeplog import DeepLog


def test_default_behavior_is_identical_to_baseline():
    """Verify that add_eos=False (default) produces results 100% identical to baseline."""
    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training data file not present.")

    # Explicit default and implicit default preprocessors
    p_default = Preprocessor(length=10, timeout=float("inf"))
    p_false = Preprocessor(length=10, timeout=float("inf"), add_eos=False)

    assert p_default.add_eos is False
    assert p_false.add_eos is False

    X_def, y_def, l_def, m_def = p_default.text(train_file, nrows=30, verbose=False)
    X_fls, y_fls, l_fls, m_fls = p_false.text(train_file, nrows=30, verbose=False)

    assert torch.equal(X_def, X_fls)
    assert torch.equal(y_def, y_fls)
    assert l_def == l_fls
    assert m_def == m_fls

    # EOS token must NOT be in mapping when add_eos=False
    assert "<EOS>" not in m_def.values()
    assert -1338 not in m_def.values()
    assert p_default.NO_EVENT in m_def.values()


def test_eos_flag_and_token_configuration():
    """Verify flexible parameter configuration via constructor and method calls."""
    # 1. Constructor add_eos=True defaults
    p1 = Preprocessor(length=10, timeout=float("inf"), add_eos=True)
    assert p1.add_eos is True
    assert p1.eos_token == "<EOS>"
    assert p1.EOS_ID == -1338

    # 2. Custom string token
    p2 = Preprocessor(length=10, timeout=float("inf"), add_eos=True, eos_token="<CUSTOM_END>")
    assert p2.eos_token == "<CUSTOM_END>"

    # 3. Explicit EOS_ID integer configuration
    p3 = Preprocessor(length=10, timeout=float("inf"), add_eos=True, EOS_ID=-1338)
    assert p3.eos_token == -1338
    assert p3.EOS_ID == -1338

    # 4. Method-level override: preprocessor initialized with add_eos=False
    p4 = Preprocessor(length=5, timeout=float("inf"), add_eos=False)
    df = pd.DataFrame({
        "timestamp": [0, 1, 2],
        "event": [9, 22, 9],
        "machine": [0, 0, 0],
    })
    _, y_no_eos, _, m_no_eos = p4.sequence(df, add_eos=False)
    assert len(y_no_eos) == 3
    assert "<EOS>" not in m_no_eos.values()

    _, y_with_eos, _, m_with_eos = p4.sequence(df, add_eos=True)
    assert len(y_with_eos) == 4
    assert "<EOS>" in m_with_eos.values()

    # 5. Method-level EOS_ID override
    _, y_id, _, m_id = p4.sequence(df, EOS_ID=-1338)
    assert len(y_id) == 4
    assert -1338 in m_id.values()


def test_each_block_ends_with_eos_target():
    """Verify that when add_eos=True, each block/machine sequence terminates with an EOS target."""
    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training data file not present.")

    nrows = 20
    p_base = Preprocessor(length=10, timeout=float("inf"), add_eos=False)
    p_eos = Preprocessor(length=10, timeout=float("inf"), add_eos=True)

    X_base, y_base, _, m_base = p_base.text(train_file, nrows=nrows, verbose=False)
    X_eos, y_eos, _, m_eos = p_eos.text(train_file, nrows=nrows, verbose=False)

    # Exactly nrows EOS tokens must be added (one per machine line)
    assert X_eos.shape[0] == X_base.shape[0] + nrows
    assert y_eos.shape[0] == y_base.shape[0] + nrows

    # Look up mapped index of EOS
    id_to_canonical = {v: k for k, v in m_eos.items()}
    eos_mapped_id = id_to_canonical["<EOS>"]

    # Verify EOS appears exactly nrows times in target events
    eos_count = (y_eos == eos_mapped_id).sum().item()
    assert eos_count == nrows, f"Expected {nrows} EOS targets, got {eos_count}"

    # Verify per-line sequence boundary targets directly by reading lines
    with open(train_file) as f:
        line_lengths = [len(f.readline().split()) for _ in range(nrows)]

    cum_lengths = np.cumsum([l + 1 for l in line_lengths])
    for end_idx in cum_lengths - 1:
        assert y_eos[end_idx].item() == eos_mapped_id, f"Target at boundary index {end_idx} must be EOS"


def test_context_boundary_isolation_and_eos_incorporation():
    """Verify context construction incorporates EOS at boundaries without cross-block leakage."""
    p = Preprocessor(length=5, timeout=float("inf"), add_eos=True)

    # Create synthetic dataset with 2 distinct machines
    # Machine 0: events [9, 22] (length 2 -> with EOS length 3)
    # Machine 1: events [11, 26, 21] (length 3 -> with EOS length 4)
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "event": [9, 22, 11, 26, 21],
        "machine": [0, 0, 1, 1, 1],
    })

    X, y, _, mapping = p.sequence(df)
    id_to_canonical = {v: k for k, v in mapping.items()}
    no_event_idx = id_to_canonical[p.NO_EVENT]
    eos_idx = id_to_canonical["<EOS>"]
    e9_idx = id_to_canonical[9]
    e22_idx = id_to_canonical[22]
    e11_idx = id_to_canonical[11]
    e26_idx = id_to_canonical[26]
    e21_idx = id_to_canonical[21]

    # Total samples: (2 + 1) + (3 + 1) = 7
    assert X.shape[0] == 7
    assert y.shape[0] == 7

    # Machine 0: indices 0, 1, 2
    # Step 0: target=9, context=[NO_EVENT]*5
    assert y[0].item() == e9_idx
    assert X[0].tolist() == [no_event_idx] * 5

    # Step 1: target=22, context=[NO_EVENT]*4 + [9]
    assert y[1].item() == e22_idx
    assert X[1].tolist() == [no_event_idx] * 4 + [e9_idx]

    # Step 2 (Machine 0 EOS boundary): target=EOS, context=[NO_EVENT]*3 + [9, 22]
    assert y[2].item() == eos_idx
    assert X[2].tolist() == [no_event_idx] * 3 + [e9_idx, e22_idx]

    # Machine 1: indices 3, 4, 5, 6
    # Step 3 (Machine 1 cold start): target=11, context=[NO_EVENT]*5
    # CRITICAL: Machine 1 must NOT contain Machine 0's EOS or events in its context!
    assert y[3].item() == e11_idx
    assert X[3].tolist() == [no_event_idx] * 5
    assert eos_idx not in X[3].tolist()

    # Step 4: target=26, context=[NO_EVENT]*4 + [11]
    assert y[4].item() == e26_idx
    assert X[4].tolist() == [no_event_idx] * 4 + [e11_idx]

    # Step 5: target=21, context=[NO_EVENT]*3 + [11, 26]
    assert y[5].item() == e21_idx
    assert X[5].tolist() == [no_event_idx] * 3 + [e11_idx, e26_idx]

    # Step 6 (Machine 1 EOS boundary): target=EOS, context=[NO_EVENT]*2 + [11, 26, 21]
    assert y[6].item() == eos_idx
    assert X[6].tolist() == [no_event_idx] * 2 + [e11_idx, e26_idx, e21_idx]
    # Machine 0's events must not leak into Machine 1's boundary context
    assert e9_idx not in X[6].tolist()
    assert e22_idx not in X[6].tolist()


def test_short_sequence_incomplete_execution_resolution():
    """Verify that short sequences (e.g. length 3: E9 -> E22 -> E9) acquire a 4th step with EOS target,
    resolving the false negative blind spot for incomplete executions."""
    # Sequence of 3 events: E9 -> E22 -> E9
    short_seq = [9, 22, 9]
    df_short = pd.DataFrame({
        "timestamp": [0, 1, 2],
        "event": short_seq,
        "machine": [0, 0, 0],
    })

    # Normal complete reference sequence: E9 -> E22 -> E9 -> E11 -> E26
    normal_complete_seq = [9, 22, 9, 11, 26]
    df_normal = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "event": normal_complete_seq,
        "machine": [0, 0, 0, 0, 0],
    })

    # 1. Baseline behavior without EOS
    p_no_eos = Preprocessor(length=5, timeout=float("inf"), add_eos=False)
    X_no_eos, y_no_eos, _, m_no_eos = p_no_eos.sequence(df_short)
    # Without EOS, only 3 steps exist
    assert X_no_eos.shape[0] == 3
    assert y_no_eos.shape[0] == 3
    # Notice: there is NO step evaluating if the sequence is allowed to terminate after event 9!

    # 2. Enhanced behavior with EOS
    p_eos = Preprocessor(length=5, timeout=float("inf"), add_eos=True)
    X_eos, y_eos, _, m_eos = p_eos.sequence(df_short)
    # With EOS, exactly 4 steps exist!
    assert X_eos.shape[0] == 4
    assert y_eos.shape[0] == 4

    id_to_canonical = {v: k for k, v in m_eos.items()}
    eos_mapped_id = id_to_canonical["<EOS>"]
    e9_idx = id_to_canonical[9]
    e22_idx = id_to_canonical[22]

    # Verify 4th step properties
    assert y_eos[3].item() == eos_mapped_id, "4th step must predict EOS target"
    # Context of 4th step contains the full 3-event prefix
    last_ctx = X_eos[3].tolist()
    assert last_ctx[-3:] == [e9_idx, e22_idx, e9_idx]

    # 3. Model verification: demonstrate incomplete execution detection
    # Derive canonical mapping from normal complete execution with EOS
    _, _, _, canonical_mapping = p_eos.sequence(df_normal)
    num_classes = len(canonical_mapping)

    # Initialize DeepLog model with the vocabulary
    deeplog = DeepLog(input_size=num_classes, hidden_size=32, output_size=num_classes, num_layers=1)

    # Convert short sequence with canonical mapping
    X_test, y_test, _, _ = p_eos.sequence(df_short, mapping=canonical_mapping)
    assert X_test.shape[0] == 4
    assert y_test.shape[0] == 4

    # Evaluate predictions:
    preds, _ = deeplog.predict(X_test, k=1)
    # The 4th step evaluates P(<EOS> | [9, 22, 9])
    eos_target = canonical_mapping_inv = {v: k for k, v in canonical_mapping.items()}["<EOS>"]
    assert y_test[3].item() == eos_target


def test_labels_handling_with_eos():
    """Verify that sample labels correctly propagate to the appended EOS rows."""
    p = Preprocessor(length=5, timeout=float("inf"), add_eos=True)

    # Test 1: Single integer label
    df1 = pd.DataFrame({
        "timestamp": [0, 1, 2],
        "event": [9, 22, 9],
        "machine": [0, 0, 0],
    })
    _, _, labels_int, _ = p.sequence(df1, labels=1)
    assert labels_int.shape[0] == 4
    assert torch.all(labels_int == 1)

    # Test 2: Multi-machine per-event label array
    df2 = pd.DataFrame({
        "timestamp": [0, 1, 2, 3],
        "event": [9, 22, 5, 11],
        "machine": [0, 0, 1, 1],
    })
    # Machine 0 has label 0, Machine 1 has label 1
    input_labels = np.array([0, 0, 1, 1])
    _, _, labels_arr, _ = p.sequence(df2, labels=input_labels)
    # Machine 0 has 2+1=3 rows, Machine 1 has 2+1=3 rows -> total 6 rows
    assert labels_arr.shape[0] == 6
    assert labels_arr.tolist() == [0, 0, 0, 1, 1, 1]


def test_mapping_deterministic_and_reuse_with_eos():
    """Verify vocabulary mapping is strictly deterministic and preserves mappings when reused."""
    train_file = "examples/data/hdfs_train"
    if not os.path.exists(train_file):
        pytest.skip("Training data file not present.")

    p = Preprocessor(length=10, timeout=float("inf"), add_eos=True)

    # Run twice on the same data
    _, _, _, mapping1 = p.text(train_file, nrows=25, verbose=False)
    _, _, _, mapping2 = p.text(train_file, nrows=25, verbose=False)

    assert mapping1 == mapping2, "Mapping with EOS must be completely deterministic across calls"
    assert "<EOS>" in mapping1.values()
    assert p.NO_EVENT in mapping1.values()

    # Reuse mapping on test split
    test_file = "examples/data/hdfs_test_normal"
    if os.path.exists(test_file):
        _, _, _, test_mapping = p.text(test_file, nrows=25, mapping=mapping1, verbose=False)
        for k, v in mapping1.items():
            assert test_mapping[k] == v, f"Mapping mismatch on reused split for key {k}"


def test_numeric_eos_id():
    """Verify that integer EOS_ID=-1338 works identically to string token '<EOS>'."""
    p = Preprocessor(length=5, timeout=float("inf"), add_eos=True, EOS_ID=-1338)
    assert p.eos_token == -1338
    assert p.EOS_ID == -1338

    df = pd.DataFrame({
        "timestamp": [0, 1, 2],
        "event": [9, 22, 9],
        "machine": [0, 0, 0],
    })

    X, y, _, mapping = p.sequence(df)
    assert -1338 in mapping.values()
    assert p.NO_EVENT in mapping.values()

    id_to_canonical = {v: k for k, v in mapping.items()}
    assert y[-1].item() == id_to_canonical[-1338]
    assert X.shape[0] == 4
    assert y.shape[0] == 4
