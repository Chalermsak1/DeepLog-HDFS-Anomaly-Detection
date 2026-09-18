import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog import DeepLog


def test_model_initialization():
    """Verify LSTM architecture parameters."""
    model = DeepLog(input_size=30, hidden_size=128, output_size=30, num_layers=2)
    assert model.input_size == 30
    assert model.hidden_size == 128
    assert model.num_layers == 2
    assert model.output_size == 30


def test_model_forward_shape():
    """Verify that forward pass produces valid log probabilities of shape (batch_size, output_size)."""
    model = DeepLog(input_size=30, hidden_size=128, output_size=30, num_layers=2)
    model.eval()

    batch_size = 16
    seq_len = 10
    dummy_input = torch.randint(0, 30, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (batch_size, 30), f"Expected shape {(batch_size, 30)}, got {out.shape}"
    # LogSoftmax outputs should exponentiate and sum to approximately 1.0 per sample
    probs = out.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-4), "Probabilities must sum to 1.0"


def test_model_predict_topk():
    """Verify top-k prediction output format and bounds."""
    model = DeepLog(input_size=30, hidden_size=128, output_size=30, num_layers=2)
    model.eval()

    batch_size = 8
    seq_len = 10
    k = 5
    dummy_input = torch.randint(0, 30, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        log_probs = model(dummy_input)
        _, topk = log_probs.topk(k, dim=-1)

    assert topk.shape == (batch_size, k), f"Top-K shape must be {(batch_size, k)}"
    assert (topk >= 0).all() and (topk < 30).all(), "All predicted class indices must fall within vocabulary range [0, 30)"
