import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.transformer import LogTransformer
from deeplog.dual_head import DeepLogDualHead


# ==============================================================================
# Track 3: LogTransformer Tests
# ==============================================================================


def test_log_transformer_init():
    """Verify LogTransformer architecture parameter initialization and aliases."""
    model = LogTransformer(
        vocab_size=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_len=256,
        pos_encoding="sinusoidal",
    )
    assert model.vocab_size == 32
    assert model.input_size == 32
    assert model.output_size == 32
    assert model.d_model == 64
    assert model.nhead == 4
    assert model.num_layers == 2
    assert len(model.layers) == 2

    # Verify input_size alias
    model_alias = LogTransformer(input_size=40, d_model=128)
    assert model_alias.vocab_size == 40
    assert model_alias.output_size == 40


def test_log_transformer_forward_shapes():
    """Verify LogTransformer forward pass produces valid log probabilities of expected shapes."""
    batch_size = 16
    seq_len = 10
    vocab_size = 30
    model = LogTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2)
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        # Standard DeepLog-compatible next-event prediction: (batch_size, vocab_size)
        out = model(dummy_input)

    assert out.shape == (batch_size, vocab_size), f"Expected {(batch_size, vocab_size)}, got {out.shape}"

    # Probabilities must sum to 1.0 per sample
    probs = out.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-4), "Probabilities must sum to 1.0"

    # All-tokens forward pass for full sequence training: (batch_size, seq_len, vocab_size)
    with torch.no_grad():
        all_tokens_out = model(dummy_input, return_all_tokens=True)

    assert all_tokens_out.shape == (batch_size, seq_len, vocab_size)
    all_probs = all_tokens_out.exp()
    all_sums = all_probs.sum(dim=-1)
    assert torch.allclose(all_sums, torch.ones(batch_size, seq_len), atol=1e-4)

    # 1D single sequence handling
    single_input = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
    with torch.no_grad():
        single_out = model(single_input)
    assert single_out.shape == (1, vocab_size)


def test_log_transformer_causal_masking():
    """Verify causal attention: position t only attends to positions <= t and ignores future tokens."""
    torch.manual_seed(42)
    vocab_size = 20
    seq_len = 8
    model = LogTransformer(vocab_size=vocab_size, d_model=64, nhead=2, num_layers=2, dropout=0.0)
    model.eval()

    # Create two sequences that are IDENTICAL for the first 5 steps, but DIFFER at steps 5..7
    prefix_len = 5
    seq1 = torch.randint(0, vocab_size, (1, seq_len))
    seq2 = seq1.clone()
    seq2[0, prefix_len:] = (seq1[0, prefix_len:] + 1) % vocab_size

    with torch.no_grad():
        out1 = model(seq1, return_all_tokens=True)
        out2 = model(seq2, return_all_tokens=True)

    # Predictions for positions 0 .. prefix_len - 1 must be strictly identical
    prefix_diff = (out1[:, :prefix_len, :] - out2[:, :prefix_len, :]).abs().max()
    assert prefix_diff < 1e-5, f"Causal leakage detected: prefix difference is {prefix_diff}"

    # Future positions where inputs diverged should produce different outputs
    future_diff = (out1[:, prefix_len:, :] - out2[:, prefix_len:, :]).abs().max()
    assert future_diff > 1e-4, "Future tokens should produce different representations"


def test_log_transformer_predict_topk():
    """Verify top-k predictions, confidence bounds, and ordering."""
    vocab_size = 25
    model = LogTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2)
    model.eval()

    batch_size = 8
    seq_len = 12
    k = 5
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    result, confidence = model.predict(dummy_input, k=k)

    # Shape checks
    assert result.shape == (batch_size, k), f"Expected result shape {(batch_size, k)}, got {result.shape}"
    assert confidence.shape == (batch_size, k), f"Expected conf shape {(batch_size, k)}, got {confidence.shape}"

    # Value ranges
    assert (result >= 0).all() and (result < vocab_size).all(), "Indices must be in [0, vocab_size)"
    assert (confidence >= 0.0).all() and (confidence <= 1.0).all(), "Confidences must be in [0.0, 1.0]"

    # Monotonicity: confidence must be sorted descending across top-k
    diffs = confidence[:, :-1] - confidence[:, 1:]
    assert (diffs >= -1e-6).all(), "Confidence values must be sorted descending"


def test_log_transformer_backward_gradient_flow():
    """Verify backward pass and gradient flow through all transformer layers without NaN."""
    torch.manual_seed(42)
    vocab_size = 30
    model = LogTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, dropout=0.0)
    model.train()

    batch_size = 8
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    log_probs = model(dummy_input)
    loss = F.nll_loss(log_probs, targets)

    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
    loss.backward()

    # Verify that all trainable parameters received valid, finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

    optimizer.step()


def test_log_transformer_save_load_roundtrip(tmp_path):
    """Verify model save and load roundtrip reproduces exact predictions."""
    vocab_size = 28
    model = LogTransformer(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pos_encoding="learned",
    )
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (4, 10), dtype=torch.long)
    with torch.no_grad():
        expected_out = model(dummy_input)

    # 1. Roundtrip with full checkpoint metadata
    checkpoint_path = str(tmp_path / "log_transformer.pt")
    model.save(checkpoint_path)

    loaded_model = LogTransformer.load(checkpoint_path)
    loaded_model.eval()

    with torch.no_grad():
        loaded_out = loaded_model(dummy_input)

    assert torch.allclose(expected_out, loaded_out, atol=1e-5), "Loaded model output does not match original"

    # 2. Roundtrip from raw state_dict
    raw_path = str(tmp_path / "raw_transformer.pt")
    torch.save(model.state_dict(), raw_path)

    raw_loaded = LogTransformer.load(raw_path)
    raw_loaded.eval()

    with torch.no_grad():
        raw_out = raw_loaded(dummy_input)

    assert torch.allclose(expected_out, raw_out, atol=1e-5), "Raw state_dict loaded model does not match"


# ==============================================================================
# Track 3: DeepLogDualHead Tests
# ==============================================================================


def test_dual_head_init():
    """Verify DeepLogDualHead architecture initialization and parameter setup."""
    model = DeepLogDualHead(
        vocab_size=35,
        hidden_size=128,
        num_layers=2,
        embedding_dim=64,
        pooling="attention",
    )
    assert model.vocab_size == 35
    assert model.input_size == 35
    assert model.output_size == 35
    assert model.hidden_size == 128
    assert model.num_layers == 2
    assert model.embedding_dim == 64
    assert model.pooling_type == "attention"
    assert model.pooler is not None

    # Verify input_size alias
    model_alias = DeepLogDualHead(input_size=40, hidden_size=64)
    assert model_alias.vocab_size == 40


def test_dual_head_forward_shapes():
    """Verify DeepLogDualHead forward pass produces valid token log-probs and sequence anomaly probs."""
    batch_size = 16
    seq_len = 10
    vocab_size = 30
    model = DeepLogDualHead(vocab_size=vocab_size, hidden_size=128, num_layers=2)
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        token_log_probs, sequence_anomaly_prob = model(dummy_input)

    # Shape checks
    assert token_log_probs.shape == (batch_size, vocab_size), f"Expected {(batch_size, vocab_size)}, got {token_log_probs.shape}"
    assert sequence_anomaly_prob.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {sequence_anomaly_prob.shape}"

    # Token log-probabilities must exponentiate and sum to 1.0
    token_probs = token_log_probs.exp()
    assert torch.allclose(token_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    # Sequence anomaly probability must be bounded in [0.0, 1.0]
    assert (sequence_anomaly_prob >= 0.0).all() and (sequence_anomaly_prob <= 1.0).all()

    # 1D single sequence handling
    single_input = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
    with torch.no_grad():
        s_token_lp, s_seq_prob = model(single_input)
    assert s_token_lp.shape == (1, vocab_size)
    assert s_seq_prob.shape == (1, 1)


def test_dual_head_predict():
    """Verify predict methods: Top-K next events and sequence-level anomaly probabilities."""
    vocab_size = 30
    model = DeepLogDualHead(vocab_size=vocab_size, hidden_size=128, num_layers=2)
    model.eval()

    batch_size = 8
    seq_len = 10
    k = 5
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Full dual-head predict
    token_result, token_confidence, seq_anomaly = model.predict(dummy_input, k=k)

    assert token_result.shape == (batch_size, k)
    assert token_confidence.shape == (batch_size, k)
    assert seq_anomaly.shape == (batch_size, 1)

    assert (token_result >= 0).all() and (token_result < vocab_size).all()
    assert (token_confidence >= 0.0).all() and (token_confidence <= 1.0).all()
    assert (seq_anomaly >= 0.0).all() and (seq_anomaly <= 1.0).all()

    # DeepLog-compatible predict_token
    token_topk, token_conf = model.predict_token(dummy_input, k=k)
    assert token_topk.shape == (batch_size, k)
    assert token_conf.shape == (batch_size, k)

    # predict_anomaly
    anomaly_prob = model.predict_anomaly(dummy_input)
    assert anomaly_prob.shape == (batch_size, 1)


def test_dual_head_backward_gradient_flow():
    """Verify multi-task backward pass and gradient flow through LSTM backbone and both heads."""
    torch.manual_seed(42)
    vocab_size = 30
    model = DeepLogDualHead(vocab_size=vocab_size, hidden_size=64, num_layers=2, dropout=0.0)
    model.train()

    batch_size = 8
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    token_targets = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)
    seq_targets = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    token_log_probs, seq_anomaly_prob = model(dummy_input)

    loss_token = F.nll_loss(token_log_probs, token_targets)
    loss_seq = F.binary_cross_entropy(seq_anomaly_prob, seq_targets)
    total_loss = loss_token + loss_seq

    assert torch.isfinite(total_loss), f"Total loss must be finite, got {total_loss.item()}"
    total_loss.backward()

    # Verify all parameters have finite gradients without NaN
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

    optimizer.step()


def test_dual_head_pooling_variants():
    """Verify DeepLogDualHead supports attention, last, and mean pooling options."""
    vocab_size = 20
    seq_len = 8
    dummy_input = torch.randint(0, vocab_size, (4, seq_len), dtype=torch.long)

    for pooling_type in ["attention", "last", "mean"]:
        model = DeepLogDualHead(
            vocab_size=vocab_size,
            hidden_size=64,
            num_layers=1,
            pooling=pooling_type,
        )
        model.eval()
        with torch.no_grad():
            token_lp, seq_prob = model(dummy_input)

        assert token_lp.shape == (4, vocab_size), f"Failed shape check for pooling={pooling_type}"
        assert seq_prob.shape == (4, 1), f"Failed shape check for pooling={pooling_type}"
        assert (seq_prob >= 0.0).all() and (seq_prob <= 1.0).all()


def test_dual_head_save_load_roundtrip(tmp_path):
    """Verify DeepLogDualHead save and load roundtrip produces identical predictions."""
    vocab_size = 32
    model = DeepLogDualHead(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=2,
        embedding_dim=32,
        pooling="attention",
    )
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (4, 10), dtype=torch.long)
    with torch.no_grad():
        expected_token_lp, expected_seq_prob = model(dummy_input)

    # 1. Roundtrip with full checkpoint metadata
    checkpoint_path = str(tmp_path / "dual_head.pt")
    model.save(checkpoint_path)

    loaded_model = DeepLogDualHead.load(checkpoint_path)
    loaded_model.eval()

    with torch.no_grad():
        loaded_token_lp, loaded_seq_prob = loaded_model(dummy_input)

    assert torch.allclose(expected_token_lp, loaded_token_lp, atol=1e-5)
    assert torch.allclose(expected_seq_prob, loaded_seq_prob, atol=1e-5)

    # 2. Roundtrip from raw state_dict
    raw_path = str(tmp_path / "raw_dual_head.pt")
    torch.save(model.state_dict(), raw_path)

    raw_loaded = DeepLogDualHead.load(raw_path)
    raw_loaded.eval()

    with torch.no_grad():
        raw_token_lp, raw_seq_prob = raw_loaded(dummy_input)

    assert torch.allclose(expected_token_lp, raw_token_lp, atol=1e-5)
    assert torch.allclose(expected_seq_prob, raw_seq_prob, atol=1e-5)
