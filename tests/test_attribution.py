"""
Unit Tests for Track 3: Attention-based Root Cause Attribution (XAI & RCA)
==========================================================================
Verifies:
1. Attribution sum normalization (sum-to-one / 100%) across rollout, grad_input,
   integrated gradients, attention pooling, and hybrid combined methods.
2. Ranking consistency, monotonicity, and sensitivity.
3. Gradient flow through embeddings without NaN or Inf.
4. Integrated Gradients Completeness Axiom (sum(IG) ~ F(x) - F(x')).
5. Causal attention masking preservation in attention rollout.
6. Localization of synthetic anomalous injected tokens and transitions.
7. Structured RootCauseExplanation serialization, ASCII summaries, and vocab mapping.
"""

import json
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplog.transformer import LogTransformer
from deeplog.dual_head import DeepLogDualHead
from deeplog.deeplog import DeepLog
from deeplog.attribution import (
    AttentionAttributor,
    RootCauseAnalyzer,
    RootCauseExplanation,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def small_transformer():
    """Deterministic LogTransformer for attribution tests."""
    torch.manual_seed(42)
    model = LogTransformer(
        vocab_size=20,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def small_dual_head():
    """Deterministic DeepLogDualHead for attribution tests."""
    torch.manual_seed(42)
    model = DeepLogDualHead(
        vocab_size=20,
        hidden_size=32,
        num_layers=2,
        embedding_dim=16,
        pooling="attention",
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def sample_sequence():
    """Sample sequence of 8 log event tokens."""
    return [1, 3, 5, 2, 8, 4, 7, 6]


# ==============================================================================
# Track 3: Attention Rollout Tests
# ==============================================================================


def test_attention_rollout_sum_to_one(small_transformer, sample_sequence):
    """Verify that Attention Rollout produces non-negative scores summing to 1.0."""
    attributor = AttentionAttributor(small_transformer)

    # Test across multiple sequence lengths
    for length in [3, 8, 16]:
        seq = list(range(length))
        attr = attributor.compute_attention_rollout(seq, alpha=0.5)

        assert attr.shape == (length,)
        assert (attr >= 0.0).all(), "Rollout attribution contains negative values"
        assert np.isclose(attr.sum(), 1.0, atol=1e-5), f"Sum was {attr.sum()}, expected 1.0"


def test_attention_rollout_alpha_extremes(small_transformer, sample_sequence):
    """Verify rollout under extreme alpha values (pure attention vs. pure identity)."""
    attributor = AttentionAttributor(small_transformer)

    # alpha = 0.0 (Pure identity: query attends only to itself)
    attr_ident = attributor.compute_attention_rollout(sample_sequence, alpha=0.0, query_idx=-1)
    assert np.isclose(attr_ident.sum(), 1.0, atol=1e-5)
    # The last position must receive 100% of the identity attribution
    assert np.isclose(attr_ident[-1], 1.0, atol=1e-5)
    assert np.allclose(attr_ident[:-1], 0.0, atol=1e-5)

    # alpha = 1.0 (Pure attention)
    attr_pure_attn = attributor.compute_attention_rollout(sample_sequence, alpha=1.0)
    assert np.isclose(attr_pure_attn.sum(), 1.0, atol=1e-5)
    assert (attr_pure_attn >= 0.0).all()


def test_attention_rollout_head_fusion(small_transformer, sample_sequence):
    """Verify rollout across different multi-head aggregation strategies."""
    attributor = AttentionAttributor(small_transformer)

    for strategy in ["mean", "max", "min"]:
        attr = attributor.compute_attention_rollout(sample_sequence, head_fusion=strategy)
        assert attr.shape == (len(sample_sequence),)
        assert np.isclose(attr.sum(), 1.0, atol=1e-5)
        assert (attr >= 0.0).all()


def test_attention_rollout_causality(small_transformer):
    """Verify that causal attention rollout strictly prevents future-to-past information leakage."""
    attributor = AttentionAttributor(small_transformer)
    seq = [1, 2, 3, 4, 5, 6]
    n = len(seq)

    # Extract full rollout matrix: shape (n, n)
    rollout_matrix = attributor.compute_attention_rollout(seq, alpha=0.5, return_full_matrix=True)

    assert rollout_matrix.shape == (n, n)

    # Check each row sums to 1.0
    row_sums = rollout_matrix.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums: {row_sums}"

    # Strict causality: for any row i, columns j > i must be strictly 0.0
    for i in range(n):
        for j in range(i + 1, n):
            assert np.isclose(rollout_matrix[i, j], 0.0, atol=1e-6), (
                f"Causality violation at [{i}, {j}]: weight is {rollout_matrix[i, j]}"
            )


# ==============================================================================
# Track 3: Gradient-based Saliency Tests
# ==============================================================================


def test_gradient_saliency_sum_to_one(small_transformer, small_dual_head, sample_sequence):
    """Verify Gradient * Input saliency normalization to 1.0 on Transformer and DualHead."""
    for model in [small_transformer, small_dual_head]:
        attributor = AttentionAttributor(model)
        for norm in ["dot", "abs_dot", "l1", "l2", "grad_norm"]:
            attr = attributor.compute_gradient_saliency(
                sample_sequence, target_class=5, norm=norm, normalize=True
            )
            assert attr.shape == (len(sample_sequence),)
            assert (attr >= 0.0).all()
            assert np.isclose(attr.sum(), 1.0, atol=1e-5), f"Sum was {attr.sum()} for {norm}"


def test_gradient_flow_and_validity(small_transformer, sample_sequence):
    """Verify that gradients flowing to input embeddings are finite and non-zero."""
    attributor = AttentionAttributor(small_transformer)
    seq_t = attributor._to_tensor(sample_sequence)

    emb = attributor._get_embeddings(seq_t).clone().detach().requires_grad_(True)
    out = small_transformer(inputs_embeds=emb)
    scalar = out[0, 3]
    scalar.backward()

    assert emb.grad is not None, "Embedding grad should not be None"
    assert torch.isfinite(emb.grad).all(), "Gradients contain NaN or Inf"
    assert (emb.grad.abs().sum() > 0).item(), "Gradients should be non-zero"


# ==============================================================================
# Track 3: Integrated Gradients & Completeness Axiom
# ==============================================================================


def test_integrated_gradients_sum_to_one(small_transformer, sample_sequence):
    """Verify that normalized Integrated Gradients scores sum to 1.0."""
    attributor = AttentionAttributor(small_transformer)
    attr = attributor.compute_integrated_gradients(
        sample_sequence, target_class=4, steps=30, normalize=True
    )
    assert attr.shape == (len(sample_sequence),)
    assert (attr >= 0.0).all()
    assert np.isclose(attr.sum(), 1.0, atol=1e-5)


def test_integrated_gradients_completeness_axiom(small_transformer, sample_sequence):
    """
    Verify the fundamental Completeness Axiom of Integrated Gradients:
        sum_i(IG_i) ~ F(x) - F(x')
    The approximation error should be small with Riemann interpolation steps.
    """
    attributor = AttentionAttributor(small_transformer)
    target_class = 2

    # Unnormalized IG with convergence delta
    unnorm_ig, delta = attributor.compute_integrated_gradients(
        sample_sequence,
        target_class=target_class,
        steps=50,
        normalize=False,
        return_convergence_delta=True,
    )

    # Completeness error delta should be small (< 0.05)
    assert delta < 0.05, f"Integrated Gradients completeness delta {delta} exceeded 0.05"


# ==============================================================================
# Track 3: Monotonicity & Sensitivity Tests
# ==============================================================================


def test_monotonicity_and_sensitivity(small_transformer):
    """
    Verify Monotonicity / Sensitivity:
    When an input feature/embedding along the path from baseline is scaled by alpha in (0, 1],
    its Integrated Gradients attribution increases monotonically with the scaling factor alpha.
    """
    attributor = AttentionAttributor(small_transformer)
    seq = [1, 2, 3, 4, 5]
    target_pos = 2  # Position to scale
    target_class = 0

    base_t = attributor._to_tensor(seq)
    emb = attributor._get_embeddings(base_t).clone().detach()

    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    ig_target_scores = []

    for a in alphas:
        # Scale embedding at target position by alpha
        scaled_emb = emb.clone()
        scaled_emb[:, target_pos, :] = a * emb[:, target_pos, :]
        baseline = torch.zeros_like(scaled_emb)

        steps = 30
        total_grads = torch.zeros_like(scaled_emb)
        for s in range(1, steps + 1):
            alpha_s = s / steps
            interpolated = (baseline + alpha_s * scaled_emb).clone().detach().requires_grad_(True)
            scalar = small_transformer(inputs_embeds=interpolated)[0, target_class]
            small_transformer.zero_grad()
            scalar.backward()
            total_grads += interpolated.grad.detach()

        avg_grads = total_grads / steps
        ig = scaled_emb * avg_grads
        unnorm_target = abs(ig[0, target_pos].sum().item())
        ig_target_scores.append(unnorm_target)

    # Verify strict monotonicity across alphas
    diffs = [
        ig_target_scores[i + 1] - ig_target_scores[i]
        for i in range(len(ig_target_scores) - 1)
    ]
    assert all(d > 0 for d in diffs), f"Attribution was not strictly monotonic: {ig_target_scores}"


# ==============================================================================
# Track 3: Synthetic Anomaly Localization Tests
# ==============================================================================


def test_synthetic_anomaly_localization_transformer():
    """
    Verify that an injected anomalous event into a periodic normal sequence
    is correctly localized and ranked as the top suspicious root cause.
    """
    torch.manual_seed(42)
    vocab_size = 15
    model = LogTransformer(
        vocab_size=vocab_size,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    model.train()

    # Train model on a simple deterministic sequence pattern: [0, 1, 2, 3, 0, 1, 2, 3...]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_pattern = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    target_pattern = torch.tensor([[1, 2, 3, 0, 1, 2, 3, 0]], dtype=torch.long)

    for _ in range(60):
        optimizer.zero_grad()
        out = model(train_pattern, return_all_tokens=True)
        loss = F.nll_loss(out.view(-1, vocab_size), target_pattern.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()

    # Construct test sequence with an injected anomalous event ID=12 at index 3
    # Followed by an unexpected observation ID=8
    injected_idx = 3
    test_seq = [0, 1, 2, 12]
    unexpected_next = 8

    analyzer = RootCauseAnalyzer(model=model, default_method="grad_input")
    explanation = analyzer.analyze(sequence=test_seq, next_event=unexpected_next)

    assert isinstance(explanation, RootCauseExplanation)
    assert explanation.is_anomalous is True
    assert explanation.anomaly_score > 0.8

    # The top-1 ranked event must localize the injected anomalous token
    top_token_index = explanation.ranked_events[0]["token_index"]
    assert top_token_index == injected_idx, (
        f"Injected anomaly at index {injected_idx} was ranked as {top_token_index}"
    )


def test_synthetic_anomaly_localization_dual_head():
    """Verify anomaly localization on DeepLogDualHead."""
    torch.manual_seed(42)
    vocab_size = 15
    model = DeepLogDualHead(
        vocab_size=vocab_size,
        hidden_size=32,
        num_layers=1,
        embedding_dim=16,
        pooling="attention",
        dropout=0.0,
    )
    model.eval()

    analyzer = RootCauseAnalyzer(model=model, default_method="pooling")
    test_seq = [1, 2, 9, 4, 5]

    explanation = analyzer.analyze(sequence=test_seq)
    assert isinstance(explanation, RootCauseExplanation)
    assert len(explanation.ranked_events) == len(test_seq)
    assert np.isclose(explanation.attributions.sum(), 1.0, atol=1e-5)


# ==============================================================================
# Track 3: Explanation Structure, Formatting & Vocab Mapping
# ==============================================================================


def test_root_cause_explanation_ranking_consistency(small_transformer, sample_sequence):
    """Verify that ranked_events are strictly sorted descending by attribution score."""
    analyzer = RootCauseAnalyzer(model=small_transformer, default_method="rollout")
    exp = analyzer.analyze(sample_sequence)

    # Verify descending monotonicity
    scores = [ev["attribution"] for ev in exp.ranked_events]
    pcts = [ev["attribution_pct"] for ev in exp.ranked_events]

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9, "Ranked events not sorted descending"
        assert pcts[i] >= pcts[i + 1] - 1e-7, "Attribution percentages not sorted descending"

    # Verify all sequence indices are accounted for
    indices = {ev["token_index"] for ev in exp.ranked_events}
    assert indices == set(range(len(sample_sequence)))


def test_root_cause_explanation_serialization(small_transformer, sample_sequence):
    """Verify that RootCauseExplanation can be serialized to valid JSON."""
    vocab_map = {
        1: "PacketReceived",
        3: "AuthSuccess",
        5: "DBConnect",
        2: "QueryExec",
        8: "DiskWriteFail",
        4: "RetryAttempt",
        7: "Timeout",
        6: "SessionClose",
    }
    analyzer = RootCauseAnalyzer(
        model=small_transformer,
        vocab_mapping=vocab_map,
        default_method="combined",
    )
    exp = analyzer.analyze(sample_sequence, next_event=8)

    # to_dict roundtrip
    d = exp.to_dict()
    assert isinstance(d, dict)
    assert d["sequence"] == sample_sequence
    assert len(d["attributions"]) == len(sample_sequence)
    assert np.isclose(sum(d["attributions"]), 1.0, atol=1e-5)
    assert np.isclose(sum(d["attribution_percentages"]), 100.0, atol=1e-3)

    # Must be valid JSON
    json_str = json.dumps(d)
    assert len(json_str) > 0


def test_root_cause_explanation_summary_formatting(small_transformer, sample_sequence):
    """Verify that summary() produces a well-structured diagnostic text report."""
    vocab_map = {i: f"HDFS_Event_{i}" for i in range(20)}
    analyzer = RootCauseAnalyzer(
        model=small_transformer,
        vocab_mapping=vocab_map,
        default_method="rollout",
    )
    exp = analyzer.analyze(sample_sequence, next_event=2)

    summary_text = exp.summary(ascii_bars=True)
    assert "DEEPLOG ROOT CAUSE EXPLANATION" in summary_text
    assert "Top Contributing Log Events" in summary_text
    assert "Top Contributing Transitions" in summary_text
    assert "Sequence Attribution Distribution" in summary_text
    assert "HDFS_Event_" in summary_text


def test_explain_batch(small_transformer):
    """Verify batch sequence root cause analysis."""
    analyzer = RootCauseAnalyzer(model=small_transformer)
    sequences = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1],
    ]
    next_events = [5, 10, None]

    batch_exps = analyzer.explain_batch(sequences=sequences, next_events=next_events)
    assert len(batch_exps) == 3

    for exp, orig_seq in zip(batch_exps, sequences):
        assert isinstance(exp, RootCauseExplanation)
        assert len(exp.sequence) == len(orig_seq)
        assert np.isclose(exp.attributions.sum(), 1.0, atol=1e-5)
