"""
Unit tests for Semantic Embedding and Zero-Shot OOV Template Handling in DeepLog.

Verifies:
1. Canonical HDFS template loading and normalization.
2. Self-contained LogSemanticEncoder (subword n-grams, projection, normalization, determinism, gradient flow).
3. TemplateSimilarityMatcher (nearest neighbor, cosine similarity ranking, raw log matching, zero-shot OOV resolution, clustering).
4. SemanticDeepLog sequence model (continuous forward pass, discrete index lookup, top-k predictions, zero-shot anomaly scoring, multi-objective loss, gradient flow, serialization).
5. End-to-end integration and dynamic template expansion.
"""

import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from deeplog.semantic import (
    load_hdfs_templates,
    normalize_log_text,
    extract_subword_features,
    LogSemanticEncoder,
    SemanticDeepLog,
    TemplateSimilarityMatcher,
    get_default_hdfs_template_path,
)


# ==============================================================================
# 1. HDFS Template Loader and Normalization Tests
# ==============================================================================


def test_load_hdfs_templates_default():
    """Verify loading the canonical 29 HDFS templates from default location."""
    templates = load_hdfs_templates()
    assert isinstance(templates, dict)
    assert len(templates) == 29, f"Expected 29 templates, got {len(templates)}"

    # Verify key templates exist
    assert "E1" in templates
    assert "E8" in templates
    assert "E22" in templates
    assert "E29" in templates

    assert "Adding an already existing block" in templates["E1"]
    assert "PacketResponder" in templates["E8"]
    assert "allocateBlock" in templates["E22"]
    assert "PendingReplicationMonitor timed out block" in templates["E29"]


def test_load_hdfs_templates_invalid_path():
    """Verify FileNotFoundError on nonexistent template CSV path."""
    with pytest.raises(FileNotFoundError):
        load_hdfs_templates("/invalid/path/to/templates.csv")


def test_normalize_log_text():
    """Verify text normalization abstracts IPs, block IDs, numbers, and camelCase words."""
    raw = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    norm = normalize_log_text(raw)
    assert "10.250.19.102" not in norm
    assert "blk_" not in norm
    assert "<param>" in norm

    camel = "PacketResponder allocateBlock addStoredBlock"
    norm_camel = normalize_log_text(camel)
    assert "allocate Block" in norm_camel
    assert "add Stored Block" in norm_camel
    assert "Packet Responder" in norm_camel


def test_extract_subword_features():
    """Verify deterministic hashing and subword feature extraction."""
    text1 = "Receiving block[*]src:[*]dest:[*]"
    feats1 = extract_subword_features(text1, num_buckets=65536)
    feats2 = extract_subword_features(text1, num_buckets=65536)

    assert isinstance(feats1, list)
    assert len(feats1) > 0
    # Deterministic across multiple calls
    assert feats1 == feats2

    # Check bounds
    for idx in feats1:
        assert 0 <= idx < 65536

    # Empty string handling
    empty_feats = extract_subword_features("")
    assert empty_feats == [0]


# ==============================================================================
# 2. LogSemanticEncoder Tests
# ==============================================================================


def test_log_semantic_encoder_init_and_shapes():
    """Verify encoder initialization, forward shapes, and L2 unit normalization."""
    encoder = LogSemanticEncoder(embedding_dim=128, mode="bag")
    assert encoder.embedding_dim == 128
    assert isinstance(encoder, nn.Module)

    # Single string
    emb_single = encoder("PacketResponder for block Interrupted")
    assert emb_single.shape == (1, 128)
    norm = torch.norm(emb_single, p=2, dim=-1).item()
    assert pytest.approx(norm, rel=1e-5) == 1.0

    # Batch of strings
    texts = [
        "Receiving block[*]src:[*]dest:[*]",
        "Received block[*]of size[*]from[*]",
        "Verification succeeded for[*]",
    ]
    emb_batch = encoder(texts)
    assert emb_batch.shape == (3, 128)
    norms = torch.norm(emb_batch, p=2, dim=-1)
    for n in norms.tolist():
        assert pytest.approx(n, rel=1e-5) == 1.0

    # Empty input list
    empty_emb = encoder([])
    assert empty_emb.shape == (0, 128)


def test_log_semantic_encoder_transformer_mode():
    """Verify encoder in transformer self-attention mode."""
    encoder = LogSemanticEncoder(embedding_dim=64, mode="transformer")
    assert encoder.transformer is not None

    emb = encoder(["writeBlock received exception", "Deleting block file"])
    assert emb.shape == (2, 64)
    assert pytest.approx(torch.norm(emb[0], p=2).item(), rel=1e-5) == 1.0


def test_log_semantic_encoder_determinism():
    """Verify deterministic representations for identical input strings."""
    encoder = LogSemanticEncoder(embedding_dim=128)
    encoder.eval()

    text = "BLOCK* NameSystem[*]allocateBlock:[*]"
    emb1 = encoder.encode(text)
    emb2 = encoder.encode(text)

    diff = (emb1 - emb2).abs().max().item()
    assert diff < 1e-6


def test_log_semantic_encoder_gradient_flow():
    """Verify gradient backpropagation through subword embedding bag and projection layers."""
    encoder = LogSemanticEncoder(embedding_dim=64)
    encoder.train()

    texts = [
        "PacketResponder for block Interrupted",
        "BLOCK* NameSystem allocateBlock",
    ]
    embs = encoder(texts, normalize=True)
    # Contrastive MSE loss between the two representations
    loss = (embs[0] - embs[1]).pow(2).sum()
    loss.backward()

    # Check embedding bag gradients
    assert encoder.embedding_bag.weight.grad is not None
    bag_grad_norm = encoder.embedding_bag.weight.grad.norm().item()
    assert bag_grad_norm > 0.0

    # Check projection layer gradients
    for name, param in encoder.projection.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert param.grad.norm().item() > 0.0, f"Zero gradient for {name}"


def test_log_semantic_encoder_semantic_similarity():
    """Verify semantically similar log templates yield higher cosine similarity than unrelated ones."""
    torch.manual_seed(42)
    encoder = LogSemanticEncoder(embedding_dim=128)
    encoder.eval()

    t_receive1 = "Receiving block[*]src:[*]dest:[*]"
    t_receive2 = "Received block[*]src:[*]dest:[*]of size[*]"
    t_name_alloc = "BLOCK* NameSystem[*]allocateBlock:[*]"
    t_timeout = "PendingReplicationMonitor timed out block[*]"

    with torch.no_grad():
        embs = encoder([t_receive1, t_receive2, t_name_alloc, t_timeout])
        sim_related = F.cosine_similarity(embs[0:1], embs[1:2]).item()
        sim_unrelated1 = F.cosine_similarity(embs[0:1], embs[2:3]).item()
        sim_unrelated2 = F.cosine_similarity(embs[0:1], embs[3:4]).item()

    assert sim_related > 0.70, f"Expected high similarity for receive templates, got {sim_related}"
    assert sim_related > sim_unrelated1, "Related templates should be closer than allocateBlock"
    assert sim_related > sim_unrelated2, "Related templates should be closer than timeout"


# ==============================================================================
# 3. TemplateSimilarityMatcher Tests
# ==============================================================================


def test_template_similarity_matcher_known_templates():
    """Verify matching exact templates yields similarity near 1.0 and correct ID."""
    matcher = TemplateSimilarityMatcher()
    assert len(matcher.template_ids) == 29
    assert matcher.template_embeddings.shape == (29, 128)

    # Query with exact template text for E8
    matches = matcher.match("[*]PacketResponder[*]for block[*]Interrupted[*]", top_k=3)
    assert len(matches) == 3
    top_id, top_sim, top_text = matches[0]

    assert top_id == "E8"
    assert top_sim >= 0.999
    assert "Interrupted" in top_text


def test_template_similarity_matcher_raw_log_lines():
    """Verify raw system log messages with variables match the correct canonical template."""
    matcher = TemplateSimilarityMatcher()

    # Raw log for E5
    raw_e5 = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    matches_e5 = matcher.match(raw_e5, top_k=1)
    assert matches_e5[0][0] == "E5"
    assert matches_e5[0][1] > 0.85

    # Raw log for E8
    raw_e8 = "PacketResponder 2 for block blk_3886505538524818829 Interrupted"
    matches_e8 = matcher.match(raw_e8, top_k=1)
    assert matches_e8[0][0] == "E8"
    assert matches_e8[0][1] > 0.85

    # Raw log for E22
    raw_e22 = "BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_0001_m_000000_0/part-00000. blk_12345"
    matches_e22 = matcher.match(raw_e22, top_k=1)
    assert matches_e22[0][0] == "E22"
    assert matches_e22[0][1] > 0.80


def test_template_similarity_matcher_batch():
    """Verify batch matching matches individual query outputs."""
    matcher = TemplateSimilarityMatcher()
    queries = [
        "PacketResponder 2 for block blk_123 Interrupted",
        "Receiving block blk_456 src: /10.0.0.1:5000 dest: /10.0.0.2:5000",
    ]

    batch_res = matcher.match_batch(queries, top_k=2)
    assert len(batch_res) == 2
    assert batch_res[0][0][0] == "E8"
    assert batch_res[1][0][0] == "E5"


def test_template_similarity_matcher_zero_shot_oov_resolution():
    """Verify out-of-vocabulary detection and dynamic template addition."""
    matcher = TemplateSimilarityMatcher()

    # Unseen template variation: PacketResponder restarting
    unseen_log = "PacketResponder 5 for block blk_8888 restarting after network glitch"
    resolution = matcher.resolve_template(unseen_log, threshold=0.75)

    # Should detect as OOV
    assert resolution["is_oov"] is True
    assert resolution["matched_id"] is None
    # Nearest template should still be semantically related (PacketResponder)
    assert resolution["nearest_id"] in ("E8", "E11", "E10")

    # Dynamically register the new template
    matcher.add_template(
        template_id="E30",
        template_text="[*]PacketResponder[*]for block[*]restarting[*]",
    )
    assert len(matcher.template_ids) == 30
    assert matcher.template_embeddings.shape[0] == 30

    # Querying again should now match E30 with high confidence
    resolution_after = matcher.resolve_template(unseen_log, threshold=0.75)
    assert resolution_after["is_oov"] is False
    assert resolution_after["matched_id"] == "E30"
    assert resolution_after["similarity"] > 0.80


def test_template_similarity_matcher_clustering():
    """Verify spherical KMeans template clustering and centroid nearest-neighbor assignment."""
    matcher = TemplateSimilarityMatcher()
    clusters = matcher.cluster_templates(n_clusters=5, random_state=42)

    assert len(clusters) == 5
    total_clustered = sum(len(c) for c in clusters.values())
    assert total_clustered == 29

    # Check centroid norms
    assert matcher.cluster_centroids.shape == (5, 128)
    for c_vec in matcher.cluster_centroids:
        assert pytest.approx(np.linalg.norm(c_vec), rel=1e-5) == 1.0

    # Nearest cluster query
    c_id, sim = matcher.find_nearest_cluster("PacketResponder 1 for block blk_100 Interrupted")
    assert 0 <= c_id < 5
    assert sim > 0.5


def test_template_similarity_matcher_pairwise_matrix():
    """Verify pairwise similarity matrix properties."""
    matcher = TemplateSimilarityMatcher()
    ids, sim_mat = matcher.pairwise_similarity_matrix()

    assert len(ids) == 29
    assert sim_mat.shape == (29, 29)
    # Diagonal should be 1.0
    for i in range(29):
        assert pytest.approx(sim_mat[i, i], abs=1e-5) == 1.0
    # Symmetric
    diff = np.abs(sim_mat - sim_mat.T).max()
    assert diff < 1e-5


# ==============================================================================
# 4. SemanticDeepLog Sequence Model Tests
# ==============================================================================


def test_semantic_deeplog_init_and_template_registration():
    """Verify SemanticDeepLog initialization and template bank registration."""
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, num_layers=2)
    assert model.embedding_dim == 128
    assert model.hidden_size == 64
    assert model.template_bank is None

    # Register HDFS templates
    templates = load_hdfs_templates()
    model.register_templates(templates)

    assert model.template_bank is not None
    assert model.template_bank.shape == (29, 128)
    assert len(model.template_ids) == 29
    assert len(model.template_texts) == 29


def test_semantic_deeplog_forward_continuous():
    """Verify forward pass given continuous sequence embeddings."""
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, num_layers=2)
    model.register_templates(load_hdfs_templates())

    batch_size = 4
    seq_len = 10
    X_cont = torch.randn(batch_size, seq_len, 128)

    # 1. Forward returning log_probs over candidate templates
    log_probs = model(X_cont)
    assert log_probs.shape == (batch_size, 29)
    # Check probabilities sum to 1.0
    probs = torch.exp(log_probs)
    for p_sum in probs.sum(dim=-1).tolist():
        assert pytest.approx(p_sum, rel=1e-4) == 1.0

    # 2. Forward returning both log_probs and predicted embeddings
    log_probs, pred_emb = model(X_cont, return_embeddings=True)
    assert log_probs.shape == (batch_size, 29)
    assert pred_emb.shape == (batch_size, 128)
    # Check predicted embeddings have unit norm
    for n in torch.norm(pred_emb, p=2, dim=-1).tolist():
        assert pytest.approx(n, rel=1e-5) == 1.0


def test_semantic_deeplog_forward_discrete_indices():
    """Verify forward pass given discrete integer template indices."""
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, num_layers=2)
    model.register_templates(load_hdfs_templates())

    batch_size = 8
    seq_len = 10
    X_idx = torch.randint(0, 29, (batch_size, seq_len), dtype=torch.long)

    log_probs = model(X_idx)
    assert log_probs.shape == (batch_size, 29)


def test_semantic_deeplog_predict_topk():
    """Verify predict returns top-k indices and valid confidence scores."""
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, num_layers=2)
    model.register_templates(load_hdfs_templates())

    X = torch.randn(2, 5, 128)
    topk_idx, topk_conf = model.predict(X, k=3)

    assert topk_idx.shape == (2, 3)
    assert topk_conf.shape == (2, 3)
    assert (topk_conf >= 0.0).all() and (topk_conf <= 1.0).all()


def test_semantic_deeplog_zero_shot_prediction():
    """Verify zero-shot semantic prediction with arbitrary candidate sets."""
    encoder = LogSemanticEncoder(embedding_dim=128)
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, encoder=encoder)

    # Candidate set including standard and novel unseen templates
    candidate_texts = [
        "Receiving block[*]src:[*]dest:[*]",
        "PacketResponder[*]for block[*]Interrupted[*]",
        "Novel failure mode[*]secondary mirror unreachable[*]",
    ]
    candidate_embs = encoder.encode(candidate_texts)
    candidate_ids = ["C1", "C2", "C_UNSEEN"]

    X = torch.randn(2, 8, 128)
    top_ids, top_scores = model.predict_semantic(
        X,
        candidate_embeddings=candidate_embs,
        candidate_ids=candidate_ids,
        k=2,
    )

    assert len(top_ids) == 2
    assert len(top_ids[0]) == 2
    assert top_scores.shape == (2, 2)


def test_semantic_deeplog_anomaly_scoring():
    """Verify anomaly detection for normal and abnormal event transitions."""
    encoder = LogSemanticEncoder(embedding_dim=128)
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, encoder=encoder)
    model.register_templates(load_hdfs_templates())

    prefix = [
        "Receiving block[*]src:[*]dest:[*]",
        "Received block[*]src:[*]dest:[*]of size[*]",
    ]

    # Evaluate a normal subsequent transition
    res_normal = model.score_anomaly(prefix, next_event="E6", threshold=0.3, top_k=9)
    assert "is_anomaly" in res_normal
    assert "similarity" in res_normal
    assert "semantic_distance" in res_normal
    assert res_normal["similarity"] + res_normal["semantic_distance"] == pytest.approx(1.0, rel=1e-5)

    # Evaluate an unseen out-of-vocabulary event
    unseen_event = "Disk corrupted during block write operation"
    res_oov = model.score_anomaly(prefix, next_event=unseen_event, threshold=0.7, top_k=3)
    assert res_oov["is_unseen_oov"] is True


def test_semantic_deeplog_score_sequence():
    """Verify windowed sequence anomaly evaluation."""
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64)
    model.register_templates(load_hdfs_templates())

    # Create sequence of length 15
    seq = [f"E{(i % 29) + 1}" for i in range(15)]
    summary = model.score_sequence(seq, window_size=5, threshold=0.5, top_k=9)

    assert "is_anomaly" in summary
    assert "anomaly_count" in summary
    assert "anomaly_ratio" in summary
    assert summary["evaluated_steps"] == 10
    assert len(summary["step_details"]) == 10


def test_semantic_deeplog_gradient_flow_and_losses():
    """Verify gradient backpropagation across all supported loss formulations."""
    model = SemanticDeepLog(embedding_dim=64, hidden_size=32, num_layers=1)
    model.register_templates(load_hdfs_templates())
    model.train()

    X = torch.randn(4, 6, 64)
    y_idx = torch.tensor([0, 5, 10, 28], dtype=torch.long)
    y_emb = torch.randn(4, 64)

    # 1. Cosine loss
    loss_cos = model.compute_loss(X, y_emb, loss_type="cosine")
    loss_cos.backward()
    assert model.rnn.weight_ih_l0.grad is not None
    assert model.rnn.weight_ih_l0.grad.norm().item() > 0.0

    model.zero_grad()

    # 2. Cross-entropy loss over template bank
    loss_ce = model.compute_loss(X, y_idx, loss_type="ce")
    loss_ce.backward()
    assert model.prediction_head[0].weight.grad is not None
    assert model.prediction_head[0].weight.grad.norm().item() > 0.0

    model.zero_grad()

    # 3. Hybrid loss
    loss_hybrid = model.compute_loss(X, y_idx, loss_type="hybrid", alpha=0.6)
    loss_hybrid.backward()
    assert model.prediction_head[0].weight.grad.norm().item() > 0.0


def test_semantic_deeplog_serialization():
    """Verify model checkpoint saving and loading preserving template bank."""
    model = SemanticDeepLog(embedding_dim=64, hidden_size=32)
    model.register_templates(load_hdfs_templates())
    model.eval()

    X = torch.randn(2, 5, 64)
    with torch.no_grad():
        out_orig = model(X)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        loaded_model = SemanticDeepLog.load(tmp_path)
        loaded_model.eval()

        assert loaded_model.embedding_dim == 64
        assert loaded_model.hidden_size == 32
        assert len(loaded_model.template_ids) == 29

        with torch.no_grad():
            out_loaded = loaded_model(X)

        diff = (out_orig - out_loaded).abs().max().item()
        assert diff < 1e-6
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# 5. End-to-End Pipeline Integration Test
# ==============================================================================


def test_end_to_end_matcher_deeplog_integration():
    """
    Test full zero-shot log processing pipeline:
    1. Matcher indexes HDFS templates.
    2. Syncs codebook to SemanticDeepLog.
    3. Processes raw log stream into continuous space.
    4. Dynamically registers an unseen OOV template on the fly.
    """
    encoder = LogSemanticEncoder(embedding_dim=128)
    matcher = TemplateSimilarityMatcher(encoder=encoder)
    model = SemanticDeepLog(embedding_dim=128, hidden_size=64, encoder=encoder)
    matcher.sync_with_model(model)

    assert model.template_bank.shape == (29, 128)

    # Process a raw log line
    raw_log = "PacketResponder 3 for block blk_987654321 Interrupted"
    res = matcher.resolve_template(raw_log, threshold=0.8)
    assert res["is_oov"] is False
    assert res["matched_id"] == "E8"

    # New unseen failure log
    oov_log = "PacketResponder 3 for block blk_987654321 restarting after cluster failover"
    oov_res = matcher.resolve_template(oov_log, threshold=0.75)
    assert oov_res["is_oov"] is True

    # Register unseen template in both matcher and model
    matcher.add_template("E30", "[*]PacketResponder[*]for block[*]restarting[*]")
    matcher.sync_with_model(model)

    assert len(model.template_ids) == 30
    assert model.template_bank.shape == (30, 128)

    # Query again: now cleanly resolved
    resolved = matcher.resolve_template(oov_log, threshold=0.75)
    assert resolved["is_oov"] is False
    assert resolved["matched_id"] == "E30"
