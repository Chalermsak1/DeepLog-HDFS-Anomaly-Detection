"""
Semantic Embedding and Zero-Shot Out-of-Vocabulary (OOV) Template Handling for DeepLog.

This module provides continuous semantic vector representations for log templates,
eliminating the out-of-vocabulary (OOV) failure modes of discrete event ID models.

Key Components:
---------------
1. `load_hdfs_templates`: Robust loader for the canonical 29 HDFS log templates.
2. `LogSemanticEncoder`: Self-contained, robust neural semantic encoder utilizing
   character/subword n-grams, word n-grams, and deep orthogonal neural projection
   without external network dependencies.
3. `SemanticDeepLog`: A neural sequence model operating directly on continuous
   semantic embeddings with dynamic template bank support and zero-shot prediction.
4. `TemplateSimilarityMatcher`: High-throughput cosine similarity search, OOV resolution,
   and semantic clustering for log templates and raw log streams.
"""

from __future__ import annotations

import csv
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ==============================================================================
# 1. HDFS Template Loader and Normalization Utilities
# ==============================================================================


def get_default_hdfs_template_path() -> str:
    """Resolve the default file path to HDFS.log_templates.csv."""
    candidates = [
        os.path.join(os.getcwd(), "dataset", "HDFS_v1", "HDFS.log_templates.csv"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset",
                "HDFS_v1",
                "HDFS.log_templates.csv",
            )
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_hdfs_templates(csv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load the 29 canonical HDFS log templates from CSV.

    Parameters
    ----------
    csv_path : str, optional
        Path to HDFS.log_templates.csv. If None, resolves from default workspace location.

    Returns
    -------
    templates : Dict[str, str]
        Dictionary mapping EventId (e.g., 'E1' ... 'E29') to EventTemplate text.
    """
    path = csv_path or get_default_hdfs_template_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HDFS log templates file not found at '{path}'. "
            "Please ensure dataset/HDFS_v1/HDFS.log_templates.csv exists."
        )

    templates: Dict[str, str] = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = row.get("EventId", "").strip()
            event_template = row.get("EventTemplate", "").strip()
            if event_id:
                templates[event_id] = event_template

    return templates


def normalize_log_text(text: str) -> str:
    """
    Normalize log text and templates to abstract away variable instances.

    Abstracts:
    - Wildcard tokens: [*], <*>, * -> <param>
    - IPv4 addresses and port numbers -> <param>
    - Block identifiers: blk_-12345... -> <param>
    - Hexadecimal representations -> <param>
    - Pure numeric literals -> <param>
    - camelCase identifier segmentation (e.g., allocateBlock -> allocate Block)
    """
    if not text:
        return ""

    # Replace wildcard tokens
    t = re.sub(r"\[\*\]", " <param> ", text)
    t = re.sub(r"<\*>", " <param> ", t)

    # Replace Unix / file system paths (e.g. /user/root/rand/...)
    t = re.sub(r"(?:/[a-zA-Z0-9_\-\.]+)+", " <param> ", t)

    # Replace IP addresses with optional ports
    t = re.sub(r"/?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?", " <param> ", t)

    # Replace HDFS block IDs
    t = re.sub(r"blk_-?\d+", " <param> ", t)

    # Replace hex numbers
    t = re.sub(r"0x[0-9a-fA-F]+", " <param> ", t)

    # Replace standalone numbers
    t = re.sub(r"\b\d+\b", " <param> ", t)

    # Split camelCase identifiers into constituent tokens
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

    return t


def extract_subword_features(
    text: str,
    min_n: int = 3,
    max_n: int = 5,
    num_buckets: int = 65536,
) -> List[int]:
    """
    Extract deterministic subword character n-grams, word unigrams, and word bigrams.

    Uses deterministic CRC32 feature hashing across a fixed bucket space,
    guaranteeing 100% out-of-vocabulary coverage without external vocabulary tables.

    Parameters
    ----------
    text : str
        Input raw log line or log template string.
    min_n : int, default=3
        Minimum character n-gram length.
    max_n : int, default=5
        Maximum character n-gram length.
    num_buckets : int, default=65536
        Number of embedding hash buckets.

    Returns
    -------
    indices : List[int]
        Hashed subword and n-gram indices bounded in [0, num_buckets - 1].
    """
    normalized = normalize_log_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_<>]+", normalized.lower())

    if not tokens:
        return [0]

    indices: List[int] = []

    # 1. Word unigrams & character subword n-grams
    for tok in tokens:
        # Word unigram
        w_feat = f"<w:{tok}>".encode("utf-8")
        indices.append(zlib.crc32(w_feat) % num_buckets)

        # Character n-grams bounded with start and end tokens
        bounded_tok = f"<{tok}>"
        tok_len = len(bounded_tok)
        for n in range(min_n, min(max_n + 1, tok_len + 1)):
            for i in range(tok_len - n + 1):
                sub = bounded_tok[i : i + n]
                c_feat = f"<c:{sub}>".encode("utf-8")
                indices.append(zlib.crc32(c_feat) % num_buckets)

    # 2. Word bigrams for local semantic context
    for i in range(len(tokens) - 1):
        bi_feat = f"<bi:{tokens[i]}_{tokens[i + 1]}>".encode("utf-8")
        indices.append(zlib.crc32(bi_feat) % num_buckets)

    return indices


# ==============================================================================
# 2. LogSemanticEncoder
# ==============================================================================


class LogSemanticEncoder(nn.Module):
    """
    Self-contained, robust neural semantic encoder for log templates.

    Maps arbitrary text strings (known or zero-shot unseen) into a continuous
    d-dimensional embedding space. Combines subword/character n-gram feature
    hashing, embedding bag aggregation, and multi-layer neural projection.
    No external network downloads or pretrained models required.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_buckets: int = 65536,
        min_n: int = 3,
        max_n: int = 5,
        mode: str = "bag",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize the LogSemanticEncoder.

        Parameters
        ----------
        embedding_dim : int, default=128
            Dimension of the output continuous embedding space.
        num_buckets : int, default=65536
            Number of hash buckets for the subword embedding bag.
        min_n : int, default=3
            Minimum character n-gram length.
        max_n : int, default=5
            Maximum character n-gram length.
        mode : str, default="bag"
            Encoder backbone architecture:
            - 'bag': FastText-style EmbeddingBag with deep orthogonal projection.
            - 'transformer': EmbeddingBag followed by lightweight self-attention block.
        hidden_dim : int, optional
            Dimension of internal projection layer (defaults to embedding_dim).
        dropout : float, default=0.0
            Dropout probability in projection / attention layers.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.min_n = min_n
        self.max_n = max_n
        self.mode = mode.lower()
        h_dim = hidden_dim or embedding_dim

        # Native PyTorch subword aggregation layer
        self.embedding_bag = nn.EmbeddingBag(num_buckets, embedding_dim, mode="mean")
        nn.init.normal_(self.embedding_bag.weight, mean=0.0, std=0.05)

        # Optional lightweight transformer self-attention block
        if self.mode == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=h_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        else:
            self.transformer = None

        # Multi-layer neural projection with orthogonal initialization
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, h_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, embedding_dim, bias=False),
        )

        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text string(s) into continuous d-dimensional vector representations.

        Parameters
        ----------
        texts : str or Sequence[str]
            A single log string or sequence of log strings.
        normalize : bool, default=True
            Whether to L2-normalize the output embeddings (unit hypersphere).

        Returns
        -------
        embeddings : torch.Tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = list(texts)

        if not input_list:
            device = self.embedding_bag.weight.device
            return torch.empty((0, self.embedding_dim), device=device)

        device = self.embedding_bag.weight.device

        all_indices: List[int] = []
        offsets: List[int] = [0]

        for text in input_list:
            indices = extract_subword_features(
                text,
                min_n=self.min_n,
                max_n=self.max_n,
                num_buckets=self.num_buckets,
            )
            all_indices.extend(indices)
            offsets.append(offsets[-1] + len(indices))

        idx_tensor = torch.tensor(all_indices, dtype=torch.long, device=device)
        off_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)

        # Subword aggregation: (batch_size, embedding_dim)
        pooled = self.embedding_bag(idx_tensor, off_tensor)

        if self.mode == "transformer" and self.transformer is not None:
            # Transformer forward with pseudo-sequence length 1
            pooled = self.transformer(pooled.unsqueeze(1)).squeeze(1)

        projected = self.projection(pooled)

        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convenience inference method. Returns embeddings under no_grad if in eval mode.
        """
        is_single = isinstance(texts, str)
        if not self.training:
            with torch.no_grad():
                out = self.forward(texts, normalize=normalize)
        else:
            out = self.forward(texts, normalize=normalize)

        if is_single and out.size(0) == 1:
            return out
        return out

    def encode_sequence(
        self,
        sequence: Sequence[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode an ordered sequence of log events into a sequence tensor (1, seq_len, d).
        """
        embeddings = self.encode(sequence, normalize=normalize)
        return embeddings.unsqueeze(0)


# ==============================================================================
# 3. SemanticDeepLog
# ==============================================================================


class SemanticDeepLog(nn.Module):
    """
    SemanticDeepLog: Neural Sequence Model for Log Anomaly Detection in Continuous Space.

    Unlike discrete ID DeepLog models, SemanticDeepLog operates directly on continuous
    semantic embeddings, enabling zero-shot generalization to unseen or modified templates.

    Key Features:
    - Sequence backbone: Stacked LSTM / GRU / Transformer modeling event transitions.
    - Continuous Semantic Prediction Head: Predicts the expected continuous next-event vector.
    - Dynamic Template Bank: Stores candidate template embeddings with runtime registration.
    - Zero-Shot Candidate Scoring: Computes cosine similarity against dynamic candidate sets.
    - Multi-Objective Training: Supports Cosine, MSE, Cross-Entropy, and Hybrid losses.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        tau: float = 0.07,
        dropout: float = 0.1,
        encoder: Optional[LogSemanticEncoder] = None,
    ):
        """
        Initialize SemanticDeepLog.

        Parameters
        ----------
        embedding_dim : int, default=128
            Dimension of log semantic embeddings.
        hidden_size : int, default=128
            Dimension of recurrent hidden state.
        num_layers : int, default=2
            Number of recurrent layers.
        rnn_type : str, default="lstm"
            Type of recurrent backbone ('lstm' or 'gru').
        tau : float, default=0.07
            Temperature scaling parameter for softmax logits over candidate templates.
        dropout : float, default=0.1
            Dropout rate in recurrent and projection layers.
        encoder : LogSemanticEncoder, optional
            Reference to semantic encoder for automated string encoding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.tau = tau
        self.dropout_rate = dropout
        self.encoder = encoder or LogSemanticEncoder(embedding_dim=embedding_dim)

        # Sequence Backbone
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        # Continuous Semantic Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
        )

        # Template bank buffer and metadata
        self.register_buffer("template_bank", None)
        self.template_ids: List[str] = []
        self.template_texts: List[str] = []

    def set_template_bank(
        self,
        embeddings: torch.Tensor,
        template_ids: Sequence[str],
        template_texts: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Register a template codebook bank of known templates.

        Parameters
        ----------
        embeddings : torch.Tensor of shape (num_templates, embedding_dim)
            Normalized semantic embeddings of candidate templates.
        template_ids : Sequence[str]
            Identifer strings for each template (e.g. ['E1', 'E2', ...]).
        template_texts : Sequence[str], optional
            Corresponding template text strings.
        """
        norm_bank = F.normalize(embeddings, p=2, dim=-1)
        self.template_bank = norm_bank
        self.template_ids = list(template_ids)
        if template_texts is not None:
            self.template_texts = list(template_texts)
        else:
            self.template_texts = [f"Template {tid}" for tid in self.template_ids]

    def register_templates(
        self,
        templates: Union[Dict[str, str], Sequence[str]],
        encoder: Optional[LogSemanticEncoder] = None,
    ) -> None:
        """
        Encode and register a dictionary or list of templates into the model's template bank.

        Parameters
        ----------
        templates : Dict[str, str] or Sequence[str]
            Dictionary of {template_id: template_text} or list of template texts.
        encoder : LogSemanticEncoder, optional
            Encoder to use (defaults to self.encoder or freshly instantiated encoder).
        """
        enc = encoder or self.encoder or LogSemanticEncoder(embedding_dim=self.embedding_dim)
        if self.encoder is None:
            self.encoder = enc

        if isinstance(templates, dict):
            t_ids = list(templates.keys())
            t_texts = [templates[k] for k in t_ids]
        else:
            t_texts = list(templates)
            t_ids = [f"E{i + 1}" for i in range(len(t_texts))]

        with torch.no_grad():
            embeddings = enc.encode(t_texts, normalize=True)

        self.set_template_bank(embeddings, t_ids, t_texts)

    def add_template(
        self,
        template_id: str,
        template_text: Union[str, torch.Tensor],
        encoder: Optional[LogSemanticEncoder] = None,
    ) -> None:
        """
        Dynamically add an unseen log template to the template bank at runtime (zero-shot continuous learning).
        """
        if isinstance(template_text, str):
            enc = encoder or self.encoder or LogSemanticEncoder(embedding_dim=self.embedding_dim)
            with torch.no_grad():
                new_emb = enc.encode(template_text, normalize=True)
            text_str = template_text
        else:
            new_emb = F.normalize(template_text, p=2, dim=-1)
            if new_emb.dim() == 1:
                new_emb = new_emb.unsqueeze(0)
            text_str = f"Template {template_id}"

        if self.template_bank is None or self.template_bank.size(0) == 0:
            self.set_template_bank(new_emb, [template_id], [text_str])
        else:
            updated_bank = torch.cat([self.template_bank, new_emb.to(self.template_bank.device)], dim=0)
            self.template_bank = updated_bank
            self.template_ids.append(template_id)
            self.template_texts.append(text_str)

    def _prepare_input(self, X: Union[torch.Tensor, Sequence[Union[str, Sequence[str]]]]) -> torch.Tensor:
        """
        Convert diverse input formats (continuous tensors, discrete indices, or text strings)
        into a continuous tensor of shape (batch_size, seq_len, embedding_dim).
        """
        enc = self.encoder or LogSemanticEncoder(embedding_dim=self.embedding_dim)

        if isinstance(X, (list, tuple)):
            if len(X) == 0:
                device = next(self.parameters()).device
                return torch.empty((0, 0, self.embedding_dim), device=device)

            # Single sequence of strings (e.g. ['E1', 'E2'] or ['Receiving...', 'Received...'])
            if isinstance(X[0], str):
                if self.template_bank is not None and all(s in self.template_ids for s in X):
                    indices = [self.template_ids.index(s) for s in X]
                    idx_t = torch.tensor(indices, dtype=torch.long, device=self.template_bank.device)
                    return F.embedding(idx_t.unsqueeze(0), self.template_bank)
                else:
                    embs = enc.encode(X, normalize=True)
                    device = next(self.parameters()).device
                    return embs.to(device).unsqueeze(0)

            # Batch of sequences: Sequence[Sequence[str]]
            batch_tensors: List[torch.Tensor] = []
            for seq in X:
                if isinstance(seq, str):
                    seq = [seq]
                if self.template_bank is not None and all(s in self.template_ids for s in seq):
                    indices = [self.template_ids.index(s) for s in seq]
                    idx_t = torch.tensor(indices, dtype=torch.long, device=self.template_bank.device)
                    seq_embs = F.embedding(idx_t, self.template_bank)
                else:
                    device = next(self.parameters()).device
                    seq_embs = enc.encode(seq, normalize=True).to(device)
                batch_tensors.append(seq_embs)
            return torch.stack(batch_tensors, dim=0)

        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Unsupported input type for X: {type(X)}")

        # Single sequence of shape (seq_len, embedding_dim) -> add batch dimension
        if X.dim() == 2 and X.size(-1) == self.embedding_dim and X.dtype in (torch.float32, torch.float64):
            return X.unsqueeze(0)

        # Discrete indices of shape (batch_size, seq_len)
        if X.dim() == 2 and X.dtype in (torch.int32, torch.int64, torch.long):
            if self.template_bank is None:
                raise ValueError(
                    "Discrete integer indices passed to SemanticDeepLog but no template_bank has been registered."
                )
            return F.embedding(X.to(torch.long), self.template_bank)

        # Single discrete sequence of shape (seq_len,)
        if X.dim() == 1 and X.dtype in (torch.int32, torch.int64, torch.long):
            if self.template_bank is None:
                raise ValueError(
                    "Discrete integer indices passed to SemanticDeepLog but no template_bank has been registered."
                )
            return F.embedding(X.unsqueeze(0).to(torch.long), self.template_bank)

        if X.dim() != 3 or X.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected continuous input X of shape (batch_size, seq_len, {self.embedding_dim}), got {X.shape}"
            )

        return X

    def forward(
        self,
        X: Union[torch.Tensor, Sequence[Sequence[str]]],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass predicting next-event semantic representation and template bank logits.

        Parameters
        ----------
        X : torch.Tensor or Sequence[Sequence[str]]
            Continuous input sequences of shape (batch_size, seq_len, embedding_dim),
            discrete index sequences (batch_size, seq_len), or nested text lists.
        return_embeddings : bool, default=False
            If True, returns a tuple of (logits_or_log_probs, predicted_next_embedding).

        Returns
        -------
        output : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If template_bank is set and return_embeddings=False:
                log_probs over template bank of shape (batch_size, num_templates)
            If template_bank is set and return_embeddings=True:
                (log_probs, predicted_next_embedding)
            If template_bank is None:
                predicted_next_embedding of shape (batch_size, embedding_dim)
        """
        X_cont = self._prepare_input(X)

        rnn_out, _ = self.rnn(X_cont)
        last_hidden = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Predict continuous next-event embedding
        pred_emb = self.prediction_head(last_hidden)
        pred_emb = F.normalize(pred_emb, p=2, dim=-1)

        if self.template_bank is not None and self.template_bank.size(0) > 0:
            # Cosine similarity logits scaled by temperature: (batch_size, num_templates)
            logits = torch.matmul(pred_emb, self.template_bank.T) / self.tau
            log_probs = F.log_softmax(logits, dim=-1)

            if return_embeddings:
                return log_probs, pred_emb
            return log_probs

        return pred_emb

    def predict_next_embedding(
        self,
        X: Union[torch.Tensor, Sequence[Sequence[str]]],
    ) -> torch.Tensor:
        """
        Generate continuous predicted next-event embedding vectors.

        Returns
        -------
        pred_emb : torch.Tensor of shape (batch_size, embedding_dim)
        """
        if self.template_bank is not None:
            _, pred_emb = self.forward(X, return_embeddings=True)
            return pred_emb
        return self.forward(X, return_embeddings=False)

    def predict(
        self,
        X: Union[torch.Tensor, Sequence[Sequence[str]]],
        k: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DeepLog-compatible Top-K prediction method over registered template bank.

        Parameters
        ----------
        X : torch.Tensor
            Input sequences.
        k : int, default=1
            Number of top candidates to return.

        Returns
        -------
        topk_indices : torch.Tensor of shape (batch_size, k)
        topk_probabilities : torch.Tensor of shape (batch_size, k)
        """
        if self.template_bank is None or self.template_bank.size(0) == 0:
            raise ValueError("Template bank is empty. Register templates prior to calling predict().")

        k_eff = min(k, self.template_bank.size(0))
        log_probs = self.forward(X, return_embeddings=False)
        probs = torch.exp(log_probs)
        confidences, indices = probs.topk(k_eff, dim=-1)
        return indices, confidences

    def predict_semantic(
        self,
        X: Union[torch.Tensor, Sequence[Sequence[str]]],
        candidate_embeddings: Optional[torch.Tensor] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        k: int = 1,
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Zero-shot candidate prediction against an arbitrary candidate set.

        Parameters
        ----------
        X : torch.Tensor or text sequences
            Input prefix sequence(s).
        candidate_embeddings : torch.Tensor, optional
            Candidate template embeddings of shape (num_candidates, embedding_dim).
            If None, defaults to the registered template_bank.
        candidate_ids : Sequence[str], optional
            Identifiers for candidate templates.
        k : int, default=1
            Number of candidates to rank.

        Returns
        -------
        topk_ids : List[List[str]]
            Predicted template IDs for each batch element.
        topk_scores : torch.Tensor
            Cosine similarities corresponding to the top predictions.
        """
        pred_emb = self.predict_next_embedding(X)

        if candidate_embeddings is not None:
            candidates = F.normalize(candidate_embeddings, p=2, dim=-1).to(pred_emb.device)
            c_ids = list(candidate_ids) if candidate_ids is not None else [f"C{i}" for i in range(candidates.size(0))]
        elif self.template_bank is not None:
            candidates = self.template_bank
            c_ids = self.template_ids
        else:
            raise ValueError("No candidate embeddings or template bank available for semantic prediction.")

        k_eff = min(k, candidates.size(0))
        similarities = torch.matmul(pred_emb, candidates.T)  # Shape: (batch_size, num_candidates)
        topk_scores, topk_indices = similarities.topk(k_eff, dim=-1)

        topk_ids: List[List[str]] = []
        for row in topk_indices.tolist():
            topk_ids.append([c_ids[idx] for idx in row])

        return topk_ids, topk_scores

    def score_anomaly(
        self,
        prefix: Union[torch.Tensor, Sequence[str]],
        next_event: Union[str, int, torch.Tensor],
        threshold: float = 0.5,
        top_k: int = 9,
    ) -> Dict[str, Any]:
        """
        Evaluate an event transition for anomalous behavior in continuous space.

        Handles both known templates and zero-shot unseen templates seamlessly.

        Parameters
        ----------
        prefix : torch.Tensor or Sequence[str]
            History sequence of events prior to the next event.
        next_event : str, int, or torch.Tensor
            Observed next event to verify (raw text, template ID/index, or embedding).
        threshold : float, default=0.5
            Cosine similarity threshold below which an event is flagged anomalous.
        top_k : int, default=9
            Top-K candidate acceptance window.

        Returns
        -------
        result : Dict[str, Any]
            Diagnostic information including anomaly flag, cosine similarity,
            semantic distance, top-K candidates, and whether the event is an unseen OOV.
        """
        self.eval()
        with torch.no_grad():
            pred_emb = self.predict_next_embedding(prefix)  # (1, embedding_dim)

            # Resolve target embedding
            is_unseen = False
            target_text = ""

            if isinstance(next_event, str):
                target_text = next_event
                # Check if it matches a known ID directly
                if next_event in self.template_ids:
                    idx = self.template_ids.index(next_event)
                    target_emb = self.template_bank[idx : idx + 1]
                else:
                    enc = self.encoder or LogSemanticEncoder(embedding_dim=self.embedding_dim)
                    target_emb = enc.encode(next_event, normalize=True)
                    is_unseen = True
            elif isinstance(next_event, int):
                if self.template_bank is None or next_event >= self.template_bank.size(0):
                    raise IndexError(f"Template index {next_event} out of bounds for template bank.")
                target_emb = self.template_bank[next_event : next_event + 1]
                target_text = self.template_texts[next_event] if next_event < len(self.template_texts) else ""
            elif isinstance(next_event, torch.Tensor):
                target_emb = F.normalize(next_event, p=2, dim=-1)
                if target_emb.dim() == 1:
                    target_emb = target_emb.unsqueeze(0)
            else:
                raise TypeError(f"Unsupported type for next_event: {type(next_event)}")

            target_emb = target_emb.to(pred_emb.device)

            # Cosine similarity between expected embedding and observed embedding
            similarity = torch.sum(pred_emb * target_emb, dim=-1).item()
            distance = 1.0 - similarity

            # Rank check among candidate templates
            rank = -1
            in_topk = False
            topk_ids: List[str] = []

            if self.template_bank is not None and self.template_bank.size(0) > 0:
                all_candidates = self.template_bank
                all_ids = list(self.template_ids)

                if is_unseen:
                    all_candidates = torch.cat([all_candidates, target_emb], dim=0)
                    all_ids.append("UNSEEN_TARGET")

                sims = torch.matmul(pred_emb, all_candidates.T).squeeze(0)
                sorted_indices = torch.argsort(sims, descending=True).tolist()

                target_pos = (
                    all_ids.index("UNSEEN_TARGET")
                    if is_unseen
                    else (
                        next_event
                        if isinstance(next_event, int)
                        else (self.template_ids.index(next_event) if next_event in self.template_ids else -1)
                    )
                )

                if target_pos != -1:
                    rank = sorted_indices.index(target_pos) + 1
                    in_topk = rank <= top_k

                k_eff = min(top_k, len(all_ids))
                topk_ids = [all_ids[idx] for idx in sorted_indices[:k_eff]]

            # Decision logic: either similarity >= threshold OR in Top-K candidate set
            is_anomaly = (similarity < threshold) and (not in_topk)

            return {
                "is_anomaly": is_anomaly,
                "similarity": float(similarity),
                "semantic_distance": float(distance),
                "rank": rank,
                "in_topk": in_topk,
                "topk_ids": topk_ids,
                "is_unseen_oov": is_unseen,
                "target_text": target_text,
            }

    def score_sequence(
        self,
        sequence: Union[Sequence[str], torch.Tensor],
        window_size: int = 10,
        threshold: float = 0.5,
        top_k: int = 9,
    ) -> Dict[str, Any]:
        """
        Evaluate an entire log sequence using a rolling context window.

        Parameters
        ----------
        sequence : Sequence[str] or torch.Tensor
            Chronological sequence of log events.
        window_size : int, default=10
            Context window length for next-event prediction.
        threshold : float, default=0.5
            Anomaly detection cosine similarity threshold.
        top_k : int, default=9
            Top-K candidate acceptance window.

        Returns
        -------
        summary : Dict[str, Any]
            Anomaly summary containing anomaly count, ratio, and per-step evaluations.
        """
        total_len = len(sequence) if isinstance(sequence, (list, tuple)) else sequence.size(0)
        if total_len <= window_size:
            return {
                "is_anomaly": False,
                "anomaly_count": 0,
                "anomaly_ratio": 0.0,
                "evaluated_steps": 0,
                "total_steps": total_len,
                "step_details": [],
            }

        step_details: List[Dict[str, Any]] = []
        anomaly_count = 0

        for t in range(window_size, total_len):
            prefix = sequence[t - window_size : t]
            target = sequence[t]

            step_res = self.score_anomaly(
                prefix=prefix,
                next_event=target,
                threshold=threshold,
                top_k=top_k,
            )
            step_details.append(step_res)
            if step_res["is_anomaly"]:
                anomaly_count += 1

        evaluated_steps = len(step_details)
        ratio = anomaly_count / evaluated_steps if evaluated_steps > 0 else 0.0

        return {
            "is_anomaly": anomaly_count > 0,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": float(ratio),
            "evaluated_steps": evaluated_steps,
            "total_steps": total_len,
            "step_details": step_details,
        }

    def compute_loss(
        self,
        X: torch.Tensor,
        target: Union[torch.Tensor, int, Sequence[int]],
        loss_type: str = "hybrid",
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute multi-objective loss for semantic sequence prediction.

        Parameters
        ----------
        X : torch.Tensor
            Input sequence tensor.
        target : torch.Tensor or int or Sequence[int]
            Target next-event representation (embedding tensor or discrete class indices).
        loss_type : str, default="hybrid"
            Loss formulation:
            - 'cosine': 1.0 - cosine_similarity(pred_emb, target_emb)
            - 'mse': Mean Squared Error between embeddings
            - 'ce': Cross-Entropy over template bank logits
            - 'hybrid': alpha * cosine_loss + (1 - alpha) * cross_entropy_loss
        alpha : float, default=0.5
            Weight for cosine loss in hybrid mode.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss tensor.
        """
        if self.template_bank is not None and self.template_bank.size(0) > 0:
            log_probs, pred_emb = self.forward(X, return_embeddings=True)
        else:
            pred_emb = self.forward(X, return_embeddings=False)
            log_probs = None

        device = pred_emb.device

        # If target is discrete integer indices
        if isinstance(target, (int, list, tuple)) or (isinstance(target, torch.Tensor) and target.dtype in (torch.int32, torch.int64, torch.long)):
            if not isinstance(target, torch.Tensor):
                y_idx = torch.tensor([target] if isinstance(target, int) else list(target), dtype=torch.long, device=device)
            else:
                y_idx = target.to(device).long()

            if self.template_bank is None:
                raise ValueError("Discrete target indices provided but template bank is not set.")

            target_emb = self.template_bank[y_idx]
            ce_loss = F.nll_loss(log_probs, y_idx)
            cos_loss = torch.mean(1.0 - torch.sum(pred_emb * target_emb, dim=-1))

            if loss_type.lower() == "ce":
                return ce_loss
            elif loss_type.lower() == "cosine":
                return cos_loss
            elif loss_type.lower() == "mse":
                return F.mse_loss(pred_emb, target_emb)
            else:  # hybrid
                return alpha * cos_loss + (1.0 - alpha) * ce_loss

        # Continuous target embedding tensor
        target_emb = F.normalize(target.to(device), p=2, dim=-1)
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)

        cos_loss = torch.mean(1.0 - torch.sum(pred_emb * target_emb, dim=-1))
        if loss_type.lower() == "mse":
            return F.mse_loss(pred_emb, target_emb)
        return cos_loss

    def save(self, filepath: str) -> None:
        """Save model weights and template bank metadata to disk."""
        state = {
            "state_dict": self.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "tau": self.tau,
            "dropout": self.dropout_rate,
            "template_ids": self.template_ids,
            "template_texts": self.template_texts,
            "template_bank": self.template_bank,
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> SemanticDeepLog:
        """Load model checkpoint and template bank metadata from disk."""
        state = torch.load(filepath, map_location=device)
        model = cls(
            embedding_dim=state["embedding_dim"],
            hidden_size=state["hidden_size"],
            num_layers=state["num_layers"],
            rnn_type=state["rnn_type"],
            tau=state["tau"],
            dropout=state["dropout"],
        )
        bank = state.get("template_bank")
        if bank is not None:
            model.template_bank = bank.to(device) if device else bank
        model.load_state_dict(state["state_dict"], strict=False)
        model.template_ids = state.get("template_ids", [])
        model.template_texts = state.get("template_texts", [])
        if device is not None:
            model.to(device)
        return model


# ==============================================================================
# 4. TemplateSimilarityMatcher
# ==============================================================================


class TemplateSimilarityMatcher:
    """
    High-throughput Cosine Similarity Matcher and Zero-Shot OOV Template Resolver.

    Provides semantic indexing, nearest-neighbor matching, OOV thresholding,
    and semantic clustering for log templates and raw system log lines.
    """

    def __init__(
        self,
        encoder: Optional[LogSemanticEncoder] = None,
        templates: Optional[Dict[str, str]] = None,
        csv_path: Optional[str] = None,
    ):
        """
        Initialize the TemplateSimilarityMatcher.

        Parameters
        ----------
        encoder : LogSemanticEncoder, optional
            Neural semantic encoder instance. If None, instantiates a default encoder.
        templates : Dict[str, str], optional
            Dictionary mapping template ID to template text.
        csv_path : str, optional
            Path to templates CSV file. If neither templates nor csv_path is provided,
            attempts to load the canonical HDFS templates.
        """
        self.encoder = encoder or LogSemanticEncoder()
        self.template_ids: List[str] = []
        self.template_texts: List[str] = []
        self.template_embeddings: Optional[torch.Tensor] = None
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_labels: Optional[Dict[str, int]] = None

        if templates is not None:
            self.fit(templates)
        elif csv_path is not None:
            loaded = load_hdfs_templates(csv_path)
            self.fit(loaded)
        else:
            # Attempt default HDFS template loading
            try:
                loaded = load_hdfs_templates()
                self.fit(loaded)
            except FileNotFoundError:
                pass

    @classmethod
    def from_csv(
        cls,
        csv_path: Optional[str] = None,
        encoder: Optional[LogSemanticEncoder] = None,
    ) -> TemplateSimilarityMatcher:
        """Factory method constructing matcher directly from template CSV file."""
        return cls(encoder=encoder, csv_path=csv_path)

    def fit(self, templates: Dict[str, str]) -> TemplateSimilarityMatcher:
        """
        Index template collection and pre-compute continuous semantic vectors.

        Parameters
        ----------
        templates : Dict[str, str]
            Dictionary of {template_id: template_text}.
        """
        self.template_ids = list(templates.keys())
        self.template_texts = [templates[k] for k in self.template_ids]

        self.encoder.eval()
        with torch.no_grad():
            self.template_embeddings = self.encoder.encode(self.template_texts, normalize=True)

        return self

    def match(
        self,
        query: str,
        top_k: int = 1,
    ) -> List[Tuple[str, float, str]]:
        """
        Match a query string (raw log line or template) against known indexed templates.

        Parameters
        ----------
        query : str
            Query text string.
        top_k : int, default=1
            Number of top candidates to return.

        Returns
        -------
        matches : List[Tuple[template_id, cosine_similarity, template_text]]
        """
        batch_matches = self.match_batch([query], top_k=top_k)
        return batch_matches[0] if batch_matches else []

    def match_batch(
        self,
        queries: Sequence[str],
        top_k: int = 1,
    ) -> List[List[Tuple[str, float, str]]]:
        """
        Batch query matching leveraging vectorized matrix multiplication.

        Parameters
        ----------
        queries : Sequence[str]
            Sequence of log query strings.
        top_k : int, default=1
            Number of nearest templates to return per query.

        Returns
        -------
        results : List[List[Tuple[template_id, similarity, template_text]]]
        """
        if self.template_embeddings is None or len(self.template_ids) == 0:
            raise ValueError("No templates indexed in TemplateSimilarityMatcher. Call fit() first.")

        k_eff = min(top_k, len(self.template_ids))
        self.encoder.eval()

        with torch.no_grad():
            query_embs = self.encoder.encode(queries, normalize=True)
            similarities = torch.matmul(query_embs, self.template_embeddings.T)

        top_scores, top_indices = similarities.topk(k_eff, dim=-1)

        results: List[List[Tuple[str, float, str]]] = []
        for scores_row, indices_row in zip(top_scores.tolist(), top_indices.tolist()):
            row_matches: List[Tuple[str, float, str]] = []
            for score, idx in zip(scores_row, indices_row):
                row_matches.append((
                    self.template_ids[idx],
                    float(score),
                    self.template_texts[idx],
                ))
            results.append(row_matches)

        return results

    def resolve_template(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Determine whether a log message matches a known template or represents an Out-of-Vocabulary (OOV) template.

        Parameters
        ----------
        query : str
            Query raw log line or template string.
        threshold : float, default=0.75
            Similarity threshold. If top match similarity is below this threshold,
            the query is flagged as an unseen OOV template.

        Returns
        -------
        resolution : Dict[str, Any]
            Dictionary containing:
            - matched_id: Matched template ID if >= threshold, else None
            - similarity: Cosine similarity to the nearest known template
            - is_oov: Boolean flag (True if similarity < threshold)
            - nearest_id: Identifier of closest known template
            - nearest_template: Text of closest known template
        """
        matches = self.match(query, top_k=1)
        if not matches:
            return {
                "matched_id": None,
                "similarity": 0.0,
                "is_oov": True,
                "nearest_id": None,
                "nearest_template": None,
            }

        top_id, top_sim, top_text = matches[0]
        is_oov = top_sim < threshold

        return {
            "matched_id": top_id if not is_oov else None,
            "similarity": float(top_sim),
            "is_oov": is_oov,
            "nearest_id": top_id,
            "nearest_template": top_text,
        }

    def add_template(
        self,
        template_id: str,
        template_text: str,
    ) -> None:
        """
        Dynamically register a new template into the matcher index.

        Parameters
        ----------
        template_id : str
            Unique identifier for the new template.
        template_text : str
            Template text representation.
        """
        self.encoder.eval()
        with torch.no_grad():
            new_emb = self.encoder.encode(template_text, normalize=True)

        if self.template_embeddings is None:
            self.template_embeddings = new_emb
            self.template_ids = [template_id]
            self.template_texts = [template_text]
        else:
            self.template_embeddings = torch.cat([self.template_embeddings, new_emb], dim=0)
            self.template_ids.append(template_id)
            self.template_texts.append(template_text)

        # Invalidate cached clusters on dynamic template addition
        self.cluster_centroids = None
        self.cluster_labels = None

    def cluster_templates(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group known log templates into semantic clusters via spherical KMeans.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of semantic clusters to extract.
        random_state : int, default=42
            Random seed for cluster reproducibility.

        Returns
        -------
        clusters : Dict[int, List[Dict[str, Any]]]
            Mapping from cluster ID to member template dictionaries.
        """
        if self.template_embeddings is None or len(self.template_ids) == 0:
            raise ValueError("No templates indexed for clustering.")

        n_c = min(n_clusters, len(self.template_ids))
        embs_np = self.template_embeddings.cpu().numpy()

        kmeans = KMeans(n_clusters=n_c, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embs_np)

        # Compute L2-normalized cluster centroids for angular distance calculation
        raw_centroids = kmeans.cluster_centers_
        norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.cluster_centroids = raw_centroids / norms

        self.cluster_labels = {tid: int(lbl) for tid, lbl in zip(self.template_ids, labels)}

        clusters: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(n_c)}
        for tid, text, emb, lbl in zip(self.template_ids, self.template_texts, embs_np, labels):
            c_idx = int(lbl)
            centroid = self.cluster_centroids[c_idx]
            sim_to_center = float(np.dot(emb, centroid))
            clusters[c_idx].append({
                "template_id": tid,
                "template_text": text,
                "similarity_to_centroid": sim_to_center,
            })

        # Sort each cluster's elements by similarity to the cluster centroid
        for c in clusters:
            clusters[c].sort(key=lambda x: x["similarity_to_centroid"], reverse=True)

        return clusters

    def find_nearest_cluster(self, query: str) -> Tuple[int, float]:
        """
        Identify the closest semantic cluster for a query string.

        Parameters
        ----------
        query : str
            Query log string or template.

        Returns
        -------
        nearest_cluster_id : int
        similarity : float
        """
        if self.cluster_centroids is None:
            self.cluster_templates()

        self.encoder.eval()
        with torch.no_grad():
            q_emb = self.encoder.encode(query, normalize=True).cpu().numpy().squeeze(0)

        sims = np.dot(self.cluster_centroids, q_emb)
        best_c = int(np.argmax(sims))
        return best_c, float(sims[best_c])

    def pairwise_similarity_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Compute the pairwise cosine similarity matrix between all indexed templates.

        Returns
        -------
        template_ids : List[str]
            Ordered list of template identifiers.
        similarity_matrix : np.ndarray of shape (num_templates, num_templates)
        """
        if self.template_embeddings is None:
            raise ValueError("No templates indexed.")

        with torch.no_grad():
            sim_matrix = torch.matmul(self.template_embeddings, self.template_embeddings.T)

        return list(self.template_ids), sim_matrix.cpu().numpy()

    def get_embedding(self, template_id: str) -> torch.Tensor:
        """Retrieve precomputed normalized embedding vector for a template ID."""
        if self.template_embeddings is None:
            raise ValueError("No templates indexed.")
        if template_id not in self.template_ids:
            raise KeyError(f"Template ID '{template_id}' not found in index.")
        idx = self.template_ids.index(template_id)
        return self.template_embeddings[idx]

    def sync_with_model(self, model: SemanticDeepLog) -> None:
        """Synchronize the matcher's indexed template codebook with a SemanticDeepLog model."""
        if self.template_embeddings is not None:
            model.set_template_bank(
                embeddings=self.template_embeddings,
                template_ids=self.template_ids,
                template_texts=self.template_texts,
            )
            model.encoder = self.encoder
