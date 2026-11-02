from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    """
    Additive self-attention pooling layer.

    Aggregates variable or fixed-length sequences of hidden states into a single
    context vector by learning attention weights over sequence positions.
    """

    def __init__(self, hidden_size: int, attn_dim: Optional[int] = None):
        super().__init__()
        dim = attn_dim or max(32, hidden_size // 2)
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_size, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor of shape (batch_size, seq_len, hidden_size)
            Hidden representations from sequence backbone.
        mask : torch.Tensor, optional
            Boolean or float mask for padded tokens.

        Returns
        -------
        pooled : torch.Tensor of shape (batch_size, hidden_size)
            Attention-weighted sequence representation.
        """
        # Compute unnormalized attention energy: (batch_size, seq_len, 1)
        energy = self.attn_net(h)

        if mask is not None:
            if mask.dtype == torch.bool:
                energy = energy.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            else:
                energy = energy + mask.unsqueeze(-1)

        weights = F.softmax(energy, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(weights * h, dim=1)  # (batch_size, hidden_size)
        return pooled


class DeepLogDualHead(nn.Module):
    """
    DeepLogDualHead: Multi-task dual-head architecture for log anomaly detection.

    Combines:
    1. Token Head: Predicts the next event log-probabilities given context (local anomaly detection).
    2. Sequence Head: Predicts the binary anomaly probability for the entire sequence/block
       (global pattern anomaly detection) via self-attention pooling.

    Backbone:
    - Stacked LSTM backbone (hidden_size=128, num_layers=2).
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: Optional[int] = 64,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        pooling: str = "attention",
        **kwargs,
    ):
        """
        Initialize DeepLogDualHead model.

        Parameters
        ----------
        vocab_size : int, optional
            Number of distinct log event IDs. Can also be passed as `input_size`.
        input_size : int, optional
            DeepLog API compatibility alias for `vocab_size`.
        hidden_size : int, default=128
            Dimension of LSTM hidden state.
        num_layers : int, default=2
            Number of stacked LSTM layers.
        embedding_dim : int, optional, default=64
            Dimension of dense token embedding. If None or 0, uses one-hot encoding.
        output_size : int, optional
            DeepLog API compatibility alias for vocabulary prediction dimension.
        dropout : float, default=0.1
            Dropout probability between LSTM layers and in sequence head MLP.
        pooling : str, default="attention"
            Sequence aggregation mechanism ("attention", "last", or "mean").
        """
        super().__init__()

        effective_vocab = vocab_size if vocab_size is not None else input_size
        if effective_vocab is None:
            raise ValueError("Either vocab_size or input_size must be specified.")

        self.vocab_size = effective_vocab
        self.input_size = effective_vocab
        self.output_size = output_size if output_size is not None else effective_vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.pooling_type = pooling.lower()

        # Input representation: learned dense embedding or one-hot
        if self.embedding_dim is not None and self.embedding_dim > 0:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            lstm_input_dim = self.embedding_dim
        else:
            self.embedding = None
            lstm_input_dim = self.vocab_size

        # Stacked LSTM backbone
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Token Head: Next-event log-probabilities (shape: batch_size, vocab_size)
        self.token_head = nn.Linear(hidden_size, self.output_size)
        self.token_softmax = nn.LogSoftmax(dim=-1)

        # Sequence Head: Attention pooling + MLP projection to binary anomaly probability
        if self.pooling_type == "attention":
            self.pooler = SelfAttentionPooling(hidden_size)
        else:
            self.pooler = None

        seq_mlp_dim = max(32, hidden_size // 2)
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden_size, seq_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_mlp_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing token next-event log-probabilities and sequence anomaly probability.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len)
            Integer event index sequence.

        Returns
        -------
        token_log_probs : torch.Tensor of shape (batch_size, vocab_size)
            Log probabilities over the event vocabulary for the next event.
        sequence_anomaly_prob : torch.Tensor of shape (batch_size, 1)
            Calibrated binary anomaly probability for the sequence.
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Prepare LSTM input
        if self.embedding is not None:
            X_emb = self.embedding(X.to(torch.int64))
        else:
            X_emb = F.one_hot(X.to(torch.int64), self.vocab_size).to(torch.float32)

        # Initial hidden states
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=X.device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=X.device)

        lstm_out, _ = self.lstm(X_emb, (h0, c0))  # (batch_size, seq_len, hidden_size)

        # 1. Token Head (next event given last time step)
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        token_logits = self.token_head(last_hidden)  # (batch_size, output_size)
        token_log_probs = self.token_softmax(token_logits)

        # 2. Sequence Head (sequence representation -> binary anomaly probability)
        if self.pooling_type == "attention" and self.pooler is not None:
            seq_repr = self.pooler(lstm_out)
        elif self.pooling_type == "mean":
            seq_repr = lstm_out.mean(dim=1)
        else:  # "last"
            seq_repr = last_hidden

        sequence_anomaly_prob = self.sequence_head(seq_repr)  # (batch_size, 1)

        return token_log_probs, sequence_anomaly_prob

    def predict(
        self,
        X: torch.Tensor,
        y=None,
        k: int = 1,
        batch_size: int = 64,
        variable: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict Top-K next events and sequence-level anomaly probabilities.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len)
            Input sequences.
        y : Ignored
            Compatibility placeholder.
        k : int, default=1
            Number of top prediction candidates to return.
        batch_size : int, default=64
            Batch size for evaluation.
        variable : bool, default=False
            Compatibility placeholder.
        verbose : bool, default=False
            Verbosity flag.

        Returns
        -------
        token_result : torch.Tensor of shape (n_samples, k)
            Top-K predicted event indices.
        token_confidence : torch.Tensor of shape (n_samples, k)
            Probabilities corresponding to top-k predictions.
        sequence_anomaly_prob : torch.Tensor of shape (n_samples, 1)
            Sequence-level anomaly probability.
        """
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long)

        all_token_log_probs = []
        all_seq_probs = []

        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch_X = X[i : i + batch_size]
                token_lp, seq_p = self.forward(batch_X)
                all_token_log_probs.append(token_lp)
                all_seq_probs.append(seq_p)

        all_token_log_probs = torch.cat(all_token_log_probs, dim=0)
        all_seq_probs = torch.cat(all_seq_probs, dim=0)

        token_probs = all_token_log_probs.exp()
        token_confidence, token_result = token_probs.topk(k, dim=-1)

        return token_result, token_confidence, all_seq_probs

    def predict_token(
        self,
        X: torch.Tensor,
        k: int = 1,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next events matching the DeepLog single-head predict API.

        Returns
        -------
        result : torch.Tensor of shape (n_samples, k)
            Top-K predicted event indices.
        confidence : torch.Tensor of shape (n_samples, k)
            Probabilities corresponding to top-k predictions.
        """
        token_result, token_confidence, _ = self.predict(X, k=k, batch_size=batch_size)
        return token_result, token_confidence

    def predict_anomaly(
        self,
        X: torch.Tensor,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Predict sequence-level anomaly probabilities.

        Returns
        -------
        sequence_anomaly_prob : torch.Tensor of shape (n_samples, 1)
        """
        _, _, seq_prob = self.predict(X, batch_size=batch_size)
        return seq_prob

    def save(self, outfile: str):
        """
        Save model state dictionary and architecture parameters to file.

        Parameters
        ----------
        outfile : str
            Path to output checkpoint file (.pt).
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "pooling": self.pooling_type,
        }
        torch.save(checkpoint, outfile)

    @classmethod
    def load(cls, infile: str, device=None, **kwargs) -> "DeepLogDualHead":
        """
        Load model checkpoint from file.

        Parameters
        ----------
        infile : str
            Path to model checkpoint (.pt).
        device : torch.device, optional
            Target device to map checkpoint tensors.
        **kwargs : optional
            Optional overrides for architecture hyperparameters.
        """
        checkpoint = torch.load(infile, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            params = {
                "vocab_size": checkpoint.get("vocab_size", 30),
                "hidden_size": checkpoint.get("hidden_size", 128),
                "num_layers": checkpoint.get("num_layers", 2),
                "embedding_dim": checkpoint.get("embedding_dim", 64),
                "output_size": checkpoint.get("output_size", None),
                "dropout": checkpoint.get("dropout", 0.1),
                "pooling": checkpoint.get("pooling", "attention"),
            }
        else:
            state_dict = checkpoint
            # Infer parameters from weights
            if "embedding.weight" in state_dict:
                vocab_size = state_dict["embedding.weight"].shape[0]
                embedding_dim = state_dict["embedding.weight"].shape[1]
            else:
                vocab_size = state_dict["lstm.weight_ih_l0"].shape[1]
                embedding_dim = None

            hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
            output_size = state_dict["token_head.weight"].shape[0]
            num_layers = len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])
            pooling = "attention" if any(k.startswith("pooler.") for k in state_dict) else "last"

            params = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "embedding_dim": embedding_dim,
                "output_size": output_size,
                "dropout": kwargs.get("dropout", 0.1),
                "pooling": pooling,
            }

        params.update(kwargs)
        model = cls(**params)
        model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)

        return model
