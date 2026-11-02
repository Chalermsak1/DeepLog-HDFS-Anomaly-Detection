import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from Vaswani et al. (2017).
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding table.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embedding to token embeddings.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        return self.dropout(x)


class CausalTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block with causal multi-head self-attention,
    feed-forward network, layer normalization, and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with residual Pre-LN connections.
        """
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN Feed-Forward Network
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)

        if need_weights:
            return x, attn_weights
        return x


class LogTransformer(nn.Module):
    """
    LogTransformer: Causal Transformer Architecture for Log Event Anomaly Detection.

    Designed for modeling sequences of log event keys without recurrence bottlenecks.
    Employs token embeddings, positional encodings, and causal self-attention blocks
    to capture both short- and long-term event transitions.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pos_encoding: str = "sinusoidal",
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize LogTransformer.

        Parameters
        ----------
        vocab_size : int, optional
            Number of distinct log event IDs in vocabulary. Can also be passed as `input_size`.
        d_model : int, default=128
            Dimension of token embeddings and attention hidden states.
        nhead : int, default=4
            Number of attention heads in multi-head attention.
        num_layers : int, default=2
            Number of stacked causal transformer blocks.
        dim_feedforward : int, default=256
            Dimension of intermediate feed-forward network layer.
        dropout : float, default=0.1
            Dropout probability.
        max_seq_len : int, default=512
            Maximum sequence length supported by positional encodings.
        pos_encoding : str, default="sinusoidal"
            Type of positional encoding ("sinusoidal" or "learned").
        input_size : int, optional
            DeepLog API compatibility alias for `vocab_size`.
        output_size : int, optional
            DeepLog API compatibility alias for vocabulary size / prediction space.
        """
        super().__init__()

        # Resolve vocabulary and output size
        effective_vocab = vocab_size if vocab_size is not None else input_size
        if effective_vocab is None:
            raise ValueError("Either vocab_size or input_size must be specified.")

        self.vocab_size = effective_vocab
        self.input_size = effective_vocab
        self.output_size = output_size if output_size is not None else effective_vocab
        self.d_model = d_model
        self.hidden_size = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding.lower()

        # Token embedding layer: maps integer event index -> d_model vector
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)

        # Positional encoding
        if self.pos_encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout
            )
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout
            )

        # Causal transformer blocks
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection head to log-probabilities over vocabulary
        self.head = nn.Linear(d_model, self.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer."""
        return self.token_embedding

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        return_attention: bool = False,
        average_attn_weights: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through LogTransformer with causal masking.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len), optional
            Integer event index sequence.
        return_all_tokens : bool, default=False
            If True, return log probabilities for each time-step (batch_size, seq_len, output_size).
            If False, return log probabilities for the next event after the last step (batch_size, output_size).
        return_attention : bool, default=False
            If True, return a tuple of (log_probs, attentions) where attentions is a list
            of attention weight tensors from each layer.
        average_attn_weights : bool, default=True
            If True, average attention weights across heads. If False, keep head dimension.
        inputs_embeds : torch.Tensor, optional
            Continuous input embeddings of shape (batch_size, seq_len, d_model) or (seq_len, d_model).

        Returns
        -------
        out : torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]
            Log probabilities over the event vocabulary, and optionally attention matrices.
        """
        if X is None and inputs_embeds is None:
            raise ValueError("Either X or inputs_embeds must be specified.")

        if inputs_embeds is not None:
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
            h = inputs_embeds * math.sqrt(self.d_model)
            h = self.pos_encoder(h)
        else:
            if X.dim() == 1:
                X = X.unsqueeze(0)

            X = X.to(torch.int64)
            batch_size, seq_len = X.shape
            device = X.device

            # Embedding with sqrt(d_model) scaling
            h = self.token_embedding(X) * math.sqrt(self.d_model)
            h = self.pos_encoder(h)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Causal mask ensuring position t only attends to positions <= t
        # Float mask with 0.0 for attend and -inf for masked positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        # Pass through causal blocks
        all_attentions: List[torch.Tensor] = []
        for layer in self.layers:
            if return_attention:
                h, attn = layer(
                    h,
                    attn_mask=causal_mask,
                    need_weights=True,
                    average_attn_weights=average_attn_weights,
                )
                all_attentions.append(attn)
            else:
                h = layer(h, attn_mask=causal_mask)

        h = self.final_norm(h)

        if return_all_tokens:
            logits = self.head(h)  # (batch_size, seq_len, output_size)
            out = self.softmax(logits)
        else:
            last_token = h[:, -1, :]  # (batch_size, d_model)
            logits = self.head(last_token)  # (batch_size, output_size)
            out = self.softmax(logits)

        if return_attention:
            return out, all_attentions
        return out

    def predict(
        self,
        X: torch.Tensor,
        y=None,
        k: int = 1,
        batch_size: int = 64,
        variable: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the Top-K most likely next events.

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
            Placeholder for variable length sequence compatibility.
        verbose : bool, default=False
            Verbosity flag.

        Returns
        -------
        result : torch.Tensor of shape (n_samples, k)
            Top-K predicted event indices.
        confidence : torch.Tensor of shape (n_samples, k)
            Probabilities corresponding to top-k predictions.
        """
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long)

        all_log_probs = []
        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch_X = X[i : i + batch_size]
                log_probs = self.forward(batch_X, return_all_tokens=False)
                all_log_probs.append(log_probs)

        all_log_probs = torch.cat(all_log_probs, dim=0)
        probs = all_log_probs.exp()
        confidence, result = probs.topk(k, dim=-1)
        return result, confidence

    def save(self, outfile: str):
        """
        Save model state dictionary and architecture parameters to file.

        Parameters
        ----------
        outfile : str
            Path to output checkpoint file (.pt).
        """
        state = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "max_seq_len": self.max_seq_len,
            "pos_encoding": self.pos_encoding_type,
            "output_size": self.output_size,
        }
        torch.save(state, outfile)

    @classmethod
    def load(cls, infile: str, device=None, **kwargs) -> "LogTransformer":
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
                "d_model": checkpoint.get("d_model", 128),
                "nhead": checkpoint.get("nhead", 4),
                "num_layers": checkpoint.get("num_layers", 2),
                "dim_feedforward": checkpoint.get("dim_feedforward", 256),
                "dropout": checkpoint.get("dropout", 0.1),
                "max_seq_len": checkpoint.get("max_seq_len", 512),
                "pos_encoding": checkpoint.get("pos_encoding", "sinusoidal"),
                "output_size": checkpoint.get("output_size", None),
            }
        else:
            # Checkpoint is raw state_dict
            state_dict = checkpoint
            emb_weight = state_dict.get("token_embedding.weight")
            vocab_size = emb_weight.shape[0] if emb_weight is not None else 30
            d_model = emb_weight.shape[1] if emb_weight is not None else 128

            layer_indices = [
                int(k.split(".")[1])
                for k in state_dict.keys()
                if k.startswith("layers.") and k.endswith(".norm1.weight")
            ]
            num_layers = max(layer_indices) + 1 if layer_indices else 2

            ffn_weight = state_dict.get("layers.0.ffn.0.weight")
            dim_feedforward = ffn_weight.shape[0] if ffn_weight is not None else 256
            head_weight = state_dict.get("head.weight")
            output_size = head_weight.shape[0] if head_weight is not None else vocab_size
            pos_encoding = (
                "learned" if "pos_encoder.pos_embedding.weight" in state_dict else "sinusoidal"
            )

            params = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "nhead": kwargs.get("nhead", 4),
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": kwargs.get("dropout", 0.1),
                "max_seq_len": kwargs.get("max_seq_len", 512),
                "pos_encoding": pos_encoding,
                "output_size": output_size,
            }

        params.update(kwargs)
        model = cls(**params)
        model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)

        return model
