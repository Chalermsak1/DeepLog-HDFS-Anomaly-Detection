import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtrain import Module


class DeepLog(Module):
    """
    DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning.
    Stacked LSTM architecture for modeling log sequence event patterns.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        """
        Initialize DeepLog model.

        Parameters
        ----------
        input_size : int
            Dimension of input vocabulary layer (number of distinct event IDs).
        hidden_size : int
            Dimension of LSTM hidden state.
        output_size : int
            Dimension of output layer (prediction candidates).
        num_layers : int, default=2
            Number of stacked LSTM layers.
        """
        super(DeepLog, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Stacked LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Linear projection to output classes
        self.out = nn.Linear(hidden_size, output_size)
        # LogSoftmax for probability distribution
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepLog.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len)
            Integer event index sequence.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, output_size)
            Log probabilities over the event vocabulary for the next event.
        """
        # One-hot encode integer event indices
        X = F.one_hot(X.to(torch.int64), self.input_size).to(torch.float)

        # Initial hidden and cell states
        hidden = self._get_initial_state(X)
        state = self._get_initial_state(X)

        # LSTM forward
        out, _ = self.lstm(X, (hidden, state))
        # Select last time-step hidden state
        out = self.out(out[:, -1, :])
        # Compute log probabilities
        out = self.softmax(out)

        return out

    def predict(self, X: torch.Tensor, y=None, k: int = 1, variable: bool = False, verbose: bool = True):
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
        variable : bool, default=False
            If True, handle variable sequence lengths.
        verbose : bool, default=True
            Verbosity flag.

        Returns
        -------
        result : torch.Tensor of shape (n_samples, k)
            Top-K predicted event indices.
        confidence : torch.Tensor of shape (n_samples, k)
            Probabilities corresponding to top-k predictions.
        """
        result = super().predict(X, variable=variable, verbose=verbose)
        result = result.exp()
        confidence, result = result.topk(k)
        return result, confidence

    def save(self, outfile: str):
        """Save model state dictionary to file."""
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile: str, device=None):
        """
        Load model checkpoint from file.

        Parameters
        ----------
        infile : str
            Path to model checkpoint (.pt).
        device : torch.device, optional
            Target device to map checkpoint tensors.
        """
        state_dict = torch.load(infile, map_location=device)

        input_size = state_dict.get('lstm.weight_ih_l0').shape[1]
        hidden_size = state_dict.get('lstm.weight_hh_l0').shape[1]
        output_size = input_size
        num_layers = (len(state_dict) - 2) // 4

        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
        )

        if device is not None:
            model = model.to(device)

        model.load_state_dict(state_dict)
        return model

    def _get_initial_state(self, X: torch.Tensor) -> torch.Tensor:
        """Return zero-initialized hidden/cell state tensor."""
        return torch.zeros(
            self.num_layers,
            X.size(0),
            self.hidden_size,
            device=X.device,
        )
