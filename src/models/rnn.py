import torch
from torch import nn

# input_size - How many features exist per day
# batch_size - How many samples are processed simultaneously
# hidden_size - How many memory neurons to remember patterns

class RNNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.name = "rnn"
        self.input_size = input_size
        

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # binary classification
        )

    def forward(self, x): # x.shape (batch_size, seq_len, input_size)

        _, h_n = self.rnn(x)

        # Use last layer hidden state
        if self.rnn.bidirectional:
            # forward + backward concatenation
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_last = torch.cat((h_forward, h_backward), dim=1)
        else:
            h_last = h_n[-1]

        logits = self.fc(h_last)
        return logits # (batch_size, 1)