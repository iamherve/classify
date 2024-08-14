import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(TextClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        dropped = self.dropout(hidden[-1])
        output = self.fc(dropped)
        return output
