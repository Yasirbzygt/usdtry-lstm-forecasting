import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # 2 katmanlı LSTM, daha yüksek temsil gücü
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Fully connected katman
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # LSTM çıktısı
        out, _ = self.lstm(x)

        # Son zaman adımı
        out = out[:, -1, :]

        # Regresyon çıktısı
        out = self.fc(out)

        return out
