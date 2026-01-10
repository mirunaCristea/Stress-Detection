# modele/cnn_model.py
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # -> (B,128,1)
            nn.Flatten(),             # -> (B,128)
            nn.Dropout(dropout),
            nn.Linear(128, 1)         # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)
