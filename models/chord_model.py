import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CHORD_CLASSES


class ChordCNNLSTM(nn.Module):
    """
    Architecture CNN-LSTM pour la reconnaissance d'accords.
    Inspirée de ChordMiniApp et des meilleures pratiques MIR.
    
    Input: Chroma CQT (12 x T frames)
    Output: Probabilités sur NUM_CHORD_CLASSES
    """

    def __init__(self, num_classes: int = NUM_CHORD_CLASSES, input_dim: int = 12):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)

        return x


class ChordConformer(nn.Module):
    """
    Architecture Conformer pour la reconnaissance d'accords large vocabulaire.
    Inspirée de ChordFormer (Akram et al., 2025).
    Combine CNN local + Self-Attention global.
    """

    def __init__(self, num_classes: int = NUM_CHORD_CLASSES, input_dim: int = 12):
        super().__init__()
        self.num_classes = num_classes

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        return x
