# src/model_engine.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MaritimeSequenceDataset(Dataset):
    """
    Converts tabular kinematic features into sliding sequential windows
    suitable for Gated Recurrent Unit (GRU) ingestion.
    """

    def __init__(self, dataframe, feature_cols, sequence_length=5):
        self.sequence_length = sequence_length

        # Extract features and targets as numpy matrices
        X_raw = dataframe[feature_cols].values.astype(np.float32)
        y_raw = dataframe['is_spoofed'].values.astype(np.float32)

        self.X_sequences = []
        self.y_labels = []

        # Build sliding windows
        for i in range(len(dataframe) - sequence_length):
            self.X_sequences.append(X_raw[i: i + sequence_length])
            # Label the sequence as spoofed if the final trailing point is compromised
            self.y_labels.append(y_raw[i + sequence_length - 1])

        self.X_sequences = torch.tensor(np.array(self.X_sequences))
        self.y_labels = torch.tensor(np.array(self.y_labels)).unsqueeze(1)

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_labels[idx]


class GRUSpoofDetector(nn.Module):
    """
    Recurrent Neural Network layout designed to spot multi-criteria
    kinematic anomalies along a vessel's path.
    """

    def __init__(self, input_dim, hidden_dim=16, num_layers=1):
        super(GRUSpoofDetector, self).__init__()

        # GRU Core Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully Connected Output Layer mapping to a binary classification logit
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Out shape: [batch, sequence_length, hidden_dim]
        gru_out, _ = self.gru(x)

        # Extract the hidden state representation of the very last step in the window
        last_step_out = gru_out[:, -1, :]

        # Pass through classifier
        out = self.fc(last_step_out)
        return self.sigmoid(out)


def train_gru_model(df_features, feature_cols, epochs=20, sequence_length=5):
    """
    Initializes components, handles forward/backward passes, and optimizes weights.
    """
    print("\nModel Engine: Initializing deep learning pipeline...")

    # 1. Instantiate dataset and data loader
    dataset = MaritimeSequenceDataset(df_features, feature_cols, sequence_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Instantiate Model architecture
    input_dim = len(feature_cols)
    model = GRUSpoofDetector(input_dim=input_dim, hidden_dim=16)

    # 3. Setup Binary Cross Entropy Loss and Adam Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. Training Loop execution
    model.train()
    print(f"Beginning optimization execution across {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            # Zero out gradients from past batch iteration
            optimizer.zero_grad()

            # Forward computation pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backpropagation optimization pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f" -> Epoch [{epoch + 1}/{epochs}] complete. Average Mini-batch Loss: {epoch_loss / len(dataloader):.5f}")

    print("Model optimization complete!")
    return model