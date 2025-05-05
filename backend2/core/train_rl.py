# backend2/core/rl_trainer.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# RL Model Definition matching your existing saved model architecture


class SimpleRLModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Add this middle layer
            nn.ReLU(),         # Add activation
            nn.Linear(8, 1)    # Final layer
        )

    def forward(self, x):
        return self.net(x)

# Dataset for RL training


class RLTrainingDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        state = sample["state"]

        query = state["query"]
        confidence = state.get("confidence", 0.5)
        reward = sample.get("reward", 0.0)
        screen_id = state.get("screen_id", "unknown")

        return {
            "query": query,
            "screen_id": screen_id,
            "confidence": torch.tensor([confidence], dtype=torch.float),
            "reward": torch.tensor([reward], dtype=torch.float)
        }


def train_rl_model(data_path="processed_data/training/rl_training_data.json",
                   save_path="model/rl_model.pt",
                   epochs=10,
                   batch_size=64,
                   lr=1e-3):
    """Train an RL model to adjust confidence scores based on feedback"""
    # Ensure model directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create dataset and loader
    dataset = RLTrainingDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleRLModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch["confidence"]
            targets = batch["reward"]

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✅ RL model saved to {save_path}")

    return model


if __name__ == "__main__":
    print("Starting RL model training...")
    train_rl_model()
