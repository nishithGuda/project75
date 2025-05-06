import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# RL Model Definition with your 3-layer architecture


class SimpleRLModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Middle layer
            nn.ReLU(),         # Activation
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

# Function to make data JSON serializable


def make_json_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


def train_rl_model_extended(data_path="processed_data/training/rl_training_data.json",
                            save_path="model/rl_model.pt",
                            plot_path="rl_training_extended.png",
                            metrics_path="rl_training_extended_metrics.json",
                            epochs=50,  # Increased from 10 to 50
                            batch_size=64,
                            lr=1e-3,
                            patience=10,  # Early stopping patience
                            input_dim=1):
    """Train an RL model with extended training and visualization"""

    # Ensure model directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load dataset
    print(f"Loading data from {data_path}")
    dataset = RLTrainingDataset(data_path)
    print(f"Dataset size: {len(dataset)} samples")

    # Split dataset into train/validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(
        f"Training on {train_size} samples, validating on {val_size} samples")

    # Initialize model with specified input dimension
    model = SimpleRLModel(input_dim=input_dim)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Modified scheduler without verbose parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    current_lr = lr

    # Start time for tracking
    start_time = datetime.now()
    print(f"Starting training at {start_time}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = batch["confidence"].to(device)
            targets = batch["reward"].to(device)

            # Forward pass
            preds = model(inputs)
            loss = loss_fn(preds, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(float(avg_train_loss))  # Convert to Python float

        # Validation phase
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs = batch["confidence"].to(device)
                targets = batch["reward"].to(device)

                # Forward pass
                preds = model(inputs)
                loss = loss_fn(preds, targets)

                val_loss += loss.item() * inputs.size(0)

                # Store predictions and targets for metrics
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate validation metrics
        # Convert to Python float
        avg_val_loss = float(val_loss / len(val_loader.dataset))
        val_losses.append(avg_val_loss)

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)

        # Mean Absolute Error
        # Convert to Python float
        mae = float(np.mean(np.abs(all_targets - all_preds)))
        val_maes.append(mae)

        # R² score
        y_mean = np.mean(all_targets)
        ss_tot = np.sum((all_targets - y_mean) ** 2)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        r2 = float(1 - (ss_res / ss_tot) if ss_tot >
                   0 else 0)  # Convert to Python float
        val_r2s.append(r2)

        # Update learning rate scheduler
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Manually print learning rate changes
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr} to {current_lr}")

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"MAE: {mae:.6f}, "
              f"R²: {r2:.6f}, "
              f"LR: {current_lr}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path),
                                           f"rl_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'mae': mae,
                'r2': r2
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
            print(f"New best model with validation loss: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Calculate training time
    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), save_path)
    print(f"✅ Best model saved to {save_path}")

    # Create training visualization
    plt.figure(figsize=(12, 10))

    # Plot loss curves
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(train_losses) + 1),
             train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1),
             val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot MAE
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(val_maes) + 1), val_maes, 'g-', label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot R²
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(val_r2s) + 1), val_r2s, 'orange', label='R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training visualization saved to {plot_path}")

    # Sample confidence values to visualize model predictions
    test_inputs = torch.linspace(0, 1, 50).view(-1, 1).to(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_inputs).cpu().numpy()

    # Plot model predictions
    plt.figure(figsize=(10, 6))
    plt.plot(test_inputs.cpu().numpy(), test_inputs.cpu().numpy(),
             'k--', label='Original Confidence')
    plt.plot(test_inputs.cpu().numpy(),
             test_outputs, 'r-', label='RL Adjusted')

    # Highlight regions where confidence is boosted or reduced
    plt.fill_between(test_inputs.cpu().numpy().flatten(),
                     test_inputs.cpu().numpy().flatten(),
                     test_outputs.flatten(),
                     where=(test_outputs.flatten() >
                            test_inputs.cpu().numpy().flatten()),
                     color='green', alpha=0.3, label='Boosted Confidence')

    plt.fill_between(test_inputs.cpu().numpy().flatten(),
                     test_inputs.cpu().numpy().flatten(),
                     test_outputs.flatten(),
                     where=(test_outputs.flatten() <
                            test_inputs.cpu().numpy().flatten()),
                     color='red', alpha=0.3, label='Reduced Confidence')

    plt.xlabel('Original Confidence Score')
    plt.ylabel('Adjusted Score')
    plt.title('RL Model Confidence Adjustments')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    adj_plot_path = plot_path.replace('.png', '_adjustments.png')
    plt.savefig(adj_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Model adjustment visualization saved to {adj_plot_path}")

    # Save training metrics for future reference - with JSON serialization fix
    metrics = {
        'epochs_completed': len(train_losses),
        'best_val_loss': float(best_val_loss),  # Convert to Python float
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_mae': val_maes[-1],
        'final_r2': val_r2s[-1],
        'training_time': str(training_time),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_r2s': val_r2s,
        'early_stopped': epochs_without_improvement >= patience,
        'model_params': {
            'input_dim': input_dim,
            'lr': float(lr),  # Convert to Python float
            'batch_size': batch_size
        }
    }

    # Ensure all data is JSON serializable
    metrics = make_json_serializable(metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Training metrics saved to {metrics_path}")

    return model, metrics


if __name__ == "__main__":
    print("Starting extended RL model training...")

    # You can adjust these parameters for longer training
    train_rl_model_extended(
        epochs=50,               # Train for 50 epochs instead of 10
        batch_size=64,
        lr=1e-3,
        patience=15,             # Wait 15 epochs before early stopping
        input_dim=1,             # For single confidence score input
        save_path="model/rl_model_extended.pt",
        plot_path="rl_training_extended.png",
        metrics_path="rl_training_extended_metrics.json"
    )
