import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import AdamW
from models.llm_model_bert import LLMQueryElementClassifier
from models.llm_dataset import LLMQueryElementDataset
import os
import time
import sys

<<<<<<< HEAD
json_path = "processed_data/training/llm_training_data.json"
epochs = 20
batch_size = 64
lr = 0.001
=======
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

json_path = "processed_data/training/llm_training_data.json"
epochs = 20
batch_size = 32
lr = 5e-5
>>>>>>> 6464533d2d85f9f254190f2eb0ef40e2f639f40d
train_split = 0.85

# Print dataset info for debugging
dataset = LLMQueryElementDataset(json_path)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Element feature shape: {sample['element_features'].shape}")

train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Initialize model with correct element_feature_dim
element_dim = sample['element_features'].shape[0]
print(f"Using element feature dimension: {element_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LLMQueryElementClassifier(element_feature_dim=element_dim)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=lr)

print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    batch_count = 0
<<<<<<< HEAD
    
=======

>>>>>>> 6464533d2d85f9f254190f2eb0ef40e2f639f40d
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        element_features = batch["element_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, element_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
<<<<<<< HEAD
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s", flush=True)
            sys.stdout.flush()
    
    avg_training_loss = train_loss / batch_count
    
=======

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s", flush=True)
            sys.stdout.flush()

    avg_training_loss = train_loss / batch_count

>>>>>>> 6464533d2d85f9f254190f2eb0ef40e2f639f40d
    # Validation phase
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            element_features = batch["element_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, element_features)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
<<<<<<< HEAD
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_training_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.4f} | Time: {epoch_time:.2f}s", flush=True)
    
=======

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_training_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.4f} | Time: {epoch_time:.2f}s", flush=True)

>>>>>>> 6464533d2d85f9f254190f2eb0ef40e2f639f40d
    # Save a checkpoint after each epoch
    checkpoint_path = os.path.join("model", f"checkpoint_epoch_{epoch+1}.pt")
    os.makedirs("model", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_training_loss,
        'val_loss': avg_val_loss,
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}", flush=True)

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/llm_bert_model.pt")
<<<<<<< HEAD
print("Model Saved to model/llm_bert_model.pt", flush=True)
=======
print("Model Saved to model/llm_bert_model.pt", flush=True)
>>>>>>> 6464533d2d85f9f254190f2eb0ef40e2f639f40d
