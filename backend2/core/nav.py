import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from embed_and_save import embed_query
from ll_connect import call_llama
from vecstore import VectorStore

# === Config ===
DATA_PATH = "processed_data/training/rl_training_data.json"
SAVE_PATH = "model/rl_model.pt"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

# === Dataset ===


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

# === Model ===


class SimpleRLModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# === Training ===


def train():
    dataset = RLTrainingDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleRLModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs = batch["confidence"]
            targets = batch["reward"]

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"RL model saved to {SAVE_PATH}")

# === Navigation with RL Integration ===


def build_prompt(query, retrieved_elements):
    prompt = f"You are an intelligent UI navigation agent. A user issued the command: '{query}'.\n"
    prompt += "You were shown the following UI elements:\n\n"

    for idx, elem in enumerate(retrieved_elements):
        prompt += f"{idx+1}. Element Label: {elem['label']}, Screen ID: {elem['screen_id']}\n"

    prompt += "\nBased on this information, which element should the user interact with? "
    prompt += "Return the element number (1 to {0}) and explain why.\n".format(
        len(retrieved_elements))

    return prompt


def apply_rl_adjustment(elements, rl_model):
    # Apply RL model on top of retrieved elements
    with torch.no_grad():
        for elem in elements:
            confidence = torch.tensor(
                [[elem.get("score", 0.5)]], dtype=torch.float)
            reward_pred = rl_model(confidence).item()
            elem["adjusted_score"] = reward_pred

    return sorted(elements, key=lambda x: x.get("adjusted_score", 0), reverse=True)


def navigate(query, top_k=5):
    # Step 1: Embed query
    query_vector = embed_query(query)

    # Step 2: Retrieve elements
    store = VectorStore()
    top_elements = store.search(query_vector, top_k=top_k)

    if not top_elements:
        return {"action": "fallback", "reason": "No elements found"}

    # Step 3: Load RL model and adjust rankings
    rl_model = SimpleRLModel()
    rl_model.load_state_dict(torch.load(SAVE_PATH))
    rl_model.eval()

    top_elements = apply_rl_adjustment(top_elements, rl_model)

    # Step 4: Build prompt
    prompt = build_prompt(query, top_elements)

    # Step 5: Query LLM
    response = call_llama(prompt)

    return {
        "llm_response": response,
        "retrieved_elements": top_elements
    }


if __name__ == "__main__":
    train()
    q = "Transfer money"
    result = navigate(q)

    print("\n--- LLM Response ---")
    print(result["llm_response"])

    print("\n--- Top Retrieved Elements ---")
    for e in result["retrieved_elements"]:
        print(
            f"{e['label']} (Screen {e['screen_id']}) — Adjusted Score: {e.get('adjusted_score', 0):.4f}")
