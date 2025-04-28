# backend2/train.py
import os
from core.embed_pipeline import embed_training_data
from core.train_rl import train_rl_model


def main():
    """Run the full training pipeline"""
    print("=== Starting Training Pipeline ===")

    # Step 1: Generate embeddings from training data
    print("\n== Generating Embeddings ==")
    embeddings = embed_training_data(
        data_path="processed_data/training/llm_training_data.json",
        out_path="processed_data/training/ui_embeddings.json"
    )

    # Step 2: Train RL model
    print("\n== Training RL Model ==")
    model = train_rl_model(
        data_path="processed_data/training/rl_training_data.json",
        save_path="model/rl_model.pt"
    )

    print("\n=== Training Complete ===")
    print(f"Generated {len(embeddings)} embeddings")
    print("Trained RL model saved to model/rl_model.pt")
    print("\nYou can now start the server with:")
    print("uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
