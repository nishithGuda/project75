import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Import custom modules
from models.ui_parser import UIFeatureExtractor
from models.llm_engine import ElementSelectorModel
from models.rl_module import CustomRLAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_banking_vocab():
    return {
        "account", "balance", "transfer", "transaction", "statement", "overview",
        "checking", "savings", "credit", "debit", "input", "submit", "select", "filter",
        "date", "amount", "view", "details", "send", "receive", "money", "dashboard",
        "payment", "from", "to", "back", "home", "export"
    }

def train_models(llm_data_path, rl_data_path, output_dir, epochs=100, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        with open(llm_data_path, 'r') as f:
            llm_data = json.load(f)

        with open(rl_data_path, 'r') as f:
            rl_data = json.load(f)

        logger.info(f"Loaded {len(llm_data)} LLM training examples")
        logger.info(f"Loaded {len(rl_data)} RL training examples")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return False

    # Initialize feature extractor
    feature_extractor = UIFeatureExtractor(max_vocab=1000, max_elements=10)
    banking_vocab = build_banking_vocab()
    feature_extractor.build_vocabulary(llm_data + [
        {'query': word, 'elements': [{'text': word, 'content_desc': word}]} for word in banking_vocab
    ])

    # Mode B: Process query-element pairs
    processed_samples = []
    for sample in llm_data:
        processed_samples.extend(feature_extractor.process_mode_b_samples(sample))

    query_features = np.array([s['query_features'] for s in processed_samples])
    element_features = np.array([s['element_feature'] for s in processed_samples])
    labels = np.array([s['label'] for s in processed_samples])

    # Split data
    X_query_train, X_query_test, X_elem_train, X_elem_test, y_train, y_test = train_test_split(
        query_features, element_features, labels, test_size=0.2, random_state=42
    )

    # Train element selector model (Mode B)
    query_input_size = X_query_train.shape[1]
    element_feature_size = X_elem_train.shape[1]

    logger.info("Initializing Element Selector model...")
    selector_model = ElementSelectorModel(
        query_input_size=query_input_size,
        element_feature_size=element_feature_size,
        max_elements=1,
        hidden_dim=64,
        learning_rate=0.01
    )

    logger.info("Training Element Selector model...")
    losses = []
    accuracies = []
    for epoch in range(epochs):
        indices = np.arange(len(X_query_train))
        np.random.shuffle(indices)
        epoch_loss = 0
        correct = 0

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_query = X_query_train[batch_idx]
            batch_elem = X_elem_train[batch_idx]
            batch_labels = y_train[batch_idx]

            preds = selector_model.forward(batch_query, batch_elem)
            loss = selector_model.backward(batch_query, batch_elem, preds, batch_labels)

            epoch_loss += loss * len(batch_idx)
            correct += np.sum((preds > 0.5).astype(int) == batch_labels)

        epoch_loss /= len(X_query_train)
        accuracy = correct / len(X_query_train)
        losses.append(epoch_loss)
        accuracies.append(accuracy)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {accuracy:.4f}")

    # Save the model
    model_path = os.path.join(output_dir, 'element_selector_model.json')
    selector_model.save_model(model_path)
    logger.info(f"Element Selector model saved to {model_path}")

    # Save feature extractor
    feature_path = os.path.join(output_dir, 'feature_extractor.json')
    feature_extractor.save(feature_path)
    logger.info(f"Feature extractor saved to {feature_path}")

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    metrics_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(metrics_path)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Evaluate on test set
    y_pred = (selector_model.forward(X_query_test, X_elem_test) > 0.5).astype(int)
    test_accuracy = np.mean(y_pred == y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Train RL agent
    rl_state_size = 10
    rl_action_size = 3

    logger.info("Initializing RL Agent...")
    rl_agent = CustomRLAgent(
        state_size=rl_state_size,
        action_size=rl_action_size,
        hidden_dim=32,
        learning_rate=0.01
    )

    logger.info("Processing RL training data...")
    for example in rl_data:
        state = example['state']
        action_str = example['action']
        reward = example['reward']
        next_state = example['next_state']
        action_map = {'click': 0, 'input': 1, 'select': 2}
        action = action_map.get(action_str, 0)
        rl_agent.remember(state, action, reward, next_state, next_state is None)

    logger.info("Training RL Agent...")
    for i in range(10):
        logger.info(f"RL training epoch {i+1}/10...")
        rl_agent.replay(batch_size=min(32, len(rl_agent.memory)))

    # Save RL agent
    rl_model_path = os.path.join(output_dir, 'rl_agent_model.json')
    rl_agent.save_model(rl_model_path)
    logger.info(f"RL Agent model saved to {rl_model_path}")

    # Save test metrics
    test_metrics = {
        'element_selector_test_accuracy': float(test_accuracy),
        'rl_agent_epsilon_final': float(rl_agent.epsilon),
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'llm_data_size': len(llm_data),
            'rl_data_size': len(rl_data)
        }
    }

    metrics_json_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_json_path}")
    logger.info("Training complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UI Navigation Assistant models")
    parser.add_argument("--llm-data", required=True, help="Path to LLM training data JSON file")
    parser.add_argument("--rl-data", required=True, help="Path to RL training data JSON file")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()

    train_models(
        llm_data_path=args.llm_data,
        rl_data_path=args.rl_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
