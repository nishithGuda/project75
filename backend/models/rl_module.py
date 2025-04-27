import numpy as np
import json
import logging
import os
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomNeuralLayer:
    def __init__(self, input_size, output_size, activation=None):
        # Initialize weights with a small random values
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # For backpropagation
        self.input = None
        self.output = None
        
    def forward(self, x):
        self.input = x
        z = np.dot(x, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # Clip to avoid overflow
        elif self.activation == 'tanh':
            self.output = np.tanh(z)
        else:
            self.output = z
            
        return self.output
    def backward(self, grad_output):
        # Clip the incoming gradients to prevent explosion
        grad_output = np.clip(grad_output, -1.0, 1.0)
        
        if self.activation == 'relu':
            grad_output = grad_output * (self.output > 0)
        elif self.activation == 'sigmoid':
            grad_output = grad_output * self.output * (1 - self.output)
        elif self.activation == 'tanh':
            grad_output = grad_output * (1 - self.output ** 2)
        
        # Calculate gradients
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Also clip gradients for weights and biases
        grad_weights = np.clip(grad_weights, -1.0, 1.0)
        grad_biases = np.clip(grad_biases, -1.0, 1.0)
        
        return grad_input, grad_weights, grad_biases
    def update_params(self, grad_weights, grad_biases, learning_rate):
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

class CustomRLAgent:
    """Reinforcement Learning agent for UI navigation"""
    
    def __init__(self, state_size=10, action_size=3, hidden_dim=64, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # RL parameters
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Memory for experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Initialize Q-network
        self.model = [
            CustomNeuralLayer(state_size, hidden_dim, activation='relu'),
            CustomNeuralLayer(hidden_dim, hidden_dim, activation='relu'),
            CustomNeuralLayer(hidden_dim, action_size, activation=None)
        ]
        
        # Tracking metrics
        self.loss_history = []
        self.reward_history = []
        
    def _process_state(self, state):
        """Process a state dict into a feature vector"""
        features = []
        
        # Extract query confidence
        confidence = state.get('confidence', 0.5)
        features.append(confidence)
        
        # Extract screen_id if it's a number
        screen_id = state.get('screen_id', '0')
        try:
            screen_id_num = float(screen_id)
            features.append(screen_id_num / 1000)  # Normalize
        except ValueError:
            features.append(0)  # Default value
            
        # Check if we extracted query tokens or features
        # For this simple version, we'll add placeholder features to reach state_size
        features.extend([0] * (self.state_size - len(features)))
        
        return np.array([features])
        
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose an action based on state"""
        processed_state = self._process_state(state)
        
        # Exploration: choose random action
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation: choose best action according to Q-values
        q_values = self._forward(processed_state)
        return np.argmax(q_values[0])
    
    def _forward(self, state):
        """Forward pass through Q-network"""
        x = state
        for layer in self.model:
            x = layer.forward(x)
        return x
    
    def _backward(self, state, target):
        """Backward pass through Q-network"""
        # Forward pass to store activations
        output = self._forward(state)
        
        # Compute error
        error = target - output
        
        # Backward pass through layers
        grad = error
        for layer in reversed(self.model):
            grad, grad_weights, grad_biases = layer.backward(grad)
            layer.update_params(grad_weights, grad_biases, self.learning_rate)
        
        # Return loss (mean squared error)
        return np.mean(error ** 2)

    def replay(self, batch_size=None):
        """Train model on experiences"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
            
        # Sample batch from memory
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        total_loss = 0
        
        for i in batch_indices:
            state, action, reward, next_state, done = self.memory[i]
            processed_state = self._process_state(state)
            
            target = self._forward(processed_state).copy()
            
            if done or next_state is None:
                target[0][action] = reward
            else:
                processed_next_state = self._process_state(next_state)
                future_reward = np.max(self._forward(processed_next_state)[0])
                target[0][action] = reward + self.gamma * future_reward
                
            # Train network
            loss = self._backward(processed_state, target)
            total_loss += loss
            
        # Reduce epsilon over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Track loss
        avg_loss = total_loss / batch_size
        self.loss_history.append(avg_loss)
        
        return avg_loss
            
    def save_model(self, filename):
        """Save model parameters to file"""
        model_params = {
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            },
            'layers': [
                {'weights': layer.weights.tolist(), 'biases': layer.biases.tolist()}
                for layer in self.model
            ],
            'metrics': {
                'loss_history': self.loss_history,
                'reward_history': self.reward_history,
                'timestamp': time.time()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(model_params, f)
            
    def load_model(self, filename):
        """Load model parameters from file"""
        if not os.path.exists(filename):
            logger.error(f"Model file not found: {filename}")
            return False
            
        try:
            with open(filename, 'r') as f:
                model_params = json.load(f)
                
            # Load configuration
            config = model_params.get('config', {})
            self.state_size = config.get('state_size', self.state_size)
            self.action_size = config.get('action_size', self.action_size)
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            self.learning_rate = config.get('learning_rate', self.learning_rate)
            self.gamma = config.get('gamma', self.gamma)
            self.epsilon = config.get('epsilon', self.epsilon)
            self.epsilon_min = config.get('epsilon_min', self.epsilon_min)
            self.epsilon_decay = config.get('epsilon_decay', self.epsilon_decay)
            
            # Load layer parameters
            if len(model_params.get('layers', [])) == len(self.model):
                for i, params in enumerate(model_params['layers']):
                    self.model[i].weights = np.array(params['weights'])
                    self.model[i].biases = np.array(params['biases'])
                    
                # Load metrics
                metrics = model_params.get('metrics', {})
                self.loss_history = metrics.get('loss_history', [])
                self.reward_history = metrics.get('reward_history', [])
                
                logger.info(f"Model loaded successfully from {filename}")
                return True
            else:
                logger.error("Model architecture mismatch")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def get_action_probs(self, state):
        """Get probabilities for each action based on state"""
        processed_state = self._process_state(state)
        q_values = self._forward(processed_state)[0]
        
        # Convert to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probs = exp_q / np.sum(exp_q)
        
        return probs

class RLModule:
    """Main interface for the Reinforcement Learning module"""
    
    def __init__(self, model_path=None):
        """Initialize the RL module"""
        # Action mapping (index to action type)
        self.action_map = {
            0: 'click',
            1: 'input',
            2: 'select'
        }
        
        # Inverse action mapping (action type to index)
        self.action_index = {
            'click': 0,
            'input': 1,
            'select': 2
        }
        
        # Initialize RL agent
        self.agent = CustomRLAgent(
            state_size=10,  # Size of state representation
            action_size=len(self.action_map),  # Number of possible actions
            hidden_dim=32,
            learning_rate=0.001
        )
        
        # Load model if specified
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
        
    def determine_action_type(self, query, confidence, screen_id):
        """
        Determine the type of action to take based on query and confidence
        
        Args:
            query (str): User's natural language query
            confidence (float): Confidence score from element selection
            screen_id (str): ID of the current screen
            
        Returns:
            str: Action type (click, input, select)
        """
        # Create state representation
        state = {
            'query': query,
            'confidence': confidence,
            'screen_id': screen_id
        }
        
        # Get action index from RL agent
        action_idx = self.agent.act(state)
        
        # Map to action type
        return self.action_map.get(action_idx, 'click')  # Default to click
    
    def record_feedback(self, query, confidence, screen_id, action_type, reward):
        """
        Record feedback for a performed action
        
        Args:
            query (str): User's natural language query
            confidence (float): Confidence score from element selection
            screen_id (str): ID of the current screen
            action_type (str): Type of action performed
            reward (float): Reward value (positive for success, negative for failure)
            
        Returns:
            None
        """
        # Create state representation
        state = {
            'query': query,
            'confidence': confidence,
            'screen_id': screen_id
        }
        
        # Convert action type to index
        action_idx = self.action_index.get(action_type, 0)
        
        # Add to agent memory
        self.agent.remember(state, action_idx, reward, None, True)
        
        # Train agent if we have enough experiences
        if len(self.agent.memory) >= 5:
            self.agent.replay(min(32, len(self.agent.memory)))
    
    def save_model(self, filepath):
        """Save RL model to file"""
        return self.agent.save_model(filepath)
    
    def get_action_confidence(self, query, confidence, screen_id):
        """
        Get confidence scores for different action types
        
        Args:
            query (str): User's natural language query
            confidence (float): Confidence score from element selection
            screen_id (str): ID of the current screen
            
        Returns:
            dict: Action types with confidence scores
        """
        # Create state representation
        state = {
            'query': query,
            'confidence': confidence,
            'screen_id': screen_id
        }
        
        # Get action probabilities
        probs = self.agent.get_action_probs(state)
        
        # Map to action types
        return {
            action_type: float(probs[idx])
            for idx, action_type in self.action_map.items()
        }