import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomNeuralLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        z = np.dot(x, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        elif self.activation == 'tanh':
            self.output = np.tanh(z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            self.output = z
        return self.output

    def backward(self, grad_output):
        if self.activation == 'relu':
            grad_output = grad_output * (self.output > 0)
        elif self.activation == 'sigmoid':
            grad_output = grad_output * self.output * (1 - self.output)
        elif self.activation == 'tanh':
            grad_output = grad_output * (1 - self.output ** 2)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input, grad_weights, grad_biases

    def update_params(self, grad_weights, grad_biases, learning_rate):
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

class ElementSelectorModel:
    def __init__(self, query_input_size, element_feature_size, max_elements=10, hidden_dim=64, learning_rate=0.01):
        self.max_elements = max_elements
        self.learning_rate = learning_rate
        self.query_input_size = query_input_size
        self.element_feature_size = element_feature_size
        self.hidden_dim = hidden_dim

        self.query_encoder = [
            CustomNeuralLayer(query_input_size, hidden_dim, activation='relu'),
            CustomNeuralLayer(hidden_dim, hidden_dim, activation='relu')
        ]
        self.element_encoder = [
            CustomNeuralLayer(element_feature_size, hidden_dim, activation='relu'),
            CustomNeuralLayer(hidden_dim, hidden_dim, activation='relu')
        ]
        self.combined_layer = CustomNeuralLayer(hidden_dim * 2, hidden_dim, activation='relu')
        self.output_layer = CustomNeuralLayer(hidden_dim, 1, activation='sigmoid')

    def forward(self, query_features, element_features):
        if element_features.ndim == 2:
            hidden_query = query_features
            for layer in self.query_encoder:
                hidden_query = layer.forward(hidden_query)
            hidden_element = element_features
            for layer in self.element_encoder:
                hidden_element = layer.forward(hidden_element)
            combined = np.concatenate([hidden_query, hidden_element], axis=1)
            hidden = self.combined_layer.forward(combined)
            probs = self.output_layer.forward(hidden)
            return probs.squeeze()

        batch_size = query_features.shape[0]
        query_encoded = query_features
        for layer in self.query_encoder:
            query_encoded = layer.forward(query_encoded)
        element_outputs = []
        for i in range(self.max_elements):
            element_features_i = element_features[:, i, :]
            element_encoded = element_features_i
            for layer in self.element_encoder:
                element_encoded = layer.forward(element_encoded)
            combined_features = np.concatenate([query_encoded, element_encoded], axis=1)
            combined_encoded = self.combined_layer.forward(combined_features)
            element_output = self.output_layer.forward(combined_encoded)
            element_outputs.append(element_output)
        return np.hstack(element_outputs)

    def backward(self, query_features, element_features, output, target):
        if element_features.ndim == 2:
            grad_logits = output.reshape(-1, 1) - target.reshape(-1, 1)
            loss = -np.mean(target * np.log(output + 1e-8) + (1 - target) * np.log(1 - output + 1e-8))
            grad_hidden, grad_weights, grad_biases = self.output_layer.backward(grad_logits)
            self.output_layer.update_params(grad_weights, grad_biases, self.learning_rate)
            grad_combined, grad_weights, grad_biases = self.combined_layer.backward(grad_hidden)
            self.combined_layer.update_params(grad_weights, grad_biases, self.learning_rate)
            hidden_dim = self.hidden_dim
            grad_query = grad_combined[:, :hidden_dim]
            grad_element = grad_combined[:, hidden_dim:]
            for j in reversed(range(len(self.element_encoder))):
                layer = self.element_encoder[j]
                grad_element, grad_weights, grad_biases = layer.backward(grad_element)
                layer.update_params(grad_weights, grad_biases, self.learning_rate)
            for j in reversed(range(len(self.query_encoder))):
                layer = self.query_encoder[j]
                grad_query, grad_weights, grad_biases = layer.backward(grad_query)
                layer.update_params(grad_weights, grad_biases, self.learning_rate)
            return loss

        batch_size = query_features.shape[0]
        grad_output = output - target
        query_grads = []
        for i in range(self.max_elements):
            element_features_i = element_features[:, i, :]
            grad_output_i = grad_output[:, i:i+1]
            grad_combined, grad_weights, grad_biases = self.output_layer.backward(grad_output_i)
            self.output_layer.update_params(grad_weights, grad_biases, self.learning_rate)
            grad_features, grad_weights, grad_biases = self.combined_layer.backward(grad_combined)
            self.combined_layer.update_params(grad_weights, grad_biases, self.learning_rate)
            hidden_dim = self.query_encoder[-1].weights.shape[1]
            grad_query = grad_features[:, :hidden_dim]
            grad_element = grad_features[:, hidden_dim:]
            for j in reversed(range(len(self.element_encoder))):
                layer = self.element_encoder[j]
                grad_element, grad_weights, grad_biases = layer.backward(grad_element)
                layer.update_params(grad_weights, grad_biases, self.learning_rate)
            query_grads.append(grad_query)
        avg_query_grad = np.mean(query_grads, axis=0)
        for j in reversed(range(len(self.query_encoder))):
            layer = self.query_encoder[j]
            avg_query_grad, grad_weights, grad_biases = layer.backward(avg_query_grad)
            layer.update_params(grad_weights, grad_biases, self.learning_rate)

    def train(self, query_features, element_features, targets, epochs=100, batch_size=32):
        num_samples = query_features.shape[0]
        
        training_loss = []
        training_accuracy = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            query_features_shuffled = query_features[indices]
            element_features_shuffled = element_features[indices]
            targets_shuffled = targets[indices]
            
            epoch_losses = []
            epoch_accuracies = []
            
            # Process in batches
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_query = query_features_shuffled[i:end]
                batch_elements = element_features_shuffled[i:end]
                batch_targets = targets_shuffled[i:end]
                
                # Forward pass
                batch_output = self.forward(batch_query, batch_elements)
                
                # Compute loss (binary cross-entropy)
                epsilon = 1e-15  # Small constant to avoid log(0)
                batch_loss = -np.mean(
                    batch_targets * np.log(batch_output + epsilon) + 
                    (1 - batch_targets) * np.log(1 - batch_output + epsilon)
                )
                
                # Compute accuracy (for monitoring)
                pred_indices = np.argmax(batch_output, axis=1)
                true_indices = np.argmax(batch_targets, axis=1)
                batch_accuracy = np.mean(pred_indices == true_indices)
                
                # Backward pass
                self.backward(batch_query, batch_elements, batch_output, batch_targets)
                
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_accuracy)
            
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
            training_loss.append(avg_loss)
            training_accuracy.append(avg_accuracy)
            
        return training_loss, training_accuracy
            
    def predict(self, query_features, element_features):
        """Make a prediction for which element to select"""
        output = self.forward(query_features, element_features)
        return output
    
    def save_model(self, filename):
        """Save model parameters to file"""
        model_params = {
            'model_config': {
                'query_input_size': self.query_input_size,
                'element_feature_size': self.element_feature_size,
                'max_elements': self.max_elements,
                'hidden_dim': self.hidden_dim
            },
            'query_encoder': [
                {'weights': layer.weights.tolist(), 'biases': layer.biases.tolist()}
                for layer in self.query_encoder
            ],
            'element_encoder': [
                {'weights': layer.weights.tolist(), 'biases': layer.biases.tolist()}
                for layer in self.element_encoder
            ],
            'combined_layer': {
                'weights': self.combined_layer.weights.tolist(),
                'biases': self.combined_layer.biases.tolist()
            },
            'output_layer': {
                'weights': self.output_layer.weights.tolist(),
                'biases': self.output_layer.biases.tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(model_params, f)
            
    def load_model(self, filename):
        """Load model parameters from file"""
        with open(filename, 'r') as f:
            model_params = json.load(f)
            
        # Load model configuration if available
        config = model_params.get('model_config', {})
        if config:
            self.query_input_size = config.get('query_input_size', self.query_input_size)
            self.element_feature_size = config.get('element_feature_size', self.element_feature_size)
            self.max_elements = config.get('max_elements', self.max_elements)
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            
        # Load query encoder parameters
        for i, params in enumerate(model_params['query_encoder']):
            self.query_encoder[i].weights = np.array(params['weights'])
            self.query_encoder[i].biases = np.array(params['biases'])
            
        # Load element encoder parameters
        for i, params in enumerate(model_params['element_encoder']):
            self.element_encoder[i].weights = np.array(params['weights'])
            self.element_encoder[i].biases = np.array(params['biases'])
            
        # Load combined layer parameters
        self.combined_layer.weights = np.array(model_params['combined_layer']['weights'])
        self.combined_layer.biases = np.array(model_params['combined_layer']['biases'])
        
        # Load output layer parameters
        self.output_layer.weights = np.array(model_params['output_layer']['weights'])
        self.output_layer.biases = np.array(model_params['output_layer']['biases'])

class LLMEngine:
    def __init__(self, model_path=None, feature_extractor=None):
        """
        Initialize the LLM Engine with a custom trained model
        
        Args:
            model_path (str): Path to the saved model file
            feature_extractor: Feature extractor instance
        """
        self.model = None
        self.feature_extractor = feature_extractor
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model from file"""
        try:
            # Extract model configuration from saved file
            with open(model_path, 'r') as f:
                model_params = json.load(f)
            
            config = model_params.get('model_config', {})
            query_input_size = config.get('query_input_size', 15)  # Default
            element_feature_size = config.get('element_feature_size', 1024)  # Default
            max_elements = config.get('max_elements', 10)  # Default
            hidden_dim = config.get('hidden_dim', 64)  # Default
            
            # Create model
            self.model = ElementSelectorModel(
                query_input_size=query_input_size,
                element_feature_size=element_feature_size,
                max_elements=max_elements,
                hidden_dim=hidden_dim
            )
            
            # Load weights
            self.model.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a default model
            self.model = None
    
    def process_query(self, query, ui_screen, feature_extractor=None):
        """
        Process a natural language query against the UI screen
        
        Args:
            query (str): User's natural language query
            ui_screen (dict): UI screen metadata
            feature_extractor: Optional feature extractor
            
        Returns:
            list: Ranked list of UI elements with confidence scores
        """
        # Use provided feature extractor or class instance
        extractor = feature_extractor or self.feature_extractor
        
        if not extractor:
            logger.error("No feature extractor available")
            return []
        
        # Check if model is available
        if not self.model:
            logger.error("Model not loaded")
            return []
        
        # Process UI screen to extract features
        ui_sample = {
            'query': query,
            'elements': ui_screen.get('elements', []),
            'screen_id': ui_screen.get('screen_id', '')
        }
        
        processed_data = extractor.process_sample(ui_sample)
        query_features = np.array([processed_data['query_features']])
        element_features = np.array([processed_data['element_features']])
        
        # Make predictions
        try:
            predictions = self.model.predict(query_features, element_features)[0]
            
            # Create response with element predictions
            response = []
            elements = ui_screen.get('elements', [])
            
            for i, confidence in enumerate(predictions):
                if i < len(elements):
                    element = elements[i]
                    
                    # Determine interaction type based on element type
                    element_type = element.get('type', '').lower()
                    if element_type == 'input':
                        interaction = 'input'
                    elif element_type == 'select' or element_type == 'dropdown':
                        interaction = 'select'
                    else:
                        interaction = 'click'
                    
                    response.append({
                        'id': element.get('id', f'element_{i}'),
                        'type': element.get('type', 'unknown'),
                        'text': element.get('text', ''),
                        'content_desc': element.get('content_desc', ''),
                        'interaction': interaction,
                        'confidence': float(confidence)
                    })
            
            # Sort by confidence
            response.sort(key=lambda x: x['confidence'], reverse=True)
            return response
            
        except Exception as e:
            logger.error(f"Error predicting elements: {e}")
            return []