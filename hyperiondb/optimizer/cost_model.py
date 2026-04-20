"""
Neural Cost Model for HyperionDB
================================

Deep learning-based cost model for estimating query execution cost.
Uses neural networks to learn accurate cost predictions from data.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import logging

from hyperiondb.core.query import QueryPlan, PlanNode

logger = logging.getLogger(__name__)


class NeuralCostModel:
    """
    Neural Network-based Cost Model.
    
    Estimates the cost of query execution plans using a neural network
    trained on historical query performance data.
    
    Architecture:
    - Input: Plan features (rows, predicates, join types, etc.)
    - Hidden layers: Fully connected with ReLU activation
    - Output: Estimated cost (CPU time + I/O time)
    
    Features:
    - Online learning from query feedback
    - Feature extraction from query plans
    - Uncertainty estimation
    - Cost component decomposition
    """
    
    def __init__(self, hidden_dims: List[int] = [128, 64, 32]):
        """
        Initialize the neural cost model.
        
        Args:
            hidden_dims: Hidden layer dimensions
        """
        self.hidden_dims = hidden_dims
        self.input_dim = 50  # Feature vector size
        self.output_dim = 5  # Cost components
        
        # Initialize network weights (simplified implementation)
        self._init_weights()
        
        # Training history
        self.training_data: deque = deque(maxlen=10000)
        self.weights_history: List[Dict] = []
        
        # Statistics
        self.prediction_count = 0
        self.total_error = 0.0
        self.max_error = 0.0
        
        # Learned parameters (simulated neural network)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        # Model parameters
        self.learning_rate = 0.001
        self.epoch_count = 0
        
        logger.info("Neural Cost Model initialized")
    
    def _init_weights(self):
        """Initialize neural network weights."""
        np.random.seed(42)
        
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros((1, dims[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def extract_features(self, plan: QueryPlan) -> np.ndarray:
        """
        Extract features from a query plan.
        
        Features include:
        - Estimated number of rows at each node
        - Predicate selectivities
        - Join cardinalities
        - Index availability
        - Table sizes
        
        Args:
            plan: Query plan to extract features from
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.input_dim)
        
        # Traverse plan tree
        nodes = [plan.root] if plan.root else []
        idx = 0
        
        while nodes and idx < self.input_dim:
            node = nodes.pop(0)
            if not node:
                continue
            
            # Node type features (one-hot encoding)
            if idx < 10:
                node_type_offset = hash(node.node_type.value) % 10
                features[idx + node_type_offset] = 1.0
                idx += 10
            
            # Row count features
            if idx < 20:
                features[idx] = np.log1p(node.estimated_rows)
                idx += 1
            
            # Cost features
            if idx < 25:
                features[idx] = np.log1p(node.estimated_cost)
                idx += 1
            
            # Add children
            nodes.extend(node.children)
        
        # Plan-level features
        if idx < 30:
            features[idx] = len(plan.tables)
            idx += 1
        if idx < 35:
            features[idx] = len(plan.join_order)
            idx += 1
        
        # Table statistics (use plan info)
        if idx < 40:
            features[idx] = sum(n.estimated_rows for n in self._iter_nodes(plan.root)) / max(1, len(list(self._iter_nodes(plan.root))))
            idx += 1
        
        # Complexity features
        if idx < 45:
            features[idx] = plan.optimization_time
            idx += 1
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features.astype(np.float32)
    
    def _iter_nodes(self, node: PlanNode):
        """Iterate over all nodes in a plan tree."""
        if node is None:
            return
        yield node
        for child in node.children:
            yield from self._iter_nodes(child)
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        Args:
            features: Input feature vector
            
        Returns:
            Predicted cost components
        """
        x = features.reshape(1, -1)
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU activation
        
        # Output layer (linear)
        output = x @ self.weights[-1] + self.biases[-1]
        
        # Cost components: [cpu_cost, io_cost, memory_cost, network_cost, total]
        return output[0]
    
    def predict(self, plan: QueryPlan) -> Tuple[float, Dict[str, float]]:
        """
        Predict the cost of a query plan.
        
        Args:
            plan: Query plan to predict cost for
            
        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        self.prediction_count += 1
        
        features = self.extract_features(plan)
        cost_components = self.forward(features)
        
        # Extract individual costs
        cpu_cost = float(cost_components[0])
        io_cost = float(cost_components[1])
        memory_cost = float(cost_components[2])
        network_cost = float(cost_components[3])
        
        # Total cost (weighted sum)
        total_cost = cpu_cost + io_cost * 2.0 + memory_cost * 1.5 + network_cost * 10.0
        
        breakdown = {
            "cpu_cost": cpu_cost,
            "io_cost": io_cost,
            "memory_cost": memory_cost,
            "network_cost": network_cost,
            "total_cost": total_cost
        }
        
        return total_cost, breakdown
    
    def estimate(self, plan: QueryPlan) -> float:
        """
        Estimate query plan cost (simplified interface).
        
        Args:
            plan: Query plan
            
        Returns:
            Estimated cost
        """
        cost, _ = self.predict(plan)
        return cost
    
    def train(self, query_history: List[Dict], epochs: int = 10):
        """
        Train the cost model on historical query data.
        
        Args:
            query_history: List of historical query records
            epochs: Number of training epochs
        """
        if not query_history:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for record in query_history:
            if "features" in record and "actual_cost" in record:
                X.append(record["features"])
                y.append([
                    record["actual_cost"] * 0.4,  # cpu
                    record["actual_cost"] * 0.3,  # io
                    record["actual_cost"] * 0.2,  # memory
                    record["actual_cost"] * 0.1,  # network
                    record["actual_cost"]         # total
                ])
        
        if not X:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Training loop (simplified gradient descent)
        for epoch in range(epochs):
            # Forward pass
            predictions = self._predict_batch(X)
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            
            # Simplified backpropagation (update last layer)
            error = 2 * (predictions - y) / len(y)
            
            # Gradient descent update
            grad_w = X.T @ error / len(y)
            grad_b = np.mean(error, axis=0)
            
            self.weights[-1] -= self.learning_rate * grad_w.T
            self.biases[-1] -= self.learning_rate * grad_b
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.epoch_count += epochs
    
    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction."""
        predictions = []
        
        for features in X:
            pred = self.forward(features)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def update(self, plan: QueryPlan, actual_latency: float):
        """
        Update the model based on actual query execution.
        
        Uses online learning to refine predictions.
        
        Args:
            plan: Executed query plan
            actual_latency: Actual execution time
        """
        # Extract features
        features = self.extract_features(plan)
        
        # Add to training data
        self.training_data.append({
            "features": features,
            "actual_cost": actual_latency,
            "plan": plan
        })
        
        # Online update (simplified)
        predicted_cost, _ = self.predict(plan)
        error = actual_latency - predicted_cost
        
        self.total_error += abs(error)
        self.max_error = max(self.max_error, abs(error))
        
        # Update statistics
        if self.prediction_count > 0:
            avg_error = self.total_error / self.prediction_count
            
            # Adjust learning rate based on error
            if abs(error) > avg_error * 2:
                self.learning_rate = min(0.01, self.learning_rate * 1.1)
            else:
                self.learning_rate = max(0.0001, self.learning_rate * 0.99)
        
        # Periodic batch training
        if len(self.training_data) >= 100:
            self._online_training()
    
    def _online_training(self):
        """Perform online training on accumulated data."""
        if len(self.training_data) < 100:
            return
        
        # Sample recent data
        samples = list(self.training_data)[-100:]
        
        X = np.array([s["features"] for s in samples])
        y = np.array([[s["actual_cost"]] * self.output_dim for s in samples])
        
        # Normalize targets
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0) + 1e-8
        y_normalized = (y - y_mean) / y_std
        
        # Single epoch of training
        predictions = self._predict_batch(X)
        errors = y_normalized - predictions
        
        # Update last layer
        grad = X.T @ errors / len(X)
        self.weights[-1] += self.learning_rate * grad.T
        self.biases[-1] += self.learning_rate * np.mean(errors, axis=0)
        
        logger.debug(f"Online training completed, buffer size: {len(self.training_data)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Approximate importance using weight magnitudes
        if not self.weights:
            return {}
        
        # Last layer weights indicate feature importance
        importance = np.mean(np.abs(self.weights[-1]), axis=1)
        importance = importance / (np.sum(importance) + 1e-8)
        
        feature_names = [
            "node_type", "row_count", "cost", "table_count", 
            "join_order", "avg_rows", "optimization_time",
            "complexity", "selectivity", "index_usage"
        ]
        
        result = {}
        for i, name in enumerate(feature_names):
            if i < len(importance):
                result[name] = float(importance[i])
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "prediction_count": self.prediction_count,
            "avg_error": self.total_error / max(1, self.prediction_count),
            "max_error": self.max_error,
            "training_samples": len(self.training_data),
            "learning_rate": self.learning_rate,
            "epoch_count": self.epoch_count,
            "weights_shape": [w.shape for w in self.weights]
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import pickle
        
        checkpoint = {
            "weights": self.weights,
            "biases": self.biases,
            "learning_rate": self.learning_rate,
            "training_data": list(self.training_data),
            "statistics": self.get_statistics()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Cost model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        import pickle
        
        if not __import__('os').path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.weights = checkpoint["weights"]
        self.biases = checkpoint["biases"]
        self.learning_rate = checkpoint["learning_rate"]
        self.training_data = deque(checkpoint["training_data"], maxlen=10000)
        
        logger.info(f"Cost model checkpoint loaded from {path}")


class CostModelEnsemble:
    """
    Ensemble of cost models for more robust predictions.
    
    Combines multiple neural cost models with different configurations
    to provide uncertainty estimates and improved accuracy.
    """
    
    def __init__(self, n_models: int = 3):
        """Initialize ensemble with multiple models."""
        self.models = [
            NeuralCostModel(hidden_dims=[64, 32, 16]),
            NeuralCostModel(hidden_dims=[128, 64, 32]),
            NeuralCostModel(hidden_dims=[256, 128, 64])
        ]
        self.n_models = n_models
    
    def predict(self, plan: QueryPlan) -> Tuple[float, float]:
        """
        Predict cost using ensemble.
        
        Returns:
            Tuple of (mean_cost, uncertainty)
        """
        predictions = []
        
        for model in self.models:
            cost, _ = model.predict(plan)
            predictions.append(cost)
        
        mean_cost = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return mean_cost, uncertainty
    
    def train(self, query_history: List[Dict]):
        """Train all models in ensemble."""
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.n_models}")
            model.train(query_history)
