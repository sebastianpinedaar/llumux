from typing import List, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ScorerModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'word_count',
            'char_count',
            'special_chars',
            'tokens_per_second',
            'base_score',
            'cost_per_1k',
            'temperature'
        ]

    def prepare_features_batch(
        self,
        prompt: str,
        model_configs: List[Dict],
        temperature: float
    ) -> np.ndarray:
        """
        Prepare features for multiple model configs in a vectorized way.
        
        Args:
            prompt: The input prompt
            model_configs: List of model configurations
            temperature: Temperature setting
            
        Returns:
            2D numpy array of shape (n_configs, n_features)
        """
        n_configs = len(model_configs)
        
        # Compute prompt features once
        word_count = len(prompt.split())
        char_count = len(prompt)
        special_chars = sum(not c.isalnum() for c in prompt) / len(prompt)
        
        # Create arrays for model-specific features
        tokens_per_second = np.array([cfg['avg_tokens_per_second'] for cfg in model_configs])
        base_scores = np.array([cfg['base_performance_score'] for cfg in model_configs])
        costs_per_1k = np.array([cfg['cost_per_1k_tokens'] for cfg in model_configs])
        temperatures = np.full(n_configs, temperature)
        
        # Stack all features into a 2D array
        features = np.column_stack([
            np.full(n_configs, word_count),
            np.full(n_configs, char_count),
            np.full(n_configs, special_chars),
            tokens_per_second,
            base_scores,
            costs_per_1k,
            temperatures
        ])
        
        return features

    def train(
        self,
        training_data: List[Dict]
    ) -> None:
        """
        Train the scorer model on historical data.
        
        Args:
            training_data: List of dictionaries containing:
                - prompt: str
                - model_config: Dict
                - temperature: float
                - actual_performance: float (target variable)
        """
        if not training_data:
            raise ValueError("No training data provided")
            
        X = []
        y = []
        
        # Group training data by prompt to vectorize feature preparation
        prompt_groups = {}
        for entry in training_data:
            prompt = entry['prompt']
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    'configs': [],
                    'temperatures': [],
                    'performances': []
                }
            prompt_groups[prompt]['configs'].append(entry['model_config'])
            prompt_groups[prompt]['temperatures'].append(entry['temperature'])
            prompt_groups[prompt]['performances'].append(entry['actual_performance'])
        
        # Prepare features in batches
        for prompt, group in prompt_groups.items():
            features = self.prepare_features_batch(
                prompt,
                group['configs'],
                np.mean(group['temperatures'])  # Use mean temperature for the batch
            )
            X.append(features)
            y.extend(group['performances'])
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True

    def predict_batch(
        self,
        prompt: str,
        model_configs: List[Dict],
        temperature: float
    ) -> np.ndarray:
        """
        Predict performance scores for multiple model configs at once.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        features = self.prepare_features_batch(prompt, model_configs, temperature)
        scaled_features = self.scaler.transform(features)
        
        predictions = self.model.predict(scaled_features)
        
        # Ensure predictions are between 0 and 1
        return np.clip(predictions, 0, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in making predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 