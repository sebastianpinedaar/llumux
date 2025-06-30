from typing import List, Optional, Dict
from pydantic import BaseModel
import numpy as np
from ..metrics.performance import PerformanceMetrics
from ..models.recommendation import LLMRecommendation
from ..models.llm_config import LLMConfig
from ..models.scorer import ScorerModel

class LLMRouter:
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.scorer = ScorerModel()
        self.llm_configs: Dict[str, LLMConfig] = {
            "gpt-4": LLMConfig(
                name="gpt-4",
                provider="openai",
                cost_per_1k_tokens=0.03,
                avg_tokens_per_second=15,
                base_performance_score=0.95
            ),
            "gpt-3.5-turbo": LLMConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                cost_per_1k_tokens=0.002,
                avg_tokens_per_second=40,
                base_performance_score=0.85
            ),
            "claude-2": LLMConfig(
                name="claude-2",
                provider="anthropic",
                cost_per_1k_tokens=0.01,
                avg_tokens_per_second=25,
                base_performance_score=0.90
            )
        }

    def train_scorer(self, training_data: List[Dict]) -> None:
        """
        Train the scorer model with historical performance data.
        """
        self.scorer.train(training_data)

    def get_recommendations(
        self,
        prompt: str,
        max_budget: Optional[float] = None,
        max_time: Optional[float] = None,
        temperature: Optional[float] = 0.7
    ) -> List[LLMRecommendation]:
        """
        Get LLM recommendations based on the input prompt and constraints.
        
        Args:
            prompt: The input prompt to analyze
            max_budget: Maximum budget in USD
            max_time: Maximum generation time in seconds
            temperature: Sampling temperature for generation
            
        Returns:
            List of LLM recommendations sorted by suitability
        """
        # Convert configs to list for vectorized operations
        model_names = list(self.llm_configs.keys())
        configs = [self.llm_configs[name].dict() for name in model_names]
        
        # Vectorized token estimation
        estimated_tokens = len(prompt.split()) * 1.3  # Simple estimation
        
        # Vectorized cost and time calculations
        costs_per_1k = np.array([cfg['cost_per_1k_tokens'] for cfg in configs])
        tokens_per_second = np.array([cfg['avg_tokens_per_second'] for cfg in configs])
        
        estimated_costs = (estimated_tokens / 1000) * costs_per_1k
        estimated_times = estimated_tokens / tokens_per_second
        
        # Apply constraints
        valid_indices = np.ones(len(configs), dtype=bool)
        if max_budget is not None:
            valid_indices &= estimated_costs <= max_budget
        if max_time is not None:
            valid_indices &= estimated_times <= max_time
            
        if not np.any(valid_indices):
            return []
            
        # Get performance scores for valid models
        valid_configs = [cfg for i, cfg in enumerate(configs) if valid_indices[i]]
        try:
            performance_scores = self.scorer.predict_batch(
                prompt,
                valid_configs,
                temperature
            )
        except RuntimeError:
            # Fallback to simple scoring if model isn't trained
            performance_scores = np.array([
                self._calculate_simple_score(
                    self.llm_configs[model_names[i]],
                    prompt,
                    temperature
                )
                for i in range(len(configs))
                if valid_indices[i]
            ])
        
        # Create recommendations for valid models
        recommendations = []
        valid_models = [model_names[i] for i in range(len(configs)) if valid_indices[i]]
        
        for i, (model_name, score) in enumerate(zip(valid_models, performance_scores)):
            config = self.llm_configs[model_name]
            recommendations.append(
                LLMRecommendation(
                    model_name=model_name,
                    provider=config.provider,
                    estimated_cost=estimated_costs[valid_indices][i],
                    estimated_time=estimated_times[valid_indices][i],
                    performance_score=score,
                    temperature=temperature
                )
            )
        
        # Sort by performance score and cost efficiency
        recommendations.sort(
            key=lambda x: (x.performance_score, -x.estimated_cost),
            reverse=True
        )
        
        return recommendations

    def _calculate_simple_score(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float
    ) -> float:
        """
        Calculate a simple performance score when the scorer model isn't trained.
        """
        prompt_complexity = self._analyze_prompt_complexity(prompt)
        complexity_factor = 1 - (prompt_complexity * 0.2)
        temperature_factor = 1 - (abs(0.7 - temperature) * 0.3)
        
        return config.base_performance_score * complexity_factor * temperature_factor

    def _analyze_prompt_complexity(self, prompt: str) -> float:
        """
        Analyze the complexity of the input prompt.
        Returns a score between 0 and 1.
        """
        # Simple complexity analysis based on length and special characters
        length_score = min(len(prompt) / 1000, 1.0)
        special_chars = sum(not c.isalnum() for c in prompt) / len(prompt)
        return (length_score + special_chars) / 2

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in making predictions.
        """
        return self.scorer.get_feature_importance() 