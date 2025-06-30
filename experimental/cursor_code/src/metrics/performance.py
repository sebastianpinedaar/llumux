from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class PerformanceMetrics:
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict]] = {}

    def record_performance(
        self,
        model_name: str,
        response_time: float,
        token_count: int,
        human_rating: Optional[float] = None
    ):
        """
        Record performance metrics for a model.
        
        Args:
            model_name: Name of the LLM
            response_time: Time taken to generate response in seconds
            token_count: Number of tokens in the response
            human_rating: Optional human rating (0-1)
        """
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
            
        self.metrics_history[model_name].append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'token_count': token_count,
            'tokens_per_second': token_count / response_time,
            'human_rating': human_rating
        })

    def get_average_performance(self, model_name: str) -> Dict:
        """
        Calculate average performance metrics for a model.
        """
        if model_name not in self.metrics_history:
            return {}
            
        metrics = self.metrics_history[model_name]
        
        avg_response_time = np.mean([m['response_time'] for m in metrics])
        avg_tokens_per_second = np.mean([m['tokens_per_second'] for m in metrics])
        
        human_ratings = [m['human_rating'] for m in metrics if m['human_rating'] is not None]
        avg_human_rating = np.mean(human_ratings) if human_ratings else None
        
        return {
            'avg_response_time': avg_response_time,
            'avg_tokens_per_second': avg_tokens_per_second,
            'avg_human_rating': avg_human_rating,
            'total_samples': len(metrics)
        }

    def get_performance_trend(self, model_name: str, window_size: int = 10) -> Dict:
        """
        Calculate performance trends over time using moving averages.
        """
        if model_name not in self.metrics_history:
            return {}
            
        metrics = self.metrics_history[model_name]
        if len(metrics) < window_size:
            return {}
            
        response_times = [m['response_time'] for m in metrics]
        tokens_per_second = [m['tokens_per_second'] for m in metrics]
        
        moving_avg_time = np.convolve(
            response_times,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        
        moving_avg_tps = np.convolve(
            tokens_per_second,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        
        return {
            'response_time_trend': moving_avg_time.tolist(),
            'tokens_per_second_trend': moving_avg_tps.tolist()
        } 