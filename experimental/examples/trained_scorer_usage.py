import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.llm_router import LLMRouter
import random
from datetime import datetime, timedelta

def generate_sample_training_data(n_samples=100):
    """
    Generate synthetic training data for demonstration purposes.
    In real applications, this would be replaced with actual historical data.
    """
    prompts = [
        "Explain quantum computing to a 5-year-old",
        "Write a complex analysis of global economic trends",
        "Generate a simple greeting message",
        "Translate this paragraph to French",
        "Debug this Python code snippet",
        "Write a research paper introduction",
        "Create a short story about space exploration",
        "Summarize the main points of this article"
    ]
    
    training_data = []
    
    for _ in range(n_samples):
        prompt = random.choice(prompts)
        model_config = {
            'name': random.choice(['gpt-4', 'gpt-3.5-turbo', 'claude-2']),
            'provider': random.choice(['openai', 'anthropic']),
            'cost_per_1k_tokens': random.uniform(0.001, 0.05),
            'avg_tokens_per_second': random.uniform(10, 50),
            'base_performance_score': random.uniform(0.8, 0.99)
        }
        
        # Simulate actual performance based on prompt and model characteristics
        base_performance = model_config['base_performance_score']
        complexity_factor = 1 - (len(prompt.split()) / 100)
        random_variation = random.uniform(-0.1, 0.1)
        
        actual_performance = max(0.0, min(1.0, base_performance * complexity_factor + random_variation))
        
        training_data.append({
            'prompt': prompt,
            'model_config': model_config,
            'temperature': random.uniform(0.1, 1.0),
            'actual_performance': actual_performance
        })
    
    return training_data

def main():
    # Initialize the router
    router = LLMRouter()
    
    # Generate and load training data
    print("Generating synthetic training data...")
    training_data = generate_sample_training_data()
    
    # Train the scorer model
    print("Training the scorer model...")
    router.train_scorer(training_data)
    
    # Get feature importance
    print("\nFeature Importance:")
    for feature, importance in router.get_feature_importance().items():
        print(f"{feature}: {importance:.4f}")
    
    # Example 1: Simple prompt
    prompt = "Hello, how are you?"
    print(f"\nExample 1: Simple prompt")
    print(f"Prompt: {prompt}")
    
    recommendations = router.get_recommendations(prompt)
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"\nModel: {rec.model_name}")
        print(f"Provider: {rec.provider}")
        print(f"Estimated Cost: ${rec.estimated_cost:.4f}")
        print(f"Estimated Time: {rec.estimated_time:.2f}s")
        print(f"Performance Score: {rec.performance_score:.4f}")
    
    # Example 2: Complex prompt with constraints
    prompt = "Write a detailed technical analysis of machine learning algorithms, including their mathematical foundations, practical applications, and current research directions."
    max_budget = 0.05
    max_time = 10.0
    
    print(f"\nExample 2: Complex prompt with constraints")
    print(f"Prompt: {prompt}")
    print(f"Max Budget: ${max_budget}")
    print(f"Max Time: {max_time}s")
    
    recommendations = router.get_recommendations(
        prompt,
        max_budget=max_budget,
        max_time=max_time
    )
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"\nModel: {rec.model_name}")
        print(f"Provider: {rec.provider}")
        print(f"Estimated Cost: ${rec.estimated_cost:.4f}")
        print(f"Estimated Time: {rec.estimated_time:.2f}s")
        print(f"Performance Score: {rec.performance_score:.4f}")

if __name__ == "__main__":
    main() 