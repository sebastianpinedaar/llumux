import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.llm_router import LLMRouter

def main():
    # Initialize the router
    router = LLMRouter()
    
    # Example 1: Basic recommendation
    prompt = "Explain quantum computing in simple terms."
    print("\nExample 1: Basic recommendation")
    print(f"Prompt: {prompt}")
    
    recommendations = router.get_recommendations(prompt)
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"\nModel: {rec.model_name}")
        print(f"Provider: {rec.provider}")
        print(f"Estimated Cost: ${rec.estimated_cost:.4f}")
        print(f"Estimated Time: {rec.estimated_time:.2f}s")
        print(f"Performance Score: {rec.performance_score:.2f}")
    
    # Example 2: Recommendation with constraints
    prompt = "Write a comprehensive analysis of climate change solutions."
    max_budget = 0.05
    max_time = 10.0
    
    print("\nExample 2: Recommendation with constraints")
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
        print(f"Performance Score: {rec.performance_score:.2f}")
    
    # Example 3: Record performance feedback
    print("\nExample 3: Recording performance feedback")
    router.performance_metrics.record_performance(
        model_name="gpt-4",
        response_time=2.5,
        token_count=150,
        human_rating=0.95
    )
    
    performance_data = router.performance_metrics.get_average_performance("gpt-4")
    print("\nPerformance metrics for GPT-4:")
    print(f"Average response time: {performance_data['avg_response_time']:.2f}s")
    print(f"Average tokens per second: {performance_data['avg_tokens_per_second']:.2f}")
    print(f"Average human rating: {performance_data['avg_human_rating']:.2f}")

if __name__ == "__main__":
    main() 