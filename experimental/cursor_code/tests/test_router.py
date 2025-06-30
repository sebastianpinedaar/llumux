import pytest
from src.router.llm_router import LLMRouter
from src.models.llm_config import LLMConfig

@pytest.fixture
def router():
    return LLMRouter()

def test_get_recommendations_basic(router):
    prompt = "What is the capital of France?"
    recommendations = router.get_recommendations(prompt)
    
    assert len(recommendations) > 0
    assert all(hasattr(rec, 'model_name') for rec in recommendations)
    assert all(hasattr(rec, 'estimated_cost') for rec in recommendations)
    assert all(hasattr(rec, 'estimated_time') for rec in recommendations)
    assert all(hasattr(rec, 'performance_score') for rec in recommendations)

def test_get_recommendations_with_constraints(router):
    prompt = "Write a long essay about AI."
    max_budget = 0.01
    max_time = 5.0
    
    recommendations = router.get_recommendations(
        prompt,
        max_budget=max_budget,
        max_time=max_time
    )
    
    assert all(rec.estimated_cost <= max_budget for rec in recommendations)
    assert all(rec.estimated_time <= max_time for rec in recommendations)

def test_prompt_complexity_analysis(router):
    simple_prompt = "Hello world"
    complex_prompt = "Analyze the socioeconomic implications of AI in healthcare, considering ethical frameworks and regulatory challenges!"
    
    simple_score = router._analyze_prompt_complexity(simple_prompt)
    complex_score = router._analyze_prompt_complexity(complex_prompt)
    
    assert 0 <= simple_score <= 1
    assert 0 <= complex_score <= 1
    assert complex_score > simple_score

def test_performance_score_calculation(router):
    config = LLMConfig(
        name="test-model",
        provider="test",
        cost_per_1k_tokens=0.01,
        avg_tokens_per_second=20,
        base_performance_score=0.9
    )
    
    score_low_complexity = router._calculate_performance_score(
        config,
        prompt_complexity=0.2,
        temperature=0.7
    )
    
    score_high_complexity = router._calculate_performance_score(
        config,
        prompt_complexity=0.8,
        temperature=0.7
    )
    
    assert 0 <= score_low_complexity <= 1
    assert 0 <= score_high_complexity <= 1
    assert score_low_complexity > score_high_complexity 