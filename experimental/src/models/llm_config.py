from pydantic import BaseModel

class LLMConfig(BaseModel):
    name: str
    provider: str
    cost_per_1k_tokens: float
    avg_tokens_per_second: float
    base_performance_score: float 