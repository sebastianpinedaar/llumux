from pydantic import BaseModel

class LLMRecommendation(BaseModel):
    model_name: str
    provider: str
    estimated_cost: float
    estimated_time: float
    performance_score: float
    temperature: float 