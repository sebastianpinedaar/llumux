from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..router.llm_router import LLMRouter
from ..models.recommendation import LLMRecommendation

app = FastAPI(
    title="Fast LLM Router API",
    description="API for intelligent LLM routing and recommendations",
    version="0.1.0"
)

router = LLMRouter()

class PromptRequest(BaseModel):
    prompt: str
    max_budget: Optional[float] = None
    max_time: Optional[float] = None
    temperature: Optional[float] = 0.7

class PerformanceFeedback(BaseModel):
    model_name: str
    response_time: float
    token_count: int
    human_rating: Optional[float] = None

@app.post("/recommendations", response_model=List[LLMRecommendation])
async def get_recommendations(request: PromptRequest):
    """
    Get LLM recommendations based on the input prompt and constraints.
    """
    try:
        recommendations = router.get_recommendations(
            prompt=request.prompt,
            max_budget=request.max_budget,
            max_time=request.max_time,
            temperature=request.temperature
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_performance(feedback: PerformanceFeedback):
    """
    Record performance metrics and human feedback for a model.
    """
    try:
        router.performance_metrics.record_performance(
            model_name=feedback.model_name,
            response_time=feedback.response_time,
            token_count=feedback.token_count,
            human_rating=feedback.human_rating
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/{model_name}")
async def get_model_performance(model_name: str):
    """
    Get performance metrics for a specific model.
    """
    try:
        avg_performance = router.performance_metrics.get_average_performance(model_name)
        performance_trend = router.performance_metrics.get_performance_trend(model_name)
        
        if not avg_performance and not performance_trend:
            raise HTTPException(
                status_code=404,
                detail=f"No performance data found for model {model_name}"
            )
            
        return {
            "average_performance": avg_performance,
            "performance_trend": performance_trend
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 