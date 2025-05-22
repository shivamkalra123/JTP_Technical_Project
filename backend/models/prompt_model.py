from typing import List, Dict
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class Recommendation(BaseModel):
    name: str
    city: str
    state: str
    description: str
    rating: float

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    classifier_probabilities: Dict[str, float]
