from typing import List, Dict
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class Recommendation(BaseModel):
    sno:int
    name: str
    city: str
    state: str
    description: str
    rating: float
class DepthRecommendation(BaseModel):
    sno:int
    name: str
    city: str
    state: str
    description: str
    rating: float
    Entrance_Fee: float
    Establishment_Year:str
    Airport_with_50km_Radius:str
    dslr_allowed: str
    Best_Time_to_visit:str

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    classifier_probabilities: Dict[str, float]
