from fastapi import APIRouter, Depends
from models.prompt_model import PromptRequest, Recommendation
from services.recommendation_service import get_recommendations

router = APIRouter()

@router.post("/recommend", response_model=list[Recommendation])
async def recommend_for_user(user_id: str, request: PromptRequest):
    recommendations = await get_recommendations(user_id, request.prompt)
    return recommendations
