from fastapi import APIRouter
from models.prompt_model import PromptRequest, Recommendation
from services.recommendation_service import get_recommendations, get_inferred_based_recommendations

router = APIRouter()

@router.post("/recommend", response_model=list[Recommendation])
async def recommend_for_user(user_id: str, request: PromptRequest):
    return await get_recommendations(user_id, request.prompt)

@router.get("/recommend/inferred", response_model=list[Recommendation])
async def recommend_by_inferred(user_id: str):
    return await get_inferred_based_recommendations(user_id)
