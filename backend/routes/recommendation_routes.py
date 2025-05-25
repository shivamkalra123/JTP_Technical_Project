from fastapi import APIRouter
from models.prompt_model import PromptRequest, Recommendation, DepthRecommendation
from services.recommendation_service import get_recommendations, get_inferred_based_recommendations,get_place_by_sno

router = APIRouter()

@router.post("/recommend", response_model=list[Recommendation])
async def recommend_for_user(user_id: str, request: PromptRequest):
    return await get_recommendations(user_id, request.prompt)

@router.get("/recommend/inferred", response_model=list[Recommendation])
async def recommend_by_inferred(user_id: str):
    return await get_inferred_based_recommendations(user_id)
@router.get("/place/{sno}", response_model=DepthRecommendation)
async def get_place_details(sno: int):
    place = await get_place_by_sno(sno)
    if place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return place