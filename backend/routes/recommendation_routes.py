from fastapi import APIRouter, HTTPException
from models.prompt_model import PromptRequest, Recommendation, DepthRecommendation
from services.recommendation_service import (
    get_recommendations,
    get_inferred_based_recommendations,
    get_place_by_sno
)
from typing import List 
from db.mongodb import prefs_collection

router = APIRouter()

async def verify_user(user_id: str):
    user = await prefs_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

@router.post("/recommend", response_model=list[Recommendation])
async def recommend_for_user(user_id: str, request: PromptRequest):
    await verify_user(user_id)
    return await get_recommendations(user_id, request.prompt)

@router.get("/recommend/inferred", response_model=list[Recommendation])
async def recommend_by_inferred(user_id: str):
    await verify_user(user_id)
    return await get_inferred_based_recommendations(user_id)

@router.get("/place/{sno}", response_model=DepthRecommendation)
async def get_place_details(sno: int):
    place = await get_place_by_sno(sno)
    if place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return place

@router.get("/user/{user_id}/prompts", response_model=List[str])
async def get_user_prompts(user_id: str):
    user_doc = await prefs_collection.find_one({"user_id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    prompts = user_doc.get("prompts", [])
    return prompts