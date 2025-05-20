from services.recommendation_service import get_recommendations


async def recommend_for_user(user_id: str, prompt: str):
    return await get_recommendations(user_id, prompt)
