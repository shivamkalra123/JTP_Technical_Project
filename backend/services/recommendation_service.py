import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from models.prompt_model import Recommendation
from db.mongodb import prefs_collection
from bson import ObjectId

# Determine device: MPS for Mac, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset and model once
df = pd.read_csv("Top Indian Places to Visit.csv")
df['description'] = df['Name'].astype(str) + " - " + df['Type'].astype(str) + " - " + df['Significance'].astype(str)

model = SentenceTransformer('all-MiniLM-L6-v2')
place_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True, device=device)


async def build_user_profile_vector(user_id: str):
    # Fetch preferences only for this user from DB using $userid key
    prefs = await prefs_collection.find({ "userid": user_id }).to_list(None)


    # Aggregate inferred preferences from all past prompts by user
    profile_weights = {}
    for pref in prefs:
        for place_type, weight in pref.get("inferred_preferences", {}).items():
            profile_weights[place_type] = profile_weights.get(place_type, 0) + weight

    # Normalize weights so total sums to 1
    total = sum(profile_weights.values())
    if total > 0:
        for k in profile_weights:
            profile_weights[k] /= total

    return profile_weights


async def get_recommendations(user_id: str, prompt: str):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True, device=device)
    cosine_scores = util.cos_sim(prompt_embedding, place_embeddings)[0]
    top_indices = cosine_scores.argsort(descending=True)

    recommendations = []
    seen_types = set()
    type_first_seen_order = []
    types_counter = {}

    # Get top 20 for analysis
    for idx in top_indices[:20].cpu().numpy():
        row = df.iloc[idx]
        type_name = row['Significance']
        types_counter[type_name] = types_counter.get(type_name, 0) + 1

        if type_name not in seen_types:
            type_first_seen_order.append(type_name)
            seen_types.add(type_name)

        recommendations.append(Recommendation(
            name=row['Name'],
            city=row['City'],
            state=row['State'],
            description=row['description'],
            rating=row['Google review rating']
        ))

    # Store only top 5 unique types (by order of relevance)
    top_types = {t: types_counter[t] for t in type_first_seen_order[:5]}

    preference_doc = {
        "userid": user_id,
        "prompt": prompt,
        "inferred_preferences": top_types
    }
    await prefs_collection.insert_one(preference_doc)

    return recommendations
