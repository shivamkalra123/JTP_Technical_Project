import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from models.prompt_model import Recommendation
from db.mongodb import prefs_collection


df = pd.read_csv("Top Indian Places to Visit.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

def get_place_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"

def extract_keywords(prompt: str):
    """Simple keyword extractor for city, state, and type from the prompt."""
    prompt = prompt.lower()
    cities = df['City'].str.lower().unique()
    states = df['State'].str.lower().unique()
    types = df['Significance'].str.lower().unique()

    found_city = next((c for c in cities if c in prompt), None)
    found_state = next((s for s in states if s in prompt), None)
    found_type = next((t for t in types if t in prompt), None)

    return found_city, found_state, found_type

async def get_recommendations(user_id: str, prompt: str):

    city, state, sig_type = extract_keywords(prompt)
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df['City'].str.lower() == city]
    if state:
        filtered_df = filtered_df[filtered_df['State'].str.lower() == state]
    if sig_type:
        filtered_df = filtered_df[filtered_df['Significance'].str.lower() == sig_type]

    if filtered_df.empty:
        filtered_df = df.copy()

    query_embedding = get_embedding(prompt)

    filtered_df['description'] = filtered_df.apply(get_place_description, axis=1)
    desc_embeddings = model.encode(filtered_df['description'].tolist(), convert_to_tensor=True)

    # Similarity scoring
    scores = util.pytorch_cos_sim(query_embedding, desc_embeddings)[0]
    top_indices = torch.topk(scores, k=min(8, len(filtered_df))).indices.tolist()

    # Prepare recommendations and inferred preferences
    recommendations = []
    inferred = {}

    for idx in top_indices:
        row = filtered_df.iloc[idx]
        significance = row['Significance']
        inferred[significance] = inferred.get(significance, 0) + 1

        recommendations.append(Recommendation(
            name=row['Name'],
            city=row['City'],
            state=row['State'],
            description=row['description'],
            rating=row['Google review rating']
        ))

    # Fetch existing preferences
    existing_doc = await prefs_collection.find_one({"user_id": user_id})
    existing_prefs = existing_doc.get("inferred_preferences", {}) if existing_doc else {}

    # Append to existing preferences
    for k, v in inferred.items():
        existing_prefs[k] = existing_prefs.get(k, 0) + v

    # Save back to DB
    await prefs_collection.update_one(
        {"user_id": user_id},
        {"$set": {"inferred_preferences": existing_prefs}},
        upsert=True
    )

    return recommendations

async def get_inferred_based_recommendations(user_id: str):
    # Fetch user prefs
    user = await prefs_collection.find_one({ "user_id": user_id })
    if not user or "inferred_preferences" not in user:
        return []

    type_scores = user["inferred_preferences"]
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)

    recommended = []
    seen = set()

    for t, _ in sorted_types:
        filtered = df[df['Significance'] == t].sort_values(by='Google review rating', ascending=False)

        count = 0
        for _, row in filtered.iterrows():
            key = f"{row['Name']}|{row['City']}"
            if key not in seen:
                recommended.append(Recommendation(
                    name=row['Name'],
                    city=row['City'],
                    state=row['State'],
                    description=get_place_description(row),
                    rating=row['Google review rating']
                ))
                seen.add(key)
                count += 1
            if count >= 5:
                break  

    return recommended
