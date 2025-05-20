import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from models.prompt_model import Recommendation
from db.mongodb import prefs_collection

# Load data and model once
df = pd.read_csv("Top Indian Places to Visit.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

def get_place_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"

async def get_recommendations(user_id: str, prompt: str):
    # Embed the prompt
    query_embedding = get_embedding(prompt)

    # Prepare place descriptions
    df['description'] = df.apply(get_place_description, axis=1)
    desc_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)

    # Similarity scoring
    scores = util.pytorch_cos_sim(query_embedding, desc_embeddings)[0]
    top_indices = torch.topk(scores, k=8).indices.tolist()

    # Generate recommendations and infer preferences
    recommendations = []
    inferred = {}

    for idx in top_indices:
        row = df.iloc[idx]
        inferred[row['Type']] = inferred.get(row['Type'], 0) + 1
        recommendations.append(Recommendation(
            name=row['Name'],
            city=row['City'],
            state=row['State'],
            description=row['description'],
            rating=row['Google review rating']
        ))

    # Save top 5 inferred preferences to DB
    top_inferred = dict(sorted(inferred.items(), key=lambda x: x[1], reverse=True)[:5])
    await prefs_collection.insert_one({
        "user_id": user_id,
        "inferred_preferences": top_inferred
    })

    return recommendations

async def get_inferred_based_recommendations(user_id: str):
    user = await prefs_collection.find_one({ "userid": user_id })
    if not user or "inferred_preferences" not in user:
        return []

    type_scores = user["inferred_preferences"]

    # Sort preferences by strength
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)

    recommended = []
    seen = set()

    for t, _ in sorted_types[:3]:
        filtered = df[df['Type'] == t].sort_values(by='Google review rating', ascending=False)

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

    return recommended
