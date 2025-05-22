import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from models.prompt_model import Recommendation
from db.mongodb import prefs_collection
from models.classifier_loader import load_classifier


class SimpleClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


df = pd.read_csv("Top Indian Places to Visit.csv")


sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

classifier, label_classes = load_classifier()
num_classes = len(label_classes)

def get_embedding(text):
    return sbert_model.encode(text, convert_to_tensor=True)

def get_place_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"

def extract_keywords(prompt: str):
  
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

    
    filtered_df = filtered_df.reset_index(drop=True)

    query_embedding = get_embedding(prompt)
    filtered_df['description'] = filtered_df.apply(get_place_description, axis=1)
    desc_embeddings = sbert_model.encode(filtered_df['description'].tolist(), convert_to_tensor=True)

    
    with torch.no_grad():
        output = classifier(query_embedding.unsqueeze(0))  
        probs = F.softmax(output, dim=1).squeeze(0)

    sig_prob_map = {label_classes[i]: probs[i].item() for i in range(num_classes)}

    
    sim_scores = util.pytorch_cos_sim(query_embedding, desc_embeddings)[0]

    combined_scores = []
    for idx, row in filtered_df.iterrows():  
        significance = row['Significance']
        prob_score = sig_prob_map.get(significance, 0)
        sim_score = sim_scores[idx].item()   
        combined_score = 0.6 * sim_score + 0.4 * prob_score
        combined_scores.append((idx, combined_score))
        

    
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    print(combined_score)
    
    top_indices = [idx for idx, _ in combined_scores[:9]]

    recommendations = []
    inferred = {}

    for idx in top_indices:
        row = filtered_df.loc[idx]
        significance = row['Significance']
        inferred[significance] = inferred.get(significance, 0) + 1

        recommendations.append(Recommendation(
            name=row['Name'],
            city=row['City'],
            state=row['State'],
            description=row['description'],
            rating=row['Google review rating']
        ))

    
    existing_doc = await prefs_collection.find_one({"user_id": user_id})
    existing_prefs = existing_doc.get("inferred_preferences", {}) if existing_doc else {}

    for k, v in inferred.items():
        existing_prefs[k] = existing_prefs.get(k, 0) + v

    await prefs_collection.update_one(
        {"user_id": user_id},
        {"$set": {"inferred_preferences": existing_prefs}},
        upsert=True
    )

    return recommendations

async def get_inferred_based_recommendations(user_id: str):
    user = await prefs_collection.find_one({"user_id": user_id})
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
