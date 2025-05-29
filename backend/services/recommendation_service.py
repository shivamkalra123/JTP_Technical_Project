import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

from models.prompt_model import Recommendation, DepthRecommendation
from db.mongodb import prefs_collection
from models.classifier_loader import load_classifier

import random
import math


class SimpleClassifier(torch.nn.Module):
    def __init__(self, embed_size, n_classes):
        super(SimpleClassifier, self).__init__()
        self.linear = torch.nn.Linear(embed_size, n_classes)

    def forward(self, x):
        return self.linear(x)



places_df = pd.read_csv("Top Indian Places to Visit.csv")


embedder = SentenceTransformer('all-MiniLM-L6-v2')


classifier_model, labels_list = load_classifier("trained_recomfmendation_model.pth")
label_count = len(labels_list)


def get_text_vector(text):
    return embedder.encode(text, convert_to_tensor=True)


def compose_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"


def extract_keywords(user_prompt: str):
    prompt_lower = user_prompt.lower()
    
    all_cities = places_df['City'].str.lower().unique()
    all_states = places_df['State'].str.lower().unique()
    all_types = places_df['Significance'].str.lower().unique()

    city_found = next((c for c in all_cities if c in prompt_lower), None)
    state_found = next((s for s in all_states if s in prompt_lower), None)
    type_found = next((t for t in all_types if t in prompt_lower), None)

    return city_found, state_found, type_found


async def get_recommendations(user_id: str, prompt: str):
    city, state, sig_type = extract_keywords(prompt)
    results_df = places_df.copy()

    
    if city:
        results_df = results_df[results_df['City'].str.lower() == city]
    if state:
        results_df = results_df[results_df['State'].str.lower() == state]
    if sig_type:
        results_df = results_df[results_df['Significance'].str.lower() == sig_type]

    if results_df.empty:
        results_df = places_df.copy()  

    results_df = results_df.reset_index(drop=True)
    
    user_embed = get_text_vector(prompt)

    results_df['description'] = results_df.apply(compose_description, axis=1)
    desc_vecs = embedder.encode(results_df['description'].tolist(), convert_to_tensor=True)

    with torch.no_grad():
        output = classifier_model(user_embed.unsqueeze(0))
        probs = F.softmax(output, dim=1).squeeze(0)

    predicted_index = torch.argmax(probs).item()
    guessed_class = labels_list[predicted_index]
    print(f"🧠 Classifier thinks this prompt relates to: '{guessed_class}'")

    
    class_prob_map = {labels_list[i]: probs[i].item() for i in range(label_count)}

    
    sim_vals = util.pytorch_cos_sim(user_embed, desc_vecs)[0]

    ranked = []
    for i, row in results_df.iterrows():
        sig = row['Significance']
        prob_val = class_prob_map.get(sig, 0)
        sim_val = sim_vals[i].item()
        score = 0.6 * sim_val + 0.4 * prob_val  
        ranked.append((i, score))

    ranked.sort(key=lambda tup: tup[1], reverse=True)

    top_idxs = [i for i, _ in ranked[:20]]
    final_recos = []
    inferred_count = {}

    for i in top_idxs:
        rec_row = results_df.loc[i]
        sig = rec_row['Significance']
        inferred_count[sig] = inferred_count.get(sig, 0) + 1

        final_recos.append(Recommendation(
            sno=rec_row['sno'],
            name=rec_row['Name'],
            city=rec_row['City'],
            state=rec_row['State'],
            description=rec_row['description'],
            rating=rec_row['Google review rating']
        ))

    
    previous_prefs_doc = await prefs_collection.find_one({"user_id": user_id})
    previous_prefs = previous_prefs_doc.get("inferred_preferences", {}) if previous_prefs_doc else {}

    for sig_type, count in inferred_count.items():
        previous_prefs[sig_type] = previous_prefs.get(sig_type, 0) + count

    await prefs_collection.update_one(
        {"user_id": user_id},
        {"$set": {"inferred_preferences": previous_prefs}},
        upsert=True
    )
    await prefs_collection.update_one(
        {"user_id": user_id},
        {"$push": {"prompts": prompt}},
        upsert=True
    )

    return final_recos



async def get_inferred_based_recommendations(user_id: str):
    user_data = await prefs_collection.find_one({"user_id": user_id})
    if not user_data or "inferred_preferences" not in user_data:
        return []

    pref_scores = user_data["inferred_preferences"]
    sorted_types = sorted(pref_scores.items(), key=lambda kv: kv[1], reverse=True)

    picked_places = []
    seen_keys = set()

    for category, _ in sorted_types:
        temp_df = places_df[places_df['Significance'] == category].copy()
        temp_df = temp_df.sample(frac=1).reset_index(drop=True)  

        count = 0
        for _, row in temp_df.iterrows():
            unique_id = f"{row['Name']}|{row['City']}"
            if unique_id not in seen_keys:
                picked_places.append(Recommendation(
                    sno=row['sno'],
                    name=row['Name'],
                    city=row['City'],
                    state=row['State'],
                    description=compose_description(row),
                    rating=row['Google review rating']
                ))
                seen_keys.add(unique_id)
                count += 1
            if count >= 5:
                break

    return picked_places



def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def safe_float(value):
    if value is None:
        return 0.0
    if isinstance(value, float) and pd.isna(value):
        return 0.0
    return float(value)


async def get_place_by_sno(sno: int):
    if 'sno' in places_df.columns:
        row_match = places_df[places_df['sno'] == sno]
        if row_match.empty:
            return None
        row = row_match.iloc[0]
    else:
        if sno <= 0 or sno > len(places_df):
            return None
        row = places_df.iloc[sno - 1]

    return DepthRecommendation(
        sno=row['sno'],
        name=row['Name'],
        city=row['City'],
        state=row['State'],
        description=compose_description(row),
        rating=float(row['Google review rating']),
        Entrance_Fee=float(row['Entrance Fee in INR']) if not math.isnan(row['Entrance Fee in INR']) else 0.0,
        Establishment_Year=safe_str(row['Establishment Year']),
        Best_Time_to_visit=safe_str(row['Best Time to visit']),
        Airport_with_50km_Radius=safe_str(row['Airport with 50km Radius']),
        dslr_allowed=safe_str(row['DSLR Allowed']),
    )
