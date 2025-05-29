import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

# Honestly just a quick wrapper for our dataset — might revisit later
class PlaceDataset(Dataset):
    def __init__(self, descriptions, labels, embedding_model):
        self.descriptions = descriptions
        self.labels = labels
        self.embedding_model = embedding_model

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        # NOTE: We're doing embeddings on-the-fly — not the most efficient, but whatever for now
        emb = self.embedding_model.encode(self.descriptions[index])
        return torch.tensor(emb, dtype=torch.float32), self.labels[index]

# Just gluing name/type/significance into one description string
def build_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"

def train_and_save_model(csv_path="Top Indian Places to Visit.csv", save_path="trained_recommendation_model.pth"):
    # Step 1: Load the data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"💥 Failed to read CSV: {e}")
        return

    # Combine relevant columns into a single "description"
    df['description'] = df.apply(build_description, axis=1)

    # Step 2: Convert 'Significance' column into integer labels
    label_encoder = LabelEncoder()
    try:
        label_ids = label_encoder.fit_transform(df['Significance'])
    except Exception as e:
        print("⚠️ Label encoding error. Check if 'Significance' column is clean.")
        raise e

    label_names = list(label_encoder.classes_)

    # Step 3: Load a sentence embedding model
    print("🔍 Loading embedding model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Dataset & dataloader — yes we're doing this old-school
    dataset = PlaceDataset(df['description'].tolist(), label_ids, sentence_model)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 4: Define the classifier model
    embedding_dim = sentence_model.get_sentence_embedding_dimension()
    num_classes = len(label_names)
    classifier = nn.Linear(embedding_dim, num_classes)  # basic softmax classifier

    # Some reasonable training defaults
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    num_epochs = 5

    print("🚀 Starting training loop...")
    classifier.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in loader:
            inputs, targets = batch
            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"🌀 Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}")

    # Step 5: Save model + label metadata
    checkpoint = {
        'model_state_dict': classifier.state_dict(),
        'label_classes': label_names
    }

    try:
        torch.save(checkpoint, save_path)
        print(f"✅ Model saved at: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")

# Only run training if script is executed directly
if __name__ == "__main__":
    train_and_save_model()
