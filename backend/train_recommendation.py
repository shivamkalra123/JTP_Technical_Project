import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class PlacesDataset(Dataset):
    def __init__(self, texts, labels, model):
        self.texts = texts
        self.labels = labels
        self.model = model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        embedding = self.model.encode(self.texts[idx])
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), label

def get_place_description(row):
    return f"{row['Name']} - {row['Type']} - {row['Significance']}"

def train():
    
    df = pd.read_csv("Top Indian Places to Visit.csv")
    df['description'] = df.apply(get_place_description, axis=1)

    
    le = LabelEncoder()
    labels = le.fit_transform(df['Significance'])
    label_classes = list(le.classes_)

    
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    
    dataset = PlacesDataset(df['description'].tolist(), labels, sbert)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    embedding_dim = sbert.get_sentence_embedding_dimension()
    num_classes = len(label_classes)

    
    classifier = nn.Linear(embedding_dim, num_classes)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    epochs = 5

    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, targets in train_loader:
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'label_classes': label_classes
    }, "trained_recommendation_model.pth")

if __name__ == "__main__":
    train()
