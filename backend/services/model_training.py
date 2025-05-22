import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

class PlacesDataset(Dataset):
    def __init__(self, df, model, device):
        self.df = df
        self.model = model
        self.device = device
        
        # Encode Significance labels as integers
        self.le = LabelEncoder()
        self.df['sig_encoded'] = self.le.fit_transform(df['Significance'])

        # Precompute embeddings for place descriptions
        self.descriptions = df.apply(lambda row: f"{row['Name']} - {row['Type']} - {row['Significance']}", axis=1).tolist()
        self.embeddings = model.encode(self.descriptions, convert_to_tensor=True).to(device)  # 🔥 Move embeddings to device
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.df.iloc[idx]['sig_encoded']
        return embedding, label

class SimpleClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def train_model():
    # Load data
    df = pd.read_csv("Top Indian Places to Visit.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PlacesDataset(df, model, device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    embedding_dim = dataset.embeddings.shape[1]
    num_classes = len(dataset.le.classes_)

    classifier = SimpleClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save the model and label encoder classes
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'label_classes': dataset.le.classes_
    }, "trained_recommendation_model.pth")

if __name__ == "__main__":
    train_model()
