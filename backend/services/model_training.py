import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Custom dataset to hold places and their descriptions
class PlacesDataset(Dataset):
    def __init__(self, data_frame, sentence_model, device):
        self.data_frame = data_frame
        self.sentence_model = sentence_model
        self.device = device

        # Encoding significance labels for classification
        self.label_encoder = LabelEncoder()
        self.data_frame['sig_encoded'] = self.label_encoder.fit_transform(self.data_frame['Significance'])

        # Build a description string for each place — yeah, this could be cleaner
        descs = []
        for idx, row in self.data_frame.iterrows():
            desc = f"{row['Name']} - {row['Type']} - {row['Significance']}"
            descs.append(desc)

        # Note: this model requires tensors on the correct device
        self.embeddings = self.sentence_model.encode(descs, convert_to_tensor=True).to(device)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        vector = self.embeddings[index]
        label = self.data_frame.iloc[index]['sig_encoded']
        return vector, label


# Just a super simple linear layer classifier — nothing fancy
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.classifier(x)


def train_model():
    # Load the dataset (make sure this CSV exists!)
    df = pd.read_csv("Top Indian Places to Visit.csv")

    # Using a decent sentence transformer (there are better ones but this is lightweight)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")

    # Initialize our dataset and dataloader
    data = PlacesDataset(df, sbert_model, device)
    loader = DataLoader(data, batch_size=16, shuffle=True)

    emb_dim = data.embeddings.shape[1]
    num_labels = len(data.label_encoder.classes_)

    # Model, loss, optimizer setup
    model = SimpleClassifier(emb_dim, num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    # Let’s train for a few epochs — adjust as needed
    total_epochs = 5
    for ep in range(total_epochs):
        model.train()
        running_loss = 0.0

        for batch in loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Gradient stuff
            opt.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f"Epoch {ep + 1}/{total_epochs}, Avg Loss: {running_loss / len(loader):.4f}")

    # Saving model + label encoder info (super useful for inference later)
    save_data = {
        'model_state_dict': model.state_dict(),
        'label_classes': data.label_encoder.classes_
    }
    torch.save(save_data, "trained_recommendation_model.pth")

# Entry point
if __name__ == "__main__":
    train_model()
