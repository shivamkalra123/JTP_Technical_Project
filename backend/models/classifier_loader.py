import torch
from torch import nn

def load_classifier(checkpoint_path="trained_recommendation_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    label_classes = checkpoint['label_classes']
    embedding_dim = 384  
    num_classes = len(label_classes)

    classifier = nn.Linear(embedding_dim, num_classes).to(device)  
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    return classifier, label_classes  

