import torch
from torch import nn

def load_saved_classifier(model_path="trained_recommendation_model.pth", dev=None):
    
    if dev is None:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    
    saved_data = torch.load(model_path, map_location=dev)
    
    all_labels = saved_data['label_classes']  
    dim_of_embedding = 384  
    total_classes = len(all_labels)

    
    clf = nn.Linear(dim_of_embedding, total_classes).to(dev)

    
    clf.load_state_dict(saved_data['model_state_dict'])
    clf.eval()  

    
    return clf, all_labels
