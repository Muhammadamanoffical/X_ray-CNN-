import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def load_model():
    model = mobilenet_v2(pretrained=False)

    # 🔴 EXACT SAME CLASSIFIER AS TRAINING
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, 2)
    )

    # Load weights
    state_dict = torch.load(
        "best_model.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model
