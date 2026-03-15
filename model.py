import torch.nn as nn
from torchvision import models

def build_model(model_name, num_classes, device):
    if model_name == 'convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
