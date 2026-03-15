import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from config import INPUT_SIZE

def get_dataloaders(train_root, val_root, batch_size):
    # 数据准备 - 删除了 RandomErasing
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 注意：此处已删除 RandomErasing
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(root=train_root, transform=train_transform)
    val_set = datasets.ImageFolder(root=val_root, transform=val_transform)
    
    class_counts_np = np.bincount(train_set.targets)
    weights = 1. / class_counts_np
    samples_weights = torch.from_numpy(weights[train_set.targets])
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, val_set.classes, class_counts_np, len(train_set), len(val_set)
