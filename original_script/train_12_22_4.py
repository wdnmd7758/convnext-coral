import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
import os
import time
from tqdm import tqdm
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# ==============================================================================
# 1. 基础配置
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
MODEL_NAME = 'convnext_base'
INPUT_SIZE = 384
TRAIN_ROOT = "/home/huangjunkai/luoshijun/coral_Acropora/dataset/train"
VAL_ROOT   = "/home/huangjunkai/luoshijun/coral_Acropora/dataset/test"
NUM_CLASSES = 49
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8 # 有效 Batch Size = 4 * 8 = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. 损失函数类
# ==============================================================================
class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0, smoothing=0.1, device='cuda'):
        super(CBFocalLoss, self).__init__()
        samples_per_cls = np.array(samples_per_cls)
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.smoothing = smoothing
        self.device = device

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (inputs.size(-1) - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = -(true_dist * log_probs).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = self.weights[targets]
        return (alpha_t * focal_term * ce_loss).mean()

# ==============================================================================
# 3. 日志美化工具
# ==============================================================================
def print_separator(char="=", length=80):
    print(char * length)

def print_class_report(report_dict, class_names):
    header = f"{'Class Name':<30} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Count':<5}"
    print(header)
    print("-" * len(header))
    for name in class_names:
        if name in report_dict:
            d = report_dict[name]
            print(f"{name[:30]:<30} | {d['precision']:<7.4f} | {d['recall']:<7.4f} | {d['f1-score']:<7.4f} | {int(d['support']):<5}")

# ==============================================================================
# 4. 实验核心逻辑
# ==============================================================================
def run_experiment(exp_name, beta_val):
    set_seed(SEED)
    save_dir = os.path.join("./ablation_no_erasing", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print_separator()
    print(f"🚀 启动实验: {exp_name}")
    print(f"配置: Sampler=ON, BETA={beta_val}, 无RandomErasing")
    print_separator()

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

    train_set = datasets.ImageFolder(root=TRAIN_ROOT, transform=train_transform)
    val_set = datasets.ImageFolder(root=VAL_ROOT, transform=val_transform)
    
    class_counts_np = np.bincount(train_set.targets)
    weights = 1. / class_counts_np
    samples_weights = torch.from_numpy(weights[train_set.targets])
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 模型与损失函数
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = CBFocalLoss(class_counts_np.tolist(), beta=beta_val, gamma=2.0, smoothing=LABEL_SMOOTHING, device=DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS-1}]")
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            all_labels, all_preds = [], []
            
            loader = train_loader if phase == 'train' else val_loader
            
            # ✅ 修复点 1: 在 phase 开始前清零梯度
            if phase == 'train':
                optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val':
                        # 2-view TTA
                        outputs = model(inputs)
                        outputs_flip = model(torch.flip(inputs, [3]))
                        outputs = (outputs + outputs_flip) / 2.0
                    else:
                        outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        # ✅ 修复点 2: 缩放 Loss 并反向传播
                        scaled_loss = loss / ACCUMULATION_STEPS
                        scaled_loss.backward()
                        
                        # ✅ 修复点 3: 只有达到累进步数才更新并清零
                        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
                            optimizer.step()
                            optimizer.zero_grad()
                
                # 统计原始 Loss (非 scaled) 用于日志
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / (len(train_set) if phase == 'train' else len(val_set))
            epoch_acc = running_corrects.double() / (len(train_set) if phase == 'train' else len(val_set))
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val':
                report = classification_report(all_labels, all_preds, target_names=val_set.classes, digits=4, zero_division=0, output_dict=True)
                print_class_report(report, val_set.classes)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                    with open(os.path.join(save_dir, 'best_report.txt'), 'w') as f:
                        f.write(classification_report(all_labels, all_preds, target_names=val_set.classes, digits=4))
                    print(f"🎉 新最佳! Acc: {epoch_acc:.4f}")
                
                if epoch_loss < best_val_loss - 0.0005:
                    best_val_loss, patience_counter = epoch_loss, 0
                else:
                    patience_counter += 1
        
        if patience_counter >= 12: 
            print(">>> ⏳ 触发早停"); break
        scheduler.step()

    # 绘图逻辑保持不变
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss']); plt.plot(history['val_loss']); plt.title('Loss'); plt.legend(['Train', 'Val'])
    plt.subplot(1, 2, 2); plt.plot(history['train_acc']); plt.plot(history['val_acc']); plt.title('Acc'); plt.legend(['Train', 'Val'])
    plt.savefig(os.path.join(save_dir, 'curves.png')); plt.close()
    
    print(f"\n📈 Loss 曲线已保存至: {os.path.join(save_dir, 'curves.png')}")
    
    return best_acc

if __name__ == "__main__":
    # 只测试 β=0.999 和 β=0.9999 两个配置
    beta_configs = [
        {"name": "NoErasing_BETA_0_999",  "val": 0.999},
        {"name": "NoErasing_BETA_0_9999", "val": 0.9999},
    ]
    
    final_results = {}
    for cfg in beta_configs:
        acc = run_experiment(cfg["name"], cfg["val"])
        final_results[cfg["name"]] = acc
        
    print_separator()
    print("📋 消融实验总结报告 (无RandomErasing + β对比)")
    print_separator()
    for name, acc in final_results.items():
        print(f"{name:<30}: Best Val Acc = {acc:.4f}")
    print_separator()
