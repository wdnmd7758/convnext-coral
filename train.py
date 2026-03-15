import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import from our new modules
from config import *
from utils import set_seed, print_separator, print_class_report
from dataset import get_dataloaders
from model import build_model
from loss import CBFocalLoss

# ==============================================================================
# 核心逻辑
# ==============================================================================
def run_experiment(exp_name, beta_val):
    set_seed(SEED)
    save_dir = os.path.join(".", "ablation_no_erasing", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print_separator()
    print(f"🚀 启动实验: {exp_name}")
    print(f"配置: Sampler=ON, BETA={beta_val}, 无RandomErasing")
    print_separator()

    # 获取数据
    train_loader, val_loader, val_classes, class_counts_np, len_train_set, len_val_set = get_dataloaders(TRAIN_ROOT, VAL_ROOT, BATCH_SIZE)

    # 模型与损失函数
    model = build_model(MODEL_NAME, NUM_CLASSES, DEVICE)

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

            epoch_loss = running_loss / (len_train_set if phase == 'train' else len_val_set)
            epoch_acc = running_corrects.double() / (len_train_set if phase == 'train' else len_val_set)
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val':
                report = classification_report(all_labels, all_preds, target_names=val_classes, digits=4, zero_division=0, output_dict=True)
                print_class_report(report, val_classes)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                    with open(os.path.join(save_dir, 'best_report.txt'), 'w') as f:
                        f.write(classification_report(all_labels, all_preds, target_names=val_classes, digits=4))
                    print(f"🎉 新最佳! Acc: {epoch_acc:.4f}")
                
                if epoch_loss < best_val_loss - 0.0005:
                    best_val_loss, patience_counter = epoch_loss, 0
                else:
                    patience_counter += 1
        
        if patience_counter >= 12: 
            print(">>> ⏳ 触发早停"); break
        scheduler.step()

    # 绘制曲线
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
