import torch
import os

# ==============================================================================
# 基础配置
# ==============================================================================

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
