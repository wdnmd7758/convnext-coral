import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
