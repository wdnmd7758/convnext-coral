import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
