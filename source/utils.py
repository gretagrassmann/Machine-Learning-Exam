import random
import os
import torch
from torch import nn
import numpy as np


class FocalLoss(nn.Module):
    # nn.Module: base class for all neural network modules.

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        # super supports cooperative multiple inheritance in a dynamic execution environment.
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input [batch_size,2]. target [batch_size]
        target = target.float()
        pt = torch.softmax(input, dim=1)
        # Applies the Softmax to a n-dimensional input Tensor rescaling the elements so that they lie in the range
        # [0,1] and sum to 1 (in each raw)
        p = pt[:, 1]  # [batch_size]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


class Option(object):
    def __init__(self, d):
        self.__dict__ = d


def seed_set(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
