import math
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimzier, warmup_epoch, max_epoch,
            warmup_start_lr = 0.0, eta_min = 0.0, last_epoch = -1):
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer,
                last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epoch:
            return [group['lr'] +
                    (base_lr - self.warmup_start_lr) / (self.warmup_epoch - 1)
                    for base_lr, group in zip(self.base_lrs,
                        self.optimizer.param_groups)]
        elif self.last_epoch == self.warmup_epoch:
            return self.base_lrs
        elif (self.last_epoch - self.max_epoch - 1) % \
                (2 * (self.max_epoch - self.warmup_epoch)) == 0:
            return [group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(
                math.pi / (self.max_epoch - self.warmup_epoch))) / 2
                for base_lr, group in zip(self.base_lrs,
                        self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch /
            (self.max_epoch - self.warmup_epoch))) /
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch - 1) /
                (self.max_epoch - self.warmup_epoch))) *
            (group['lr'] - self.eta_min) + self.eta_min)
            for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [self.warmup_start_lr + self.last_epoch * (base_lr -
                self.warmup_start_lr) /
                (self.warmup_epoch - 1) for base_lr in self.base_lrs]
        return [self.eta_min + 0.5 * (basr_lr - self.eta_min) * (1 + math.cos(
            math.pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch -
                self.warmup_epoch))) for base_lr in self.base_lrs]
