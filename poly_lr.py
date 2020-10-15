from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_decay_steps, final_learning_rate = 1e-4,
            power = 1.0):
        if max_decay_steps <= 1.0:
            raise ValueError('max_decay_steps should be greater than 1.0')
        self.max_decay_steps = max_decay_steps
        self.final_learning_rate = final_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_decay_lrs(self):
        return [(base_lr - self.final_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.final_learning_rate for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.final_learning_rate for _ in self.base_lrs]
        return self.get_decay_lrs()

    def step(self, step = None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = self.get_decay_lrs()
        for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
            param_group['lr'] = lr
