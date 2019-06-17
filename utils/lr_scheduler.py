from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

class WarmupStepLR(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        lr = super().get_lr()
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]
        return lr
        
class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        lr = super().get_lr()
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]
        return lr

class ReduceLROnPerform(ReduceLROnPlateau):
    def __init__(self, optimizer, mode_min_max = 'min', lr_factor = 0.5, step_patience = 1, lr_update_threshold = 1e-4, min_lr = 0.00001):
        super().__init__(optimizer = optimizer, mode = mode_min_max, factor = lr_factor, patience = step_patience, threshold = lr_update_threshold, min_lr = min_lr)

    def get_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = float(param_group['lr'])
        return lr