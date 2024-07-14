
import math


class CosineDecay:
    """ 
    Cosine decay function for learning rate scheduling, with support for a 
    k-decay parameter (https://arxiv.org/pdf/2004.05909).
    """
    def __init__(self, k=1) -> None:
        self.k = k
        
    def __call__(self, step, total_steps) -> float:
        t = step; T = total_steps
        decay_ratio = t**self.k / T**self.k
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Starts at 1 and goes to 0
        return coeff # Always in the range [0, 1]
    
    
class LinearDecay:
    """
    Linear decay function for learning rate scheduling.
    """
    def __call__(self, step, total_steps) -> float:
        t = step; T = total_steps
        decay_ratio = min(t / T, 1)
        coeff = 1 - decay_ratio # Starts at 1 and goes to 0
        return coeff # Always in the range [0, 1]


class LRScheduler:
    """
    Learning rate scheduler.
    """
    
    def __init__(self, lr0, lrf, total_it, warmup_it=0, decay_fn=None) -> None:
        """ 
        Args:
            lr0: Initial learning rate.
            lrf: Final learning rate.
            total_it: Total number of iterations.
            warmup_it: Number of warmup iterations.
            decay_fn: Decay function to use.
        """
        self.lr0 = lr0
        self.lrf = lrf
        self.total_it = total_it
        self.warmup_it = warmup_it
        self.decay_fn = decay_fn if decay_fn else lambda step, total: 1.0
        self.reset()
        
    def reset(self):
        self.nit = 0
    
    def step(self) -> float:
        # Linear warmup
        if self.nit < self.warmup_it:
            return self.lr0 * ((self.nit + 1) / self.warmup_it)
        # Cooldown period if nit > total_it
        elif self.nit > self.total_it:
            return self.lrf
        # Lr decay
        decay_step = self.nit - self.warmup_it
        decay_steps = self.total_it - self.warmup_it
        coeff = self.decay_fn(decay_step, decay_steps)
        new_lr = self.lrf + coeff * (self.lr0 - self.lrf)
        self.nit += 1
        return new_lr