
import math


class NoDecay:
    """
    No decay function.
    """
    def __call__(self, step, total_steps) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "NoDecay"

class CosineDecay:
    """ 
    Cosine decay function, with support for a k-decay parameter
    (https://arxiv.org/pdf/2004.05909).
    """
    def __init__(self, k=1) -> None:
        self.k = k
        
    def __call__(self, step, total_steps) -> float:
        t = step; T = total_steps
        decay_ratio = t**self.k / T**self.k
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Starts at 1 and goes to 0
        return coeff # Always in the range [0, 1]
    
    def __repr__(self) -> str:
        return f"CosineDecay(k={self.k})"
    
class LinearDecay:
    """
    Linear decay function
    """
    def __call__(self, step, total_steps) -> float:
        t = step; T = total_steps
        decay_ratio = min(t / T, 1.0)
        coeff = 1.0 - decay_ratio # Starts at 1 and goes to 0
        return coeff # Always in the range [0, 1]

    def __repr__(self) -> str:
        return "LinearDecay"


class VariableScheduler:
    """
    A variable scheduler with support for warmup and decay functions.

    This scheduler is designed to manage the progression of a variable value
    over a specified number of iterations, including an initial warmup period
    and a decay phase using a specified decay function. It is flexible and can
    be used for any variable that needs to be adjusted over time.
    
    When the it >= total_iterations_provided, the variable value is set to the 
    final value (varf).
    
    Note that the range of decay values goes from [var0, varf). It approaches varf
    but never reaches it, until you set a cooldown period (total_it < total_steps)

    Example usage (Linear decay with warmup over the first 100 steps, with a 
    final cooldown stage of 50 steps with constant value at varf):
    >>> steps = 150
    >>> cooldown_steps = 50
    >>> scheduler = VariableScheduler(var0=1.0, varf=0.0, total_it=steps-cooldown_steps, 
                                      warmup_it=10, decay_fn=LinearDecay())
    >>> for i in range(steps):
    >>>     # Use new_var in your code
    >>>     new_var = scheduler.step()
    """
    def __init__(self, var0, varf, total_it, warmup_it=0, decay_fn=None) -> None:
        """ 
        Args:
            var0: The initial value of the variable.
            varf: The final value of the variable.
            total_it: The total number of iterations over which the variable
                is scheduled to change.
            warmup_it: The number of initial iterations during which the variable
                linearly increases from 0 to var0 (default is 0).
            decay_fn: The function used to compute the decay of the variable after
                the warmup period. If not provided, defaults to NoDecay, which means
                the variable remains constant after the warmup period.
        """
        self.var0 = var0
        self.varf = varf
        self.total_it = total_it
        self.warmup_it = warmup_it
        self.decay_fn = decay_fn if decay_fn else NoDecay()
        self.reset()
        
    def reset(self):
        self.nit = 0
    
    def step(self) -> float:
        # Linear warmup
        if self.nit < self.warmup_it:
            new_var = self.var0 * ((self.nit + 1) / self.warmup_it)
        # Cooldown period if nit > total_it
        elif self.nit >= self.total_it:
            new_var = self.varf
        # Variable decay
        else:
            decay_step = self.nit - self.warmup_it
            decay_steps = self.total_it - self.warmup_it
            coeff = self.decay_fn(decay_step, decay_steps)
            new_var = self.varf + coeff * (self.var0 - self.varf)
        self.nit += 1
        return new_var
    
    def __repr__(self) -> str:
        return (f"VariableScheduler(var0={self.var0}, varf={self.varf}, total_it={self.total_it}, " +
                f"warmup_it={self.warmup_it}, decay_fn={self.decay_fn})")
