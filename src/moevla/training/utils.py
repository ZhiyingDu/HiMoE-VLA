from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
import math


def build_cosine_decay_schedule_with_wramup(optimizer: Optimizer, peak_lr: int, decay_lr: int, num_warmup_steps: int, num_decay_steps: int) -> LambdaLR:

    def lr_lambda(current_step):
        def linear_warmup_schedule(current_step):
            if current_step <= 0:
                return 1 / (num_warmup_steps + 1)
            frac = 1 - current_step / num_warmup_steps
            return (1 / (num_warmup_steps + 1) - 1) * frac + 1

        def cosine_decay_schedule(current_step):
            step = min(current_step, num_decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / num_decay_steps))
            alpha = decay_lr / peak_lr
            decayed = (1 - alpha) * cosine_decay + alpha
            return decayed

        if current_step < num_warmup_steps:
            return linear_warmup_schedule(current_step)

        return cosine_decay_schedule(current_step)

    return LambdaLR(optimizer, lr_lambda, -1)

def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num