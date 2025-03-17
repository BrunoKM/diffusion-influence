from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler


def get_warmup_scheduler(optimizer: Optimizer, warmup_steps: int) -> LRScheduler:
    return LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
