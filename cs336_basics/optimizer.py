import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params, lr: float=1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # 第一次遇到这个参数，初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1

                step = state["step"]
                m = state["m"]
                v = state["v"]

                # 一阶动量
                m.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # 二阶动量
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # bias correction
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                # Adam 更新
                p.addcdiv_(
                    m_hat,
                    torch.sqrt(v_hat) + eps,
                    value=-lr,
                )

                # Weight Decay
                p.mul_(1 - lr * weight_decay)

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    # 第一阶段：linear warmup
    if it < warmup_iters:
        return (
            max_learning_rate
            * it
            / warmup_iters
        )

    # 第三阶段：cosine 已经结束
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 第二阶段：cosine decay
    progress = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    return (
        min_learning_rate
        + 0.5
        * (max_learning_rate - min_learning_rate)
        * (1 + math.cos(math.pi * progress))
    )