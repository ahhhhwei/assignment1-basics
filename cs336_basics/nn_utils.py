import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True).values
    # softmax(x-c)=softmax(x) 直接做指数变换数字太大，计算机可能溢出
    exp_x = torch.exp(x - max_value)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_values = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1))
    target_logits = shifted_inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    loss = log_sum_exp - target_logits
    return loss.mean()

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.mul_(scale)