import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # dim 沿哪一个维度做 softmax，如果 dim=-1，就是每一行分别做 Softmax。
    max_value = torch.max(x, dim=dim, keepdim=True).values
    # softmax(x-c)=softmax(x) 直接做指数变换数字太大，计算机可能溢出
    exp_x = torch.exp(x - max_value)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

# 交叉熵
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_values = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1))
    target_logits = shifted_inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    loss = log_sum_exp - target_logits
    return loss.mean()

# 梯度裁剪
def gradient_clipping(parameters, max_l2_norm: float) -> None:
    parameters = [p for p in parameters if p.grad is not None] # 过滤掉没有梯度的参数
    total_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters)) # 计算所有梯度的 L2 范数 
    if total_norm > max_l2_norm: # 梯度太大需要裁剪
        scale = max_l2_norm / (total_norm + 1e-6) # 算缩放比例
        for p in parameters:
            p.grad.mul_(scale) 