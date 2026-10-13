import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    checkpoint = {
        "model": model.state_dict(), # 返回模型当前的参数状态
        "optimizer": optimizer.state_dict(), # 保存优化器当前状态
        "iteration": iteration # 当前训练到第几步了
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]