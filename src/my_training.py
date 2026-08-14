import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import einsum
from einops import rearrange
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math



def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - m 

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    target_logits = shifted_logits[torch.arange(shifted_logits.shape[0]), targets]

    loss = log_sum_exp - target_logits

    return torch.mean(loss)

'''
SGD 是课程内代码，并非我实现
'''
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas =(0.9,0.99),weight_decay=(0.9, 0.95),eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas":betas,
            "weight_decay":weight_decay,
            "eps":eps
        }

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"]+=1

                t = state["t"]
                m = state["m"]
                v = state["v"]

                grad = p.grad.data
                lr_t = lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                p -= lr * weight_decay * p

                m = betas[0]*m + (1-betas[0])*grad
                v = betas[1]*v + (1-betas[1])*(grad**2)
                p -= lr_t*m/(torch.sqrt(v)+eps)

                state["m"]=m
                state["v"]=v
        return loss


def get_lr_cos_schedule(t,a_max,a_min,T_w,T_c):
    if t < T_w:
        a_t = (t/T_w)*a_max

    elif t>=T_w and t<=T_c:
        a_t = a_min + 0.5*(1+math.cos((t-T_w)/(T_c-T_w)*math.pi))*(a_max-a_min)

    elif t>T_c:
        a_t = a_min

    return a_t


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float,eps = 1e-6):
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    sum_sq = torch.tensor(0.)

    for p in params_with_grad:
        sum_sq+=(p.grad.data ** 2).sum().item()

    total_norm = math.sqrt(sum_sq)

    if max_l2_norm<total_norm:
        scale = max_l2_norm / (total_norm + eps)

        for p in params_with_grad:
            p.grad.data *= scale

    return



    




                

