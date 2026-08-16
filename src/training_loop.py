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
import numpy as np


def get_batch(x,batch_size,context_length,device = None):

    n = len(x)
    m = context_length

    maximun = n-m

    pts = np.random.randint(0, maximun, size=batch_size)

    ix = pts[:, None] + np.arange(m)

    inputs_np = x[ix]
    targets_np = x[ix+1]

    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model,optimizer,iteration,out):
    model_weight = model.state_dict()
    optimizer_weight = optimizer.state_dict()

    dic = {}
    dic["model"] = model_weight
    dic["optimizer"] = optimizer_weight
    dic["iteration"] = iteration

    torch.save(dic,out)

    return

def load_checkpoint(src, model, optimizer=None, map_location=None):
    if map_location is None:
        map_location = next(model.parameters()).device
    dic = torch.load(src, map_location=map_location, weights_only=False)

    model.load_state_dict(dic["model"])
    if optimizer is not None:
        optimizer.load_state_dict(dic["optimizer"])

    return dic["iteration"]