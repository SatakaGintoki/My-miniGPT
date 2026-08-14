import math
import numpy as np
import torch

from src.my_model import Transformer_LM, get_device
from src.my_training import (
    AdamW,
    cross_entropy,
    get_lr_cos_schedule,
    gradient_clipping,
)
from src.training_loop import (
    get_batch,
    save_checkpoint,
    load_checkpoint,
)
from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS

#----------------超参数--------------------------------------

vocab_size = MODEL_CONFIG["vocab_size"]
context_length = MODEL_CONFIG["context_length"]
num_layers = MODEL_CONFIG["num_layers"]
d_model = MODEL_CONFIG["d_model"]
num_heads = MODEL_CONFIG["num_heads"]
d_ff = MODEL_CONFIG["d_ff"]
rope_theta = MODEL_CONFIG["rope_theta"]

batch_size = TRAIN_CONFIG["batch_size"]
max_steps = TRAIN_CONFIG["max_steps"]
max_lr = TRAIN_CONFIG["max_lr"]
min_lr = TRAIN_CONFIG["min_lr"]
warmup_steps = TRAIN_CONFIG["warmup_steps"]
grad_clip_val = TRAIN_CONFIG["grad_clip_val"]

path = PATHS["train_tokens"]

#----------------------------------------------------------------

device = get_device()

text = np.load(path)

model = Transformer_LM(
    vocab_size = vocab_size,
    context_length = context_length,
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    d_ff = d_ff,
    rope_theta  = rope_theta
).to(device)

optimizer = AdamW(
    params = model.parameters(),
    lr = TRAIN_CONFIG["lr"],
    betas = TRAIN_CONFIG["betas"],
    weight_decay  = TRAIN_CONFIG["weight_decay"],
    eps = TRAIN_CONFIG["eps"]
)

model.train()

train_data = text[:len(text)*9//10]
val_data = text[len(text)*9//10:]

for step in range(max_steps):
    optimizer.zero_grad()

    lr = get_lr_cos_schedule(
        t=step,
        a_max=max_lr,
        a_min=min_lr,
        T_w=warmup_steps,
        T_c=max_steps
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x,y = get_batch(train_data,batch_size=batch_size,context_length=context_length,device=device)

    logits = model(x)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = y.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    loss.backward()

    gradient_clipping(model.parameters(), max_l2_norm=grad_clip_val)

    optimizer.step()

    if step % 10 == 0 or step == max_steps - 1:
        with torch.no_grad():
            test_x,test_y = get_batch(val_data,batch_size=batch_size,context_length=context_length,device=device)
            logits = model(test_x)
            
            test_logits_flat = logits.view(-1, vocab_size)
            test_targets_flat = test_y.view(-1)
            test_loss = cross_entropy(test_logits_flat, test_targets_flat)

            print(f"Step {step:3d}/{max_steps} | train: {loss.item():.4f} | val: {test_loss:.4f}")


save_checkpoint(model, optimizer, iteration=max_steps, out=PATHS["checkpoint"])