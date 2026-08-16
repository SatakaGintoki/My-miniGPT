import math
import time
import os
import csv
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
CHECKPOINT_EVERY = 500  # save a recoverable checkpoint this often

# 新一次训练不覆盖原来的 checkpoint.pt（那份还能用来生成）
CKPT_OUT = "checkpoints/run170k.pt"
METRICS = "results/metrics_170k.csv"

#----------------------------------------------------------------

device = get_device()

text = np.load(path)

model_raw = Transformer_LM(
    vocab_size = vocab_size,
    context_length = context_length,
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    d_ff = d_ff,
    rope_theta  = rope_theta
).to(device)

# fp16 compute on GPU (autocast + GradScaler). No-op on CPU.
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# torch.compile gives ~2x on this small model; falls back if it fails.
# Checkpoints must be saved from `model_raw` (the compiled wrapper prefixes
# state-dict keys with `_orig_mod.`, which breaks generate.py's load_state_dict).
model = model_raw
try:
    model = torch.compile(model)
    print("torch.compile enabled", flush=True)
except Exception as e:
    print(f"torch.compile unavailable, continuing uncompiled: {e}", flush=True)

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

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
with open(METRICS, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=["step", "train", "val", "lr", "elapsed_sec"]).writeheader()

start_step = 0
start_t = time.time()
last_save = -1

for step in range(start_step, max_steps):
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

    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x)

    # compute loss in fp32 to avoid overflow in the hand-written cross_entropy
    logits_flat = logits.view(-1, vocab_size).float()
    targets_flat = y.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    gradient_clipping(model.parameters(), max_l2_norm=grad_clip_val)

    scaler.step(optimizer)
    scaler.update()

    if step % 10 == 0 or step == max_steps - 1:
        with torch.no_grad():
            test_x,test_y = get_batch(val_data,batch_size=batch_size,context_length=context_length,device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(test_x)

            test_logits_flat = logits.view(-1, vocab_size).float()
            test_targets_flat = test_y.view(-1)
            test_loss = cross_entropy(test_logits_flat, test_targets_flat)

        elapsed = time.time() - start_t
        sps = (step - start_step) / max(elapsed, 1e-9)
        eta_min = (max_steps - step) / max(sps, 1e-9) / 60
        print(f"Step {step:5d}/{max_steps} | train: {loss.item():.4f} | val: {test_loss.item():.4f} | lr: {lr:.2e} | {sps:.1f} sps | ETA {eta_min:.0f}min", flush=True)
        with open(METRICS, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["step", "train", "val", "lr", "elapsed_sec"]).writerow(
                {
                    "step": step,
                    "train": f"{loss.item():.6f}",
                    "val": f"{test_loss.item():.6f}",
                    "lr": f"{lr:.8e}",
                    "elapsed_sec": f"{elapsed:.1f}",
                }
            )

    if step % CHECKPOINT_EVERY == 0 and step > last_save:
        save_checkpoint(model_raw, optimizer, iteration=step, out=CKPT_OUT)
        last_save = step

save_checkpoint(model_raw, optimizer, iteration=max_steps, out=CKPT_OUT)
print(f"done in {(time.time()-start_t)/60:.1f} min", flush=True)
