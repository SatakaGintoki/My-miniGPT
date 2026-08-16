"""Quick sanity check on the encoded train_tokens.npy before training."""
import numpy as np

a = np.load("data/train_tokens.npy", mmap_mode="r")
print("dtype:", a.dtype)
print("len:", len(a))
print("max id:", a.max(), "(vocab_size 10000, so uint16 is fine:", a.max() < 10000, ")")
print("<|endoftext|> (id 256) count:", int((a == 256).sum()))
print("first 20:", a[:20].tolist())
print("last 20:", a[-20:].tolist())
print("decode-check non-empty, avg tokens/story:", int((a == 256).sum()), "special tokens across corpus")
