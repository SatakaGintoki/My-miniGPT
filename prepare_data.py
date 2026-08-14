import os
import numpy as np
import pickle

from src.my_bpe import train_bpe
from src.my_tokenizer import Tokenizer
from config import TOKENIZER_CONFIG, PATHS

path = PATHS["corpus"]

vocab_size = TOKENIZER_CONFIG["vocab_size"]
special_tokens = TOKENIZER_CONFIG["special_tokens"]

vocab, merges = train_bpe(path, vocab_size=vocab_size,special_tokens=special_tokens)

tokenizer  = Tokenizer(vocab=vocab,merges=merges,special_tokens=special_tokens)

text = open(path, encoding="utf-8").read()

train_tokens = np.array(tokenizer.encode(text=text))

pickle.dump(vocab, open(PATHS["vocab"], "wb"))
pickle.dump(merges, open(PATHS["merges"], "wb"))

np.save(PATHS["train_tokens"], train_tokens)