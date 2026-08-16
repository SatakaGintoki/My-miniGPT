"""Stream-encode the full corpus into data/train_tokens.npy (uint16).

Reuses the existing BPE vocab/merges (trained on a 50MB sample — the same
distribution, so no need to retrain on the full corpus, which would OOM).

Memory-safe design:
  * The file is read once to find chunk boundaries (at ASCII newlines, so a
    boundary never splits a word or a multi-byte UTF-8 char).
  * Each chunk is encoded in a worker process and written straight to a raw
    uint16 file (no giant Python int list held in RAM).
  * The chunk files are then assembled into a .npy via a memory-mapped array.

Usage:  python prepare_data.py            (uses os.cpu_count() workers)
        python prepare_data.py 8          (force 8 workers)
"""
import os
import sys
import time
import pickle
import array
import regex as re
import numpy as np
from multiprocessing import Pool

from src.my_tokenizer import Tokenizer
from src.pretokenization_example import find_chunk_boundaries
from config import TOKENIZER_CONFIG, PATHS

CORPUS = PATHS["corpus"]
OUT = PATHS["train_tokens"]
SPECIAL_TOKENS = TOKENIZER_CONFIG["special_tokens"]

# Load the BPE trained on the 50MB sample.
with open(PATHS["vocab"], "rb") as f:
    VOCAB = pickle.load(f)
with open(PATHS["merges"], "rb") as f:
    MERGES = pickle.load(f)

WORKDIR = os.path.join(os.path.dirname(OUT), "enc_chunks")
os.makedirs(WORKDIR, exist_ok=True)

# The word pattern (identical to the Tokenizer's). A single combined regex
# scans the chunk once instead of the Tokenizer's split-then-findall (2 passes),
# and keeps <|endoftext|> whole (it must match before the word pattern).
_WORD_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
if SPECIAL_TOKENS:
    _SPEC_ALT = "|".join(re.escape(t) for t in SPECIAL_TOKENS)
else:
    _SPEC_ALT = r"(?!)"  # never matches when there are no special tokens
_COMBINED = re.compile("(" + _SPEC_ALT + ")|(" + _WORD_PATTERN + ")")


def _encode_chunk(args):
    start, end, chunk_id = args
    with open(CORPUS, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    tok = Tokenizer(vocab=VOCAB, merges=MERGES, special_tokens=SPECIAL_TOKENS)

    out = array.array("H")  # uint16
    for m in _COMBINED.finditer(text):
        if m.group(1) is not None:
            # special token (<|endoftext|>); kept whole
            out.append(tok.bytes_to_id[m.group(1).encode("utf-8")])
        else:
            word = m.group(2)
            # byte-level vocab is the identity (id == byte value), so the
            # initial ids are just the word's bytes — no dict lookups needed.
            out.extend(tok._merge_ids(list(word.encode("utf-8"))))

    chunk_path = os.path.join(WORKDIR, f"chunk_{chunk_id:05d}.u16")
    with open(chunk_path, "wb") as f:
        out.tofile(f)
    return len(out)


def main():
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() or 8

    t0 = time.time()
    with open(CORPUS, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"\n")

    tasks = [
        (boundaries[i], boundaries[i + 1], i)
        for i in range(len(boundaries) - 1)
    ]
    print(f"chunking: {len(tasks)} chunks, encoding with {num_processes} workers...")

    with Pool(processes=num_processes) as pool:
        counts = pool.map(_encode_chunk, tasks)

    total = sum(counts)
    print(f"encoded {total} tokens in {time.time() - t0:.0f}s")

    # Assemble chunk files into a single uint16 .npy (bounded memory).
    if os.path.exists(OUT):
        os.remove(OUT)
    mm = np.lib.format.open_memmap(OUT, dtype=np.uint16, mode="w+", shape=(total,))
    offset = 0
    for i, c in enumerate(counts):
        chunk_path = os.path.join(WORKDIR, f"chunk_{i:05d}.u16")
        chunk = np.fromfile(chunk_path, dtype=np.uint16)
        assert len(chunk) == c
        mm[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
        os.remove(chunk_path)
    mm.flush()
    del mm
    try:
        os.rmdir(WORKDIR)
    except OSError:
        pass

    print(f"saved {total} tokens -> {OUT} ({total * 2 / 1e9:.2f} GB uint16)")
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
